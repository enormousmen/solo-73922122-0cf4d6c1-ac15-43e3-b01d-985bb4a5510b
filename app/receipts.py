"""留证回执：归档模型状态与首次推理结果，支持凭号复算。

设计要点：
- 留证 = 模型状态快照（权重/列顺序/推理精度）+ 原始读数 + 首次推理结果；
- 模型状态按内容指纹（SHA-256）命名归档，多张回执共用同一状态时
  磁盘上只保留一份；
- 回执按归档成功的先后保留最近 N 张（默认 32），超额淘汰最旧回执；
- 状态文件按"保留回执引用计数"垃圾回收：淘汰最后一张引用它的回执时
  才清理；
- 已受理的复算对状态文件持有活动引用，淘汰遇到活动引用时延迟清理，
  保证复算一定算完；
- 归档目录持久化在磁盘（启动时扫描重建索引），覆盖检查点或重启后
  仍可凭号复算；
- 归档失败抛 OSError 由上层报错，不影响已有回执的复算。

目录布局：
    <archive_dir>/
        receipts/R-000001.json   # 回执（原始读数 + 首次结果 + 状态指纹）
        states/<state_id>.pt     # 模型状态（权重/列顺序/精度，按指纹去重）
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any

import torch

from .checkpoint import ModelSnapshot, resolve_dtype, snapshot_fingerprint
from .model import TwoLayerClassifier
from .service import predict_with

RECEIPT_ID_PREFIX = "R-"


class ReceiptNotFound(KeyError):
    """回执不存在或已被淘汰。"""


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """临时文件 + rename 的原子写，失败不留下半成品。"""
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _atomic_torch_save(blob: dict, path: str) -> None:
    directory = os.path.dirname(path)
    # PyTorchFileWriter 要求文件名带扩展名，临时文件统一加 .pt 后缀
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".pt", dir=directory)
    os.close(fd)
    try:
        torch.save(blob, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def predict_with_snapshot(
    snapshot: ModelSnapshot, instances: list[dict]
) -> list[dict]:
    """用快照重建临时模型并推理（与在线预测同一条推理路径）。"""
    model = TwoLayerClassifier(
        input_dim=len(snapshot.feature_names),
        hidden_dim=snapshot.hidden_dim,
        num_classes=snapshot.num_classes,
    )
    model.load_state_dict(snapshot.model_state)
    model.to(dtype=snapshot.dtype)
    return predict_with(model, snapshot.feature_names, snapshot.dtype, instances)


def load_archived_state(state_path: str) -> dict[str, Any]:
    """从归档文件读出模型状态字典。"""
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"归档状态缺失: {state_path}")
    return torch.load(state_path, map_location="cpu", weights_only=False)


def predict_with_archived_state(
    blob: dict[str, Any], instances: list[dict]
) -> list[dict]:
    """用归档的权重/列顺序/精度重建模型并复算。"""
    feature_names = list(blob["feature_names"])
    dtype = resolve_dtype(blob.get("dtype"))
    model = TwoLayerClassifier(
        input_dim=len(feature_names),
        hidden_dim=int(blob["hidden_dim"]),
        num_classes=int(blob["num_classes"]),
    )
    model.load_state_dict(blob["model_state"])
    model.to(dtype=dtype)
    return predict_with(model, feature_names, dtype, instances)


class ReceiptStore:
    """回执与模型状态归档。

    线程安全：索引变更与归档/淘汰的文件操作在同一把锁内串行；
    仅复算时的状态文件读取放在锁外（由活动引用计数保护不被清理）。
    """

    def __init__(self, archive_dir: str, keep: int = 32):
        if keep < 1:
            raise ValueError("keep 必须 >= 1")
        self._dir = os.path.abspath(archive_dir)
        self._keep = int(keep)
        self._lock = threading.Lock()
        self._order: list[str] = []             # 按归档成功先后排列的回执号
        self._refs: dict[str, int] = {}         # state_id -> 保留回执引用数
        self._active: dict[str, int] = {}       # state_id -> 进行中的复算数
        self._pending_delete: set[str] = set()  # 引用归零但复算在进行的状态
        self._next_seq = 1
        self._scan_existing()

    # ---- 基本属性与路径 ------------------------------------------
    @property
    def keep(self) -> int:
        return self._keep

    @property
    def archive_dir(self) -> str:
        return self._dir

    def _receipts_dir(self) -> str:
        return os.path.join(self._dir, "receipts")

    def _states_dir(self) -> str:
        return os.path.join(self._dir, "states")

    def _receipt_path(self, receipt_id: str) -> str:
        return os.path.join(self._receipts_dir(), f"{receipt_id}.json")

    def _state_path(self, state_id: str) -> str:
        return os.path.join(self._states_dir(), f"{state_id}.pt")

    def _ensure_dirs(self) -> None:
        os.makedirs(self._receipts_dir(), exist_ok=True)
        os.makedirs(self._states_dir(), exist_ok=True)

    @staticmethod
    def _read_receipt(path: str) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ---- 启动扫描（重启/覆盖检查点后恢复索引） --------------------
    def _scan_existing(self) -> None:
        receipts_dir = self._receipts_dir()
        if not os.path.isdir(receipts_dir):
            return
        found: list[tuple[int, str, str]] = []
        for name in os.listdir(receipts_dir):
            if not name.endswith(".json"):
                continue
            path = os.path.join(receipts_dir, name)
            try:
                data = self._read_receipt(path)
                found.append(
                    (int(data["seq"]), str(data["receipt_id"]), str(data["state_id"]))
                )
            except (OSError, ValueError, KeyError, TypeError):
                # 损坏文件：改名隔离，不计入索引
                try:
                    os.replace(path, path + ".corrupt")
                except OSError:
                    pass
        found.sort(key=lambda item: item[0])
        with self._lock:
            for seq, receipt_id, state_id in found:
                self._order.append(receipt_id)
                self._refs[state_id] = self._refs.get(state_id, 0) + 1
                self._next_seq = max(self._next_seq, seq + 1)
            self._evict_locked()  # 若 keep 调小，启动即按新上限淘汰
            self._gc_orphan_states_locked()

    def _gc_orphan_states_locked(self) -> None:
        """清理没有任何保留回执引用的状态文件（崩溃残留等）。"""
        states_dir = self._states_dir()
        if not os.path.isdir(states_dir):
            return
        for name in os.listdir(states_dir):
            if not name.endswith(".pt"):
                continue
            state_id = name[: -len(".pt")]
            if self._refs.get(state_id, 0) == 0 and self._active.get(state_id, 0) == 0:
                try:
                    os.remove(os.path.join(states_dir, name))
                except OSError:
                    pass

    # ---- 归档（留证） ---------------------------------------------
    def archive(
        self,
        snapshot: ModelSnapshot,
        instances: list[dict],
        predictions: list[dict],
    ) -> dict[str, Any]:
        """归档一张回执并触发淘汰；失败抛 OSError，不影响已有回执。"""
        state_id = snapshot_fingerprint(snapshot)
        with self._lock:
            self._ensure_dirs()

            # 1. 模型状态按指纹去重归档：已存在则共享，不重复写
            state_path = self._state_path(state_id)
            wrote_state = False
            if not os.path.exists(state_path):
                blob = snapshot.archived_state_dict()
                blob["state_id"] = state_id
                _atomic_torch_save(blob, state_path)
                wrote_state = True

            # 2. 写回执（原始读数 + 首次结果 + 状态指纹）
            receipt_id = f"{RECEIPT_ID_PREFIX}{self._next_seq:06d}"
            receipt: dict[str, Any] = {
                "receipt_id": receipt_id,
                "seq": self._next_seq,
                "state_id": state_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dtype": str(snapshot.dtype).replace("torch.", ""),
                "feature_names": list(snapshot.feature_names),
                "instances": [dict(inst) for inst in instances],
                "predictions": predictions,
            }
            try:
                _atomic_write_bytes(
                    self._receipt_path(receipt_id),
                    json.dumps(receipt, ensure_ascii=False, indent=2).encode("utf-8"),
                )
            except OSError:
                # 回执写失败：本次新写且无人引用的状态文件一并回滚
                if wrote_state and self._refs.get(state_id, 0) == 0:
                    try:
                        os.remove(state_path)
                    except OSError:
                        pass
                raise

            # 3. 登记顺序（归档成功才算数）并淘汰超额回执
            self._next_seq += 1
            self._order.append(receipt_id)
            self._refs[state_id] = self._refs.get(state_id, 0) + 1
            self._evict_locked()
            return receipt

    # ---- 淘汰与状态垃圾回收 ---------------------------------------
    def _evict_locked(self) -> None:
        """只保留最近 keep 张回执；引用归零的状态文件随之清理。"""
        while len(self._order) > self._keep:
            victim = self._order.pop(0)
            state_id = None
            try:
                state_id = str(self._read_receipt(self._receipt_path(victim))["state_id"])
            except (OSError, ValueError, KeyError, TypeError):
                pass
            try:
                os.remove(self._receipt_path(victim))
            except OSError:
                pass
            if state_id is None:
                continue
            remaining = self._refs.get(state_id, 1) - 1
            if remaining > 0:
                self._refs[state_id] = remaining
                continue
            self._refs.pop(state_id, None)
            self._gc_state_locked(state_id)

    def _gc_state_locked(self, state_id: str) -> None:
        """引用归零时清理状态文件；有复算在进行则延迟到 release。"""
        if self._active.get(state_id, 0) > 0:
            self._pending_delete.add(state_id)
            return
        try:
            os.remove(self._state_path(state_id))
        except OSError:
            pass  # 清理失败不致命，启动扫描会再尝试

    # ---- 复算受理（淘汰保护） -------------------------------------
    def acquire(self, receipt_id: str) -> tuple[dict[str, Any], str]:
        """受理一次复算：返回（回执内容, 状态文件路径）。

        受理后即持有该状态的活动引用：即使回执随后被淘汰，
        状态文件也会保留到 release，复算一定算完。
        """
        with self._lock:
            if receipt_id not in self._order:
                raise ReceiptNotFound(receipt_id)
            try:
                receipt = self._read_receipt(self._receipt_path(receipt_id))
            except (OSError, ValueError) as exc:
                raise ReceiptNotFound(receipt_id) from exc
            state_id = str(receipt["state_id"])
            self._active[state_id] = self._active.get(state_id, 0) + 1
            return receipt, self._state_path(state_id)

    def release(self, state_id: str) -> None:
        """复算结束：归还活动引用，必要时完成被延迟的清理。"""
        with self._lock:
            remaining = self._active.get(state_id, 0) - 1
            if remaining > 0:
                self._active[state_id] = remaining
                return
            self._active.pop(state_id, None)
            if state_id in self._pending_delete and self._refs.get(state_id, 0) == 0:
                self._pending_delete.discard(state_id)
                try:
                    os.remove(self._state_path(state_id))
                except OSError:
                    pass

    # ---- 查询 -----------------------------------------------------
    def list_receipts(self) -> list[dict[str, Any]]:
        """按归档先后列出当前保留的回执摘要。"""
        with self._lock:
            out: list[dict[str, Any]] = []
            for receipt_id in self._order:
                try:
                    data = self._read_receipt(self._receipt_path(receipt_id))
                except (OSError, ValueError):
                    continue
                out.append(
                    {
                        "receipt_id": data["receipt_id"],
                        "seq": data["seq"],
                        "state_id": data["state_id"],
                        "created_at": data["created_at"],
                        "dtype": data["dtype"],
                        "num_instances": len(data.get("instances", [])),
                    }
                )
            return out

    def get(self, receipt_id: str) -> dict[str, Any]:
        """读取单张保留回执的完整内容。"""
        with self._lock:
            if receipt_id not in self._order:
                raise ReceiptNotFound(receipt_id)
            try:
                return self._read_receipt(self._receipt_path(receipt_id))
            except (OSError, ValueError) as exc:
                raise ReceiptNotFound(receipt_id) from exc
