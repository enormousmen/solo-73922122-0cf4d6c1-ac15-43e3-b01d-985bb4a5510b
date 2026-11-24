"""单作业分类模型 API（FastAPI + CPU PyTorch）。

启动时从 CHECKPOINT_PATH 加载含 Adam 状态的检查点。
环境变量：
    CHECKPOINT_PATH   检查点文件（默认 checkpoints/model.pt）
    MODEL_DTYPE       float32（默认）或 float64
    TORCH_NUM_THREADS CPU 线程数（默认 1，保证可复现的逐位一致行为）
    ARCHIVE_DIR       留证归档目录（默认 archive/）
    RECEIPT_KEEP      回执保留张数（默认 32，按归档成功先后淘汰）
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

# 必须在 import torch 前设置线程数，保证 GEMM 路径稳定可复现
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("TORCH_NUM_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("TORCH_NUM_THREADS", "1"))

from fastapi import FastAPI, HTTPException

from .checkpoint import load_checkpoint
from .receipts import (
    ReceiptNotFound,
    ReceiptStore,
    load_archived_state,
    predict_with_archived_state,
    predict_with_snapshot,
)
from .schemas import (
    AttestedPredictResponse,
    ExpandRequest,
    ExpandResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReceiptDetailResponse,
    ReceiptListResponse,
    ReloadResponse,
    ReplayResponse,
    SaveRequest,
    SaveResponse,
    TrainRequest,
    TrainResponse,
)
from .service import ModelService

CHECKPOINT_PATH = os.path.abspath(
    os.environ.get("CHECKPOINT_PATH", "checkpoints/model.pt")
)
MODEL_DTYPE = "float64" if os.environ.get("MODEL_DTYPE", "float32") == "float64" else "float32"
ARCHIVE_DIR = os.path.abspath(os.environ.get("ARCHIVE_DIR", "archive"))
RECEIPT_KEEP = int(os.environ.get("RECEIPT_KEEP", "32"))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
torch.use_deterministic_algorithms(True)

_DTYPE = {"float32": torch.float32, "float64": torch.float64}[MODEL_DTYPE]

service: ModelService | None = None
receipt_store: ReceiptStore | None = None


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    global service, receipt_store
    try:
        bundle = load_checkpoint(CHECKPOINT_PATH, dtype=_DTYPE)
    except Exception as exc:  # 启动期失败直接暴露，避免空服务
        raise RuntimeError(f"加载检查点失败 {CHECKPOINT_PATH}: {exc}") from exc
    service = ModelService(bundle)
    # 扫描归档目录重建回执索引：覆盖检查点/重启后仍可凭号复算
    receipt_store = ReceiptStore(ARCHIVE_DIR, keep=RECEIPT_KEEP)
    yield


app = FastAPI(
    title="设备振动测点扩列续训 API",
    version="1.1.0",
    description="两层分类模型：按列名扩列（缺列补 0）、续训、原子保存、留证预测与凭号复算。",
    lifespan=lifespan,
)


def _svc() -> ModelService:
    if service is None:
        raise HTTPException(status_code=503, detail="服务尚未完成初始化")
    return service


def _store() -> ReceiptStore:
    if receipt_store is None:
        raise HTTPException(status_code=503, detail="服务尚未完成初始化")
    return receipt_store


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> dict:
    return _svc().info()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> dict:
    svc = _svc()
    try:
        preds = svc.predict(req.instances)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"feature_names": svc.bundle.feature_names, "predictions": preds}


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest) -> dict:
    svc = _svc()
    async with svc.lock:
        try:
            return svc.train_step(req.instances, req.labels)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/expand", response_model=ExpandResponse)
async def expand(req: ExpandRequest) -> dict:
    svc = _svc()
    async with svc.lock:
        try:
            return svc.expand(req.new_feature_names)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/save", response_model=SaveResponse)
async def save(req: SaveRequest) -> dict:
    svc = _svc()
    async with svc.lock:
        try:
            path = svc.save(req.path)
        except OSError as exc:
            # 保存失败：内存模型与原检查点均不受影响
            raise HTTPException(status_code=500, detail=f"保存失败: {exc}") from exc
    return {"path": path, "ok": True}


@app.post("/reload", response_model=ReloadResponse)
async def reload(path: str | None = None) -> dict:
    svc = _svc()
    async with svc.lock:
        try:
            return svc.reload(path)
        except (FileNotFoundError, OSError, RuntimeError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"重载失败: {exc}") from exc


# ---- 留证预测与凭号复算 -------------------------------------------


@app.post("/predict/attested", response_model=AttestedPredictResponse)
async def predict_attested(req: PredictRequest) -> dict:
    """留证预测：按原格式提交读数，返回分类结果和回执号。

    快照（取齐权重/列顺序/精度）在锁内完成；推理与归档在锁外进行，
    因此续训/扩列只需等待快照克隆，无需等待本次推理和归档。
    """
    svc = _svc()
    store = _store()
    async with svc.lock:
        snap = svc.snapshot()
    try:
        preds = predict_with_snapshot(snap, req.instances)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        receipt = await asyncio.to_thread(
            store.archive, snap, req.instances, preds
        )
    except OSError as exc:
        # 归档失败：报错；已有回执的复算不受影响
        raise HTTPException(status_code=500, detail=f"归档失败: {exc}") from exc
    return {
        "receipt_id": receipt["receipt_id"],
        "feature_names": list(snap.feature_names),
        "predictions": preds,
    }


@app.post("/receipts/{receipt_id}/replay", response_model=ReplayResponse)
async def replay_receipt(receipt_id: str) -> dict:
    """凭号复算：用归档的权重/列顺序/精度/原始读数重新推理。

    受理即对模型状态加活动引用：复算期间即使回执被淘汰也算完。
    """
    store = _store()
    try:
        receipt, state_path = store.acquire(receipt_id)
    except ReceiptNotFound:
        raise HTTPException(
            status_code=404, detail=f"回执不存在或已淘汰: {receipt_id}"
        ) from None
    state_id = str(receipt["state_id"])
    try:
        try:
            blob = await asyncio.to_thread(load_archived_state, state_path)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(
                status_code=500, detail=f"归档状态读取失败: {exc}"
            ) from exc
        replayed = predict_with_archived_state(blob, receipt["instances"])
    finally:
        store.release(state_id)

    first = receipt["predictions"]
    max_diff = 0.0
    labels_match = True
    zero_filled_match = True
    for f_item, r_item in zip(first, replayed):
        labels_match = labels_match and f_item["label"] == r_item["label"]
        zero_filled_match = zero_filled_match and (
            list(f_item["zero_filled_columns"]) == list(r_item["zero_filled_columns"])
        )
        for p_first, p_replay in zip(f_item["probabilities"], r_item["probabilities"]):
            max_diff = max(max_diff, abs(float(p_first) - float(p_replay)))
    return {
        "receipt_id": receipt_id,
        "state_id": state_id,
        "feature_names": list(receipt["feature_names"]),
        "dtype": receipt["dtype"],
        "predictions": replayed,
        "first_predictions": first,
        "max_probability_diff": max_diff,
        "labels_match": labels_match,
        "zero_filled_match": zero_filled_match,
    }


@app.get("/receipts", response_model=ReceiptListResponse)
def list_receipts() -> dict:
    store = _store()
    items = store.list_receipts()
    return {"keep": store.keep, "count": len(items), "receipts": items}


@app.get("/receipts/{receipt_id}", response_model=ReceiptDetailResponse)
def get_receipt(receipt_id: str) -> dict:
    store = _store()
    try:
        return store.get(receipt_id)
    except ReceiptNotFound:
        raise HTTPException(
            status_code=404, detail=f"回执不存在或已淘汰: {receipt_id}"
        ) from None
