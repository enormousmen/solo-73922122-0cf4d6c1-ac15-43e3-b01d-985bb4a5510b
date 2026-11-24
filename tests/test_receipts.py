"""留证预测与凭号复算的端到端 / 单元测试。

覆盖需求硬性条款：
1. 留证预测返回回执号，凭号复算与首次结果一致（概率差 <= 1e-6，
   类别与补零列相同）；
2. 续训/扩列改变模型后，复算仍用当时的权重与列顺序；
3. 覆盖检查点并重启（重建服务）后仍可凭号复算；
4. 回执按归档成功先后只保留最近 N 张，超额淘汰；
5. 共用模型状态只归档一份，最后一张引用的回执淘汰时清理；
6. 归档失败时报错（500），已有回执照常可复算；
7. 已受理的复算遇到回执淘汰也算完（状态文件延迟清理）；
8. 留证取齐快照后，续训/扩列无需等待本次推理和归档。
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import shutil
import time

import pytest
from fastapi.testclient import TestClient

SRC_CKPT = os.path.join("checkpoints", "model.pt")

ROWS = [
    {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
    {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9},
]
# 缺 current 列的样本：用于核对补零列在复算中保持一致
ROWS_MISSING = [
    {"temp": 0.5, "pressure": -0.3, "rpm": 1.2},
]
TRAIN_ROWS = [
    {"temp": 0.9, "pressure": -0.8, "rpm": 1.5, "current": -0.2},
    {"temp": -0.7, "pressure": 1.2, "rpm": -0.1, "current": 0.6},
]


@pytest.fixture()
def open_client(tmp_path, monkeypatch):
    """客户端工厂：检查点复制到 tmp（可安全覆盖），归档目录独立。

    每次 _open() 重新加载 app.main 并重建 TestClient，等价于一次重启。
    """
    monkeypatch.setenv("TORCH_NUM_THREADS", "1")
    ckpt = str(tmp_path / "model.pt")
    shutil.copyfile(SRC_CKPT, ckpt)
    monkeypatch.setenv("CHECKPOINT_PATH", ckpt)

    @contextlib.contextmanager
    def _open(keep: int = 32, archive_name: str = "archive"):
        monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / archive_name))
        monkeypatch.setenv("RECEIPT_KEEP", str(keep))
        import app.main as main

        importlib.reload(main)
        with TestClient(main.app) as c:
            yield c, main

    return _open


def _attest(client, rows=ROWS):
    resp = client.post("/predict/attested", json={"instances": rows})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _train(client):
    resp = client.post(
        "/train", json={"instances": TRAIN_ROWS, "labels": [0, 1]}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---- 条款 1：留证 -> 复算一致 --------------------------------------


def test_attested_predict_returns_receipt_and_replays_identically(open_client):
    with open_client() as (c, _):
        body = _attest(c)
        rid = body["receipt_id"]
        assert rid.startswith("R-")
        assert len(body["predictions"]) == 2

        resp = c.post(f"/receipts/{rid}/replay")
        assert resp.status_code == 200, resp.text
        rb = resp.json()
        assert rb["receipt_id"] == rid
        assert rb["max_probability_diff"] <= 1e-6
        assert rb["labels_match"] is True
        assert rb["zero_filled_match"] is True
        # 确定性复算：与首次结果逐位一致
        assert rb["predictions"] == rb["first_predictions"]


def test_replay_preserves_zero_filled_columns(open_client):
    with open_client() as (c, _):
        body = _attest(c, ROWS_MISSING)
        rid = body["receipt_id"]
        assert body["predictions"][0]["zero_filled_columns"] == ["current"]
        rb = c.post(f"/receipts/{rid}/replay").json()
        assert rb["zero_filled_match"] is True
        assert rb["predictions"][0]["zero_filled_columns"] == ["current"]


# ---- 条款 2：续训/扩列后复算仍用当时状态 ----------------------------


def test_replay_uses_historical_weights_after_train(open_client):
    with open_client() as (c, _):
        first = _attest(c)
        rid = first["receipt_id"]
        _train(c)  # 权重已变化

        rb = c.post(f"/receipts/{rid}/replay").json()
        assert rb["max_probability_diff"] <= 1e-6
        assert rb["labels_match"] is True
        assert rb["predictions"] == first["predictions"]

        # 旁证：当前在线预测确实已偏离留证时的结果
        now = c.post("/predict", json={"instances": ROWS}).json()["predictions"]
        drift = max(
            abs(a - b)
            for p_now, p_first in zip(now, first["predictions"])
            for a, b in zip(p_now["probabilities"], p_first["probabilities"])
        )
        assert drift > 1e-6, "续训后在线预测未发生变化，对照无效"


def test_replay_uses_historical_columns_after_expand(open_client):
    with open_client() as (c, _):
        first = _attest(c, ROWS + ROWS_MISSING)
        rid = first["receipt_id"]
        assert first["feature_names"] == ["temp", "pressure", "rpm", "current"]

        resp = c.post("/expand", json={"new_feature_names": ["vibration"]})
        assert resp.status_code == 200, resp.text

        rb = c.post(f"/receipts/{rid}/replay").json()
        # 复算用当时的 4 列与当时精度，类别/补零列/概率与首次一致
        assert rb["feature_names"] == ["temp", "pressure", "rpm", "current"]
        assert rb["max_probability_diff"] <= 1e-6
        assert rb["labels_match"] is True
        assert rb["zero_filled_match"] is True
        assert rb["predictions"][2]["zero_filled_columns"] == ["current"]


# ---- 条款 3：覆盖检查点 + 重启后凭号复算 ----------------------------


def test_replay_survives_checkpoint_overwrite_and_restart(open_client):
    with open_client() as (c1, _):
        first = _attest(c1)
        rid = first["receipt_id"]
        _train(c1)
        # 覆盖保存到 CHECKPOINT_PATH（重启后将加载新权重）
        resp = c1.post("/save", json={})
        assert resp.status_code == 200 and resp.json()["ok"] is True

    # 重启：同一归档目录、同一（已覆盖的）检查点
    with open_client() as (c2, _):
        assert c2.get("/model").json()["optimizer_step"] == 201
        # 旧回执仍在（重启扫描恢复）
        listing = c2.get("/receipts").json()
        assert [r["receipt_id"] for r in listing["receipts"]] == [rid]
        # 凭号复算仍用留证时的旧权重
        rb = c2.post(f"/receipts/{rid}/replay").json()
        assert rb["max_probability_diff"] <= 1e-6
        assert rb["labels_match"] is True
        assert rb["zero_filled_match"] is True
        assert rb["predictions"] == first["predictions"]


# ---- 条款 4：按归档成功先后只保留最近 N 张 --------------------------


def test_eviction_keeps_only_most_recent_receipts(open_client):
    with open_client(keep=3) as (c, main):
        rids = [_attest(c)["receipt_id"] for _ in range(5)]

        listing = c.get("/receipts").json()
        assert listing["keep"] == 3
        assert listing["count"] == 3
        assert [r["receipt_id"] for r in listing["receipts"]] == rids[2:]

        # 最旧两张已淘汰：复算与查询均 404
        for rid in rids[:2]:
            assert c.post(f"/receipts/{rid}/replay").status_code == 404
            assert c.get(f"/receipts/{rid}").status_code == 404
        # 最近三张可正常复算
        for rid in rids[2:]:
            assert c.post(f"/receipts/{rid}/replay").status_code == 200

        # 未续训：5 张回执共用同一状态，磁盘上只有一份状态文件
        states = os.listdir(os.path.join(main.ARCHIVE_DIR, "states"))
        assert len([s for s in states if s.endswith(".pt")]) == 1


# ---- 条款 5：状态去重与最后引用淘汰时清理 ---------------------------


def test_shared_state_archived_once_and_gc_after_last_reference(open_client):
    with open_client(keep=2) as (c, main):
        states_dir = os.path.join(main.ARCHIVE_DIR, "states")

        def state_files():
            return sorted(
                f for f in os.listdir(states_dir) if f.endswith(".pt")
            )

        r1 = _attest(c)["receipt_id"]      # 状态 A
        _train(c)
        r2 = _attest(c)["receipt_id"]      # 状态 B
        _train(c)
        r3 = _attest(c)["receipt_id"]      # 状态 C，淘汰 r1 -> A 清理
        assert len(state_files()) == 2     # B、C 各一份

        _train(c)
        r4 = _attest(c)["receipt_id"]      # 状态 D，淘汰 r2 -> B 清理
        files = state_files()
        assert len(files) == 2             # C、D

        # 被淘汰的 r1/r2 不可复算；保留中的 r3/r4 正常
        assert c.post(f"/receipts/{r1}/replay").status_code == 404
        assert c.post(f"/receipts/{r2}/replay").status_code == 404
        assert c.post(f"/receipts/{r3}/replay").status_code == 200
        assert c.post(f"/receipts/{r4}/replay").status_code == 200

        # 共用一个状态的连续留证只占一份归档
        r5 = _attest(c)["receipt_id"]      # 未续训，与 r4 同状态 D；淘汰 r3 -> C 清理
        assert len(state_files()) == 1     # 只剩 D（r4/r5 共享）
        assert c.post(f"/receipts/{r3}/replay").status_code == 404
        assert c.post(f"/receipts/{r4}/replay").status_code == 200
        assert c.post(f"/receipts/{r5}/replay").status_code == 200


# ---- 条款 6：归档失败时报错，已有回执照常可用 ------------------------


def test_archive_failure_reported_and_existing_receipts_usable(open_client):
    with open_client() as (c, main):
        rid = _attest(c)["receipt_id"]       # R-000001，状态 A
        _train(c)                            # 使下次留证产生新状态 B

        # 故障注入：把下一张回执的目标路径占位成目录，
        # 原子写 rename 必失败（不触碰已有回执与状态文件）
        receipts_dir = os.path.join(main.ARCHIVE_DIR, "receipts")
        os.makedirs(os.path.join(receipts_dir, "R-000002.json"))

        resp = c.post("/predict/attested", json={"instances": ROWS})
        assert resp.status_code == 500
        assert "归档失败" in resp.json()["detail"]

        # 归档失败回滚干净：磁盘上仍只有状态 A 一份
        states_dir = os.path.join(main.ARCHIVE_DIR, "states")
        assert len([f for f in os.listdir(states_dir) if f.endswith(".pt")]) == 1

        # 已有回执照常可复算、可查询
        rb = c.post(f"/receipts/{rid}/replay")
        assert rb.status_code == 200
        assert rb.json()["max_probability_diff"] <= 1e-6
        assert c.get("/receipts").json()["count"] == 1
        # 普通预测/训练也不受影响
        assert c.post("/predict", json={"instances": ROWS}).status_code == 200


# ---- 条款 7：已受理的复算遇到淘汰也算完（store 级） ------------------


def test_accepted_replay_survives_eviction(tmp_path):
    from app.checkpoint import load_checkpoint
    from app.receipts import (
        ReceiptStore,
        load_archived_state,
        predict_with_archived_state,
        predict_with_snapshot,
    )
    from app.service import ModelService

    store = ReceiptStore(str(tmp_path / "arc"), keep=1)
    svc = ModelService(load_checkpoint(SRC_CKPT))

    snap1 = svc.snapshot()
    preds1 = predict_with_snapshot(snap1, ROWS)
    r1 = store.archive(snap1, ROWS, preds1)

    # 受理复算（持有状态活动引用），但暂不读取状态文件
    receipt, state_path = store.acquire(r1["receipt_id"])
    assert os.path.exists(state_path)

    # 续训后再留证 -> keep=1，r1 被淘汰，其状态引用归零
    svc.train_step(TRAIN_ROWS, [0, 1])
    snap2 = svc.snapshot()
    r2 = store.archive(snap2, ROWS, predict_with_snapshot(snap2, ROWS))
    assert [r["receipt_id"] for r in store.list_receipts()] == [r2["receipt_id"]]

    # 已受理的复算：状态文件仍在，复算照常算完且与首次一致
    assert os.path.exists(state_path)
    blob = load_archived_state(state_path)
    replayed = predict_with_archived_state(blob, receipt["instances"])
    assert replayed == receipt["predictions"]

    # 复算结束释放引用后，被延迟的清理完成
    store.release(r1["state_id"])
    assert not os.path.exists(state_path)


# ---- 条款 8：取齐快照后续训/扩列无需等待推理和归档 -------------------


def test_attest_does_not_block_train_during_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_NUM_THREADS", "1")
    ckpt = str(tmp_path / "model.pt")
    shutil.copyfile(SRC_CKPT, ckpt)
    monkeypatch.setenv("CHECKPOINT_PATH", ckpt)
    monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path / "archive"))
    monkeypatch.setenv("RECEIPT_KEEP", "32")
    import app.main as main

    importlib.reload(main)

    async def scenario():
        async with main.lifespan(main.app):
            svc = main.service
            store = main.receipt_store

            # 人为拖慢归档，放大“推理+归档”窗口
            orig_archive = store.archive

            def slow_archive(*args, **kwargs):
                time.sleep(0.5)
                return orig_archive(*args, **kwargs)

            monkeypatch.setattr(store, "archive", slow_archive)

            from app.schemas import PredictRequest

            req = PredictRequest(instances=ROWS)
            start = time.monotonic()
            task = asyncio.create_task(main.predict_attested(req))
            await asyncio.sleep(0.05)  # 让留证先取齐快照
            async with svc.lock:  # 模拟续训/扩列取锁
                waited = time.monotonic() - start
            # 锁只被快照占用：远小于归档耗时 0.5s
            assert waited < 0.3, f"续训/扩列被留证阻塞了 {waited:.3f}s"
            result = await task
            assert result["receipt_id"].startswith("R-")

    asyncio.run(scenario())


# ---- 查询与校验 ------------------------------------------------------


def test_receipt_list_and_detail(open_client):
    with open_client() as (c, _):
        rid = _attest(c)["receipt_id"]

        listing = c.get("/receipts").json()
        assert listing["keep"] == 32
        assert listing["count"] == 1
        item = listing["receipts"][0]
        assert item["receipt_id"] == rid
        assert item["num_instances"] == 2
        assert item["dtype"] == "float32"

        detail = c.get(f"/receipts/{rid}").json()
        assert detail["receipt_id"] == rid
        assert detail["instances"] == ROWS
        assert len(detail["predictions"]) == 2
        assert detail["feature_names"] == ["temp", "pressure", "rpm", "current"]


def test_unknown_receipt_returns_404(open_client):
    with open_client() as (c, _):
        assert c.get("/receipts/R-999999").status_code == 404
        assert c.post("/receipts/R-999999/replay").status_code == 404


def test_attested_predict_validation(open_client):
    with open_client() as (c, _):
        # 空样本列表 -> 422（与 /predict 同一请求格式与校验）
        resp = c.post("/predict/attested", json={"instances": []})
        assert resp.status_code == 422
        # 校验失败不产生任何回执
        assert c.get("/receipts").json()["count"] == 0
