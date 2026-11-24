"""FastAPI 请求/响应模型。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # 每条样本为 {列名: 数值}；模型需要但未提供的列自动补 0
    instances: list[dict[str, float]] = Field(min_length=1)


class PredictedItem(BaseModel):
    label: int
    score: float
    probabilities: list[float]
    zero_filled_columns: list[str]


class PredictResponse(BaseModel):
    feature_names: list[str]
    predictions: list[PredictedItem]


class TrainRequest(BaseModel):
    instances: list[dict[str, float]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)


class TrainResponse(BaseModel):
    step: int
    loss_before: float
    loss_after: float
    num_samples: int
    feature_names: list[str]


class ExpandRequest(BaseModel):
    new_feature_names: list[str] = Field(min_length=1)


class ExpandResponse(BaseModel):
    feature_names: list[str]
    input_dim: int
    added: list[str]


class ModelInfoResponse(BaseModel):
    feature_names: list[str]
    input_dim: int
    hidden_dim: int
    num_classes: int
    dtype: str
    optimizer_step: int
    source_path: str


class SaveRequest(BaseModel):
    path: str | None = None  # 为空则覆盖当前检查点路径


class SaveResponse(BaseModel):
    path: str
    ok: bool


class ReloadResponse(BaseModel):
    path: str
    feature_names: list[str]
    input_dim: int
    optimizer_step: int


class AttestedPredictResponse(BaseModel):
    # 留证预测：分类结果 + 回执号（凭号可复算）
    receipt_id: str
    feature_names: list[str]
    predictions: list[PredictedItem]


class ReplayResponse(BaseModel):
    # 凭号复算：用归档的权重/列顺序/精度/原始读数重新推理并与首次对照
    receipt_id: str
    state_id: str
    feature_names: list[str]
    dtype: str
    predictions: list[PredictedItem]  # 本次复算结果
    first_predictions: list[PredictedItem]  # 留证时的首次结果
    max_probability_diff: float  # 各类概率与首次的最大绝对差
    labels_match: bool
    zero_filled_match: bool


class ReceiptSummary(BaseModel):
    receipt_id: str
    seq: int
    state_id: str
    created_at: str
    dtype: str
    num_instances: int


class ReceiptListResponse(BaseModel):
    keep: int
    count: int
    receipts: list[ReceiptSummary]


class ReceiptDetailResponse(BaseModel):
    receipt_id: str
    seq: int
    state_id: str
    created_at: str
    dtype: str
    feature_names: list[str]
    instances: list[dict[str, float]]
    predictions: list[PredictedItem]


class ErrorMessage(BaseModel):
    detail: Any
