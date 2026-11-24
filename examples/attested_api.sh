#!/usr/bin/env bash
# 留证预测 API 示例：演示"续训 -> 覆盖检查点 -> 重启 -> 凭号复算旧回执"。
#
# 前提：docker compose up -d 已启动服务（默认 8000 端口）。
# 流程：
#   1. 留证预测（记录回执号）
#   2. 立即复算，验证与首次一致
#   3. 续训一步（权重发生变化）
#   4. 覆盖保存检查点（CHECKPOINT_PATH 指向的 model.pt 被更新）
#   5. 重启容器（重新加载覆盖后的检查点）
#   6. 凭第 1 步的回执号复算：仍用留证时的旧权重，结果与首次一致
#
# 注意：第 4 步会覆盖宿主机 ./checkpoints/model.pt（git 已跟踪，
#       可用 git checkout -- checkpoints/model.pt 还原）。
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
COMPOSE="${COMPOSE:-docker compose}"
SERVICE="${SERVICE:-classifier-api}"

json_get() { python3 -c "import sys,json;print(json.load(sys.stdin)$1)"; }

echo "== 1. 留证预测：返回分类结果和回执号 =="
ATTEST=$(curl -s -X POST "${BASE_URL}/predict/attested" \
    -H "Content-Type: application/json" \
    -d '{
          "instances": [
            {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1},
            {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9}
          ]
        }')
echo "${ATTEST}" | python3 -m json.tool
RID=$(echo "${ATTEST}" | json_get "['receipt_id']")
echo "回执号: ${RID}"

echo
echo "== 2. 立即凭号复算（max_probability_diff 应为 0）=="
curl -s -X POST "${BASE_URL}/receipts/${RID}/replay" | python3 -m json.tool

echo
echo "== 3. 续训一步（模型权重发生变化）=="
curl -s -X POST "${BASE_URL}/train" \
    -H "Content-Type: application/json" \
    -d '{
          "instances": [
            {"temp": 0.9, "pressure": -0.8, "rpm": 1.5, "current": -0.2},
            {"temp": -0.7, "pressure": 1.2, "rpm": -0.1, "current": 0.6}
          ],
          "labels": [0, 1]
        }' | python3 -m json.tool

echo
echo "== 4. 覆盖保存检查点（./checkpoints/model.pt 被更新）=="
curl -s -X POST "${BASE_URL}/save" \
    -H "Content-Type: application/json" \
    -d '{}' | python3 -m json.tool

echo
echo "== 5. 重启容器（重新加载覆盖后的检查点）=="
${COMPOSE} restart "${SERVICE}"
for i in $(seq 1 30); do
    if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
        echo "服务已恢复（第 ${i} 次探测）"
        break
    fi
    sleep 1
    if [ "${i}" -eq 30 ]; then
        echo "服务未在 30s 内恢复健康" >&2
        exit 1
    fi
done
echo "重启后模型信息（optimizer_step 已 +1，说明加载的是覆盖后的检查点）:"
curl -s "${BASE_URL}/model" | python3 -m json.tool

echo
echo "== 6. 重启后凭旧回执号复算（仍用留证时的旧权重）=="
REPLAY=$(curl -s -X POST "${BASE_URL}/receipts/${RID}/replay")
echo "${REPLAY}" | python3 -m json.tool

echo
echo "== 7. 校验：各类概率与首次相差 <= 1e-6，类别和补零列相同 =="
echo "${REPLAY}" | python3 -c "
import sys, json
body = json.load(sys.stdin)
diff = body['max_probability_diff']
assert diff <= 1e-6, f'概率偏差超限: {diff}'
assert body['labels_match'] is True, '类别不一致'
assert body['zero_filled_match'] is True, '补零列不一致'
print(f\"复算校验通过：max_probability_diff={diff:.3e}，类别/补零列一致\")
"

echo
echo "== 8. 当前保留的回执列表 =="
curl -s "${BASE_URL}/receipts" | python3 -m json.tool
