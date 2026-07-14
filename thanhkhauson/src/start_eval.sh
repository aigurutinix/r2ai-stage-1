#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
GPU_ID="${GPU_ID:-1}"
REQUIRED_FREE_MIB=36000

echo "Đang đợi GPU ${GPU_ID} trống ít nhất ${REQUIRED_FREE_MIB} MiB..."

while true; do
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}")
    
    if [ "$FREE_MIB" -ge "$REQUIRED_FREE_MIB" ]; then
        echo -e "\nGPU ${GPU_ID} có ${FREE_MIB} MiB trống. Khởi chạy vLLM..."
        break
    else
        echo -ne "GPU ${GPU_ID} hiện có ${FREE_MIB} MiB trống. Đang chờ... \r"
        sleep 10
    fi
done

CUDA_VISIBLE_DEVICES="${GPU_ID}" FLASHINFER_DISABLE_VERSION_CHECK=1 vllm serve model_cache/Qwen3.5-9B \
    --served-model-name qwen3.5-9b \
    --port 8000 \
    --gpu-memory-utilization 0.45 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --max-model-len 65536
