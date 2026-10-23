#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
BENCH_CONFIG_DIR="${SCRIPT_DIR}/../bench"
SERVER_CONFIG_DIR="${SCRIPT_DIR}/../server"

# Mixed GPUs (A100 + RTX 3060): pin PCI order so YAML devices "0" is the A100.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# python, which now has a NumPy 1.x / 2.x mix after `pip install -e .`.
VLLM_ENV="${VLLM_ENV:-/home/sdl/Desktop/github/furiosa-perf/.venv-gpu}"
export PATH="${VLLM_ENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export FURIOSA_PERF_RESULTS_DIR="${SCRIPT_DIR}/../results"
export FURIOSA_PERF_BENCHMARK_NAME="$(basename -- "${BENCH_CONFIG_DIR}/prefill.yaml" .yaml)"
# FlashInfer 0.7 uses an nvcc option unavailable in the installed CUDA 12.4.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"


MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
if [[ "${1:-}" == "--model" ]]; then
  MODEL_ID="${2:?Usage: bash $0 --model MODEL_ID}"
fi

cd -- "$REPO_ROOT"

"${VLLM_ENV}/bin/python" -m furiosa_perf.cli.main run \
  --backend vllm \
  --hardware-type gpu \
  --server-config "${SERVER_CONFIG_DIR}/Llama3.1-8B-Instruct.yaml" \
  --benchmark-config "${BENCH_CONFIG_DIR}/prefill.yaml" \
  --model-id "${MODEL_ID}"
