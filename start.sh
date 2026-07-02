#!/usr/bin/env bash
# FREEPIO launcher — sets up the CUDA runtime path so the GPU solver engages
# (without it the server silently falls back to CPU), then starts the server.
set -euo pipefail
cd "$(dirname "$0")"

# nvrtc from the pip-installed nvidia-cuda-nvrtc package (see README), plus
# common system CUDA locations as fallbacks.
for d in \
  "$HOME/.local/cuda-nvrtc/nvidia/cuda_nvrtc/lib" \
  /usr/local/cuda/lib64 \
  /usr/lib/x86_64-linux-gnu; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done

# Build with the GPU feature only when an NVIDIA CUDA runtime is actually
# present (nvidia-smi, or a libcuda the loader can find — WSL puts it in
# /usr/lib/wsl/lib). On a GPU-less machine this builds and runs CPU-only.
# Cargo is incremental, so rebuilding is a fast no-op when nothing changed.
FEATURES="--features gpu"
if ! command -v nvidia-smi >/dev/null 2>&1 \
   && ! ldconfig -p 2>/dev/null | grep -q libcuda \
   && [ ! -e /usr/lib/wsl/lib/libcuda.so.1 ]; then
  FEATURES=""
  echo "no NVIDIA CUDA runtime found — building CPU-only"
fi
cargo build --release -p server $FEATURES

exec ./target/release/gto-server "$@"
