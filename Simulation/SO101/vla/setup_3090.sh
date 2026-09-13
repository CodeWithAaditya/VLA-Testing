#!/usr/bin/env bash
# One-time setup on the Linux + RTX 3090 machine.
#
#   bash setup_3090.sh          # creates ./.venv and installs everything
#   source .venv/bin/activate
#
# Requirements: Python 3.12 (lerobot needs >=3.12), an NVIDIA driver that
# supports CUDA 12, ffmpeg (video encoding for the dataset), and EGL for
# headless MuJoCo rendering (libegl1 / libgl1, usually already present).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3.12}"
command -v "$PY" >/dev/null || { echo "need $PY (or run with PY=python3.x)"; exit 1; }
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found -- sudo apt install ffmpeg (needed to encode dataset videos)"

"$PY" -m venv .venv
source .venv/bin/activate
pip install -U pip
# lerobot[pi] = pi0/pi0.5 deps (transformers etc.), [peft] = LoRA
pip install "lerobot[dataset,training,pi,peft]" mujoco imageio imageio-ffmpeg

python - <<'PY'
import torch, mujoco, lerobot, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("mujoco", mujoco.__version__, "| lerobot", lerobot.__version__, "| peft", peft.__version__)
PY

cat <<'MSG'

Next:
  1. The pi0 base model uses Google's gated PaliGemma tokenizer. Accept the licence at
        https://huggingface.co/google/paligemma-3b-pt-224
     then log in:   hf auth login
  2. Headless rendering:   export MUJOCO_GL=egl     (put it in ~/.bashrc)
  3. Collect data:         python collect_demos.py --episodes 200
  4. Train:                bash train_pi0_lora.sh
  5. Evaluate:             python eval_policy.py --checkpoint outputs/pi0_so101_lora/checkpoints/last/pretrained_model --video eval.mp4
MSG
