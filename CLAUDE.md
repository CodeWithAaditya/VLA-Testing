# SO-ARM100 fork — π0 VLA fine-tuning on the SO-101 MuJoCo sim

Upstream is TheRobotStudio/SO-ARM100 (CAD, STLs, docs). Everything specific to
this fork lives in `Simulation/SO101/`:

- `teleop.py`, `tabletop_scene.xml`, `make_scene.py` — MuJoCo scene + IK controller (see `README_TELEOP.md`)
- `vla/` — the VLA pipeline: `sim_env.py` (env), `collect_demos.py` (scripted demos → LeRobot dataset),
  `train_pi0_lora.sh` (π0 LoRA fine-tune), `eval_policy.py` (closed-loop sim eval). **Read `vla/README.md` first.**

The pipeline was developed and data-tested on a Mac (no CUDA). Training and the
checkpoint-loading path in `eval_policy.py` were written against LeRobot 0.6.2
source but not yet executed on a GPU — expect to fix small CLI/API drift.

## Setting up on the RTX 3090 Linux box

```bash
cd Simulation/SO101/vla
bash setup_3090.sh            # python3.12 venv + lerobot[dataset,training,pi,peft] + mujoco
source .venv/bin/activate
export MUJOCO_GL=egl          # headless rendering; add to ~/.bashrc
hf auth login                 # π0 uses the gated google/paligemma-3b-pt-224 tokenizer —
                              # the user must accept its licence on the Hub once
python collect_demos.py --dry-run --episodes 20     # expert should score ~85 %
python collect_demos.py --episodes 200              # ~1 h; writes data/so101_pick_cube
bash train_pi0_lora.sh                              # ~4–6 h; BATCH_SIZE=4 GRAD_ACCUM=2 if OOM
python eval_policy.py --checkpoint outputs/pi0_so101_lora/checkpoints/last/pretrained_model --video eval.mp4
```

Things that bite:
- LeRobot needs Python ≥ 3.12; `ffmpeg` must be on PATH (dataset videos); `libegl1` for EGL.
- `sim_env.py` imports `../teleop.py` — run scripts from inside `vla/`, or keep the folder layout.
- If `lerobot-train` rejects a flag, check `lerobot/configs/train.py` / `configs/policies.py` in the
  installed version — the script uses `--policy.path`, `--peft.r`, `--dataset.image_transforms.enable`.
- `data/`, `outputs/`, `.venv/` are git-ignored; don't commit them.

## Conventions
- Match the existing code style (numpy + mujoco, no frameworks beyond LeRobot). Comments explain *why*.
- After changing the scene or the expert, run `collect_demos.py --dry-run` before collecting.
- Sim conventions: state/action = 6 joint positions (5 arm + gripper, rad); cameras `front` and `wrist`;
  30 Hz control with 20 physics substeps; task string is `TASK` in `sim_env.py`.
