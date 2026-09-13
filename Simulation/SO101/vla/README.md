# Fine-tuning π0 on the SO-101 simulation

End-to-end pipeline: scripted demonstrations in the MuJoCo tabletop scene →
LeRobot dataset → LoRA fine-tune of [π0](https://www.physicalintelligence.company/blog/pi0)
(the LeRobot PyTorch port, `lerobot/pi0_base`) → closed-loop evaluation back in
the sim.

Task: **"Pick up the red cube and put it in the bowl."** Every episode
randomizes the cube and bowl positions, the clutter, the lighting and the
static camera, so the policy has to *look*, not replay a trajectory.

```
vla/
├── sim_env.py          SO101PickEnv: 30 Hz control loop, randomization, camera rendering
├── collect_demos.py    scripted expert → LeRobotDataset (videos + parquet)
├── train_pi0_lora.sh   lerobot-train invocation sized for one RTX 3090
├── eval_policy.py      run a checkpoint in the sim, report success rate, save video
├── setup_3090.sh       one-time environment setup on the Linux/CUDA machine
└── requirements-mac.txt
```

`sim_env.py` imports `../teleop.py` (the IK controller and scene loader), so
copy the whole `Simulation/SO101` folder, not just `vla/`.

## What the policy sees and outputs

| key | shape | meaning |
|---|---|---|
| `observation.images.front` | 256×256×3 | static camera in front of the table |
| `observation.images.wrist` | 256×256×3 | camera on the gripper |
| `observation.state` | 6 | joint positions (rad): `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` |
| `action` | 6 | joint position **targets** (rad) for the same joints, i.e. what the position servos are told |

This is the same layout a real LeRobot SO-101 follower records (`lerobot-record`),
so the same training command works on a hardware dataset later — only the
normalization stats change. π0 resizes images to 224×224 internally.

## 1. Transfer to the 3090 PC

```bash
# from this repo, on the Mac
rsync -avz --exclude data --exclude outputs Simulation/SO101/ user@pc:~/so101/
```

On the PC:

```bash
cd ~/so101/vla
bash setup_3090.sh                 # venv + lerobot[dataset,training,pi,peft] + mujoco; prints next steps
source .venv/bin/activate
export MUJOCO_GL=egl               # headless rendering on Linux
hf auth login                      # pi0 uses the gated google/paligemma-3b-pt-224 tokenizer:
                                   # accept its licence on the Hub first, once
```

## 2. Collect demonstrations

```bash
python collect_demos.py --episodes 200 --root data/so101_pick_cube
```

- ~11 s of sim time (≈330 frames) per episode, ~20 s wall on a laptop (rendering
  + AV1 encoding), faster on the Linux box. Episodes where the scripted expert
  fails are discarded, so `--episodes` is the number *kept*: expect ~1.2×
  attempts.
- `--dry-run` runs the expert without writing anything and prints its
  success rate — do this after changing the scene or the expert.
- `--seed` changes the randomization stream; the evaluator uses a different
  default seed so eval scenes are never training scenes.
- Sanity-check what you recorded:
  `lerobot-dataset-viz --repo-id local/so101_sim_pick_cube --root data/so101_pick_cube --episode-index 0`

200 episodes is a reasonable first run for a single task. If the policy is
flaky, more data (500+) helps more than more steps.

## 3. Train (LoRA, RTX 3090)

```bash
bash train_pi0_lora.sh                      # ~4-6 h for 15k steps on a 3090
STEPS=20000 LORA_R=64 bash train_pi0_lora.sh   # any variable at the top of the script
bash train_pi0_lora.sh --wandb.enable=true  # extra args pass straight to lerobot-train
```

How it fits in 24 GB: the 3B PaliGemma backbone is frozen, LoRA adapters go
on the 300M action expert's attention q/v projections plus the action/state
projection layers (LeRobot's default π0 PEFT targets), weights are bf16, and
gradient checkpointing is on. Batch 8 should sit around 16–20 GB; if you OOM
use `BATCH_SIZE=4 GRAD_ACCUM=2`.

Checkpoints land in `outputs/pi0_so101_lora/checkpoints/<step>/pretrained_model/`
(`last` is a symlink) and contain the adapter weights, the config and the
pre/post-processor pipelines. Loss should drop quickly in the first ~1k steps
and then creep down; a flow-matching loss around 0.05–0.1 is typical when it
starts working.

Resume an interrupted run: `bash train_pi0_lora.sh --resume=true --config_path=outputs/pi0_so101_lora/checkpoints/last/pretrained_model/train_config.json`

## 4. Evaluate in the sim

```bash
python eval_policy.py \
  --checkpoint outputs/pi0_so101_lora/checkpoints/last/pretrained_model \
  --dataset-root data/so101_pick_cube \
  --episodes 20 --video eval.mp4
```

Prints per-episode SUCCESS/fail and the overall rate; the mp4 shows the front
and wrist views side by side. `--n-action-steps 25` executes half of each
50-step action chunk before re-planning (more reactive, twice the compute).
This also runs on the Mac (`--device mps`, slowly) — copy the checkpoint and
`data/so101_pick_cube/meta` back if you want to watch it locally.

## Tuning notes

- **Expert quality bounds policy quality.** Run `--dry-run` first. The scripted
  expert currently succeeds on ~85 % of randomized scenes (33/40 and 35/40 on
  two seeds); failures are discarded, so that only costs collection time. If
  it drops well below that after a change, something broke.
- **What the expert needed to work at 30 Hz** (all in `sim_env.py` /
  `ScriptedExpert`, found by sweeping the randomized scene):
  - joint targets are ramped across the physics substeps (`SO101PickEnv.step`);
    a raw 30 Hz staircase shakes the cube out of the torque-limited jaw.
  - stiffer contacts (`CONTACT_SOLREF`); with the defaults a cube resting
    against the bowl's thin staves would spontaneously launch itself out.
  - the bowl is a free body — it is placed through `qpos`, and the arm can nudge it.
  - the grasp point is offset 4 mm sideways so the fixed jaw clears the cube's
    face, and the release happens from ~9 cm above the bowl.
- **Randomization ranges** live at the top of `sim_env.py`. Two are deliberately
  narrow because of the 5-DOF wrist: the cube radius (≤ 0.27 m) and the cube's
  yaw relative to its azimuth (−30°…+10°, mod 90°). Widen them only if
  `--dry-run` still passes; a learned policy will not do better than its data.
- **Language conditioning.** The task string is a constant (`TASK` in
  `sim_env.py`). To make the prompt matter, collect a second dataset for
  another cube with a different task string and train on both
  (`--dataset.repo_id` accepts a list in recent LeRobot versions).
- **Alternatives on the same data:** `--policy.type=smolvla` (450M, trains on
  far less memory) or π0.5 (`--policy.path=lerobot/pi05_base`) are drop-in
  swaps of the base model in `train_pi0_lora.sh`.
