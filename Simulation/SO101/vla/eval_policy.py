#!/usr/bin/env python3
"""Run a fine-tuned pi0 (LoRA adapter or full checkpoint) in the SO-101 sim.

    python eval_policy.py --checkpoint outputs/pi0_so101_lora/checkpoints/last/pretrained_model \
                          --dataset-root data/so101_pick_cube --episodes 20 --video eval.mp4

Loads the policy the same way `lerobot-train` saved it (PEFT adapter on top of
the base model, plus the saved pre/post-processors), then closes the loop at
30 Hz: render -> policy.select_action -> joint targets -> physics.  Reports the
success rate and optionally writes a side-by-side video of the front and
wrist cameras.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sim_env import CAMERAS, FPS, TASK, SO101PickEnv


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_policy(checkpoint: str, dataset_root: str | None, repo_id: str, device: torch.device,
                n_action_steps: int | None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint          # make_policy loads weights (and the PEFT adapter) from here
    cfg.device = str(device)
    if n_action_steps is not None:
        cfg.n_action_steps = n_action_steps

    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=checkpoint, dataset_stats=ds_meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, pre, post


def obs_to_batch(obs: dict, device: torch.device) -> dict:
    batch = {
        "observation.state": torch.from_numpy(obs["state"]).unsqueeze(0).to(device),
        "task": [TASK],
    }
    for key in CAMERAS:
        img = torch.from_numpy(obs[key]).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        batch[f"observation.images.{key}"] = img.to(device)
    return batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help=".../checkpoints/<step>/pretrained_model directory (or a Hub repo id)")
    ap.add_argument("--dataset-root", default="data/so101_pick_cube",
                    help="the training dataset (for normalization stats / feature names)")
    ap.add_argument("--repo-id", default="local/so101_sim_pick_cube")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=15 * FPS, help="per episode (default 15 s)")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="actions executed per policy call (default: the checkpoint's; "
                         "smaller = more reactive, slower)")
    ap.add_argument("--img-size", type=int, default=256, help="must match the dataset")
    ap.add_argument("--seed", type=int, default=1000, help="different from collection seeds")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (auto)")
    ap.add_argument("--video", default=None, help="write an mp4 of all episodes here")
    ap.add_argument("--no-randomize", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")
    policy, pre, post = load_policy(args.checkpoint, args.dataset_root, args.repo_id, device,
                                    args.n_action_steps)
    env = SO101PickEnv(img_size=args.img_size, seed=args.seed)

    writer = None
    if args.video:
        import imageio
        writer = imageio.get_writer(args.video, fps=FPS, codec="libx264", quality=7)

    successes = 0
    for ep in range(args.episodes):
        obs = env.reset(randomize=not args.no_randomize)
        policy.reset()
        t0 = time.time()
        steps = 0
        done = False
        for steps in range(1, args.max_steps + 1):
            with torch.inference_mode():
                batch = pre(obs_to_batch(obs, device))
                action = post(policy.select_action(batch))
            action = action.squeeze(0).float().cpu().numpy()
            obs = env.step(action)
            if writer is not None:
                writer.append_data(np.concatenate([obs["front"], obs["wrist"]], axis=1))
            if env.success():
                done = True
                # keep rolling a moment so a "success" that bounces out does not count
                for _ in range(FPS):
                    with torch.inference_mode():
                        action = post(policy.select_action(pre(obs_to_batch(obs, device))))
                    obs = env.step(action.squeeze(0).float().cpu().numpy())
                    if writer is not None:
                        writer.append_data(np.concatenate([obs["front"], obs["wrist"]], axis=1))
                done = env.success()
                break
        successes += done
        hz = steps / (time.time() - t0)
        print(f"episode {ep + 1:3d}/{args.episodes}: {'SUCCESS' if done else 'fail   '} "
              f"after {steps} steps  ({hz:.1f} control Hz)   running {successes}/{ep + 1}")

    if writer is not None:
        writer.close()
        print(f"video: {args.video}")
    env.close()
    print(f"\nsuccess rate: {successes}/{args.episodes} = {100 * successes / args.episodes:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
