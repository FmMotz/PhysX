"""Timing for any rough-velocity task with seeded random actions, for cross-robot comparison."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation
import isaaclab_tasks  # registers the velocity tasks
from isaaclab_tasks.utils import resolve_task_config

parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True)
parser.add_argument('--num-envs', type=int, default=4096)
parser.add_argument('--warmup', type=int, default=300)
parser.add_argument('--steps', type=int, default=120)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--profile-steps', type=int, default=0)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
cfg, _ = resolve_task_config(args.task, None, overrides=['physics=ovphysx'])
cfg.scene.num_envs = args.num_envs
cfg.seed = 42
cfg.sim.device = 'cuda:0'
cfg.curriculum.terrain_levels = None
cfg.scene.terrain.max_init_terrain_level = 0
with launch_simulation(cfg, {'visualizer': [], 'device': 'cuda:0'}):
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        with torch.inference_mode():
            env.reset()
            robot = env.scene['robot']
            generator = torch.Generator(device='cuda:0').manual_seed(7)
            dim = env.action_manager.total_action_dim
            act = lambda: 0.5 * torch.randn(args.num_envs, dim, device='cuda:0', generator=generator)
            result = dict(task=args.task, num_envs=args.num_envs, warmup=args.warmup, dt=cfg.sim.dt,
                          decimation=cfg.decimation, action_dim=dim, joints=len(robot.joint_names),
                          bodies=len(robot.body_names), elapsed_seconds=[])
            done = torch.zeros((), dtype=torch.int64, device='cuda:0')
            for repeat in range(-1, args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.warmup if repeat == -1 else args.steps):
                    _, _, terminated, truncated, _ = env.step(act())
                    done += (terminated | truncated).sum()
                torch.cuda.synchronize()
                if repeat >= 0:
                    result['elapsed_seconds'].append(time.perf_counter() - start)
            result['completion_seconds'] = statistics.median(result['elapsed_seconds'])
            result['resets'] = done.item()
            result['state_sha256'] = {
                name: hashlib.sha256(getattr(robot.data, name).torch.detach().cpu().numpy().tobytes()).hexdigest()
                for name in ('root_pos_w', 'root_quat_w', 'joint_pos', 'joint_vel')}
            if args.profile_steps:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                for _ in range(args.profile_steps):
                    env.step(act())
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            args.output.write_text(json.dumps(result, indent=1) + '\n')
    finally:
        env.close()
