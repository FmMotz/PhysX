"""Go2 Rough completion timing with an unchanged, previously saved policy."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation
from isaaclab.utils.io import dump_yaml
import isaaclab_tasks  # registers the velocity tasks
from isaaclab_tasks.utils import resolve_task_config

parser = argparse.ArgumentParser()
parser.add_argument('--num-envs', type=int, default=4096)
parser.add_argument('--warmup', type=int, default=1000)
parser.add_argument('--steps', type=int, default=120)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--plane', action='store_true')
parser.add_argument('--native-actuators', action=argparse.BooleanOptionalAction, default=None)
parser.add_argument('--reset-checks', type=int, default=0)
parser.add_argument('--profile-steps', type=int, default=0,
                    help='Steps between cudaProfilerStart/Stop after the timing windows.')
args = parser.parse_args()
assert min(args.num_envs, args.steps, args.repeats) > 0 and args.warmup >= 0
assert args.reset_checks >= 0
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
task = 'Isaac-Velocity-Rough-UnitreeGo2'
cfg, _ = resolve_task_config(task, None, overrides=['physics=ovphysx'])
cfg.scene.num_envs = args.num_envs
cfg.seed = 42
cfg.sim.device = 'cuda:0'
cfg.curriculum.terrain_levels = None
cfg.scene.terrain.max_init_terrain_level = 0
if args.plane:
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
if args.native_actuators is not None:
    cfg.sim.use_newton_actuators = args.native_actuators
assert cfg.sim.dt == 0.005 and cfg.decimation == 4
args.output.parent.mkdir(parents=True, exist_ok=True)
dump_yaml(str(args.output.with_suffix('.yaml')), cfg)
saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
assert set(saved) == {'std'} | {f'{network}.{layer}.{param}' for network in ('actor', 'critic')
                              for layer in (0, 2, 4, 6) for param in ('weight', 'bias')}
actor = torch.nn.Sequential(torch.nn.Linear(235, 512), torch.nn.ELU(),
                            torch.nn.Linear(512, 256), torch.nn.ELU(),
                            torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 12))
actor.load_state_dict({k.removeprefix('actor.'): v for k, v in saved.items() if k.startswith('actor.')})
actor = actor.cuda().eval()
std = saved['std'].cuda()
torch.distributions.Normal.set_default_validate_args(False)
with launch_simulation(cfg, {'visualizer': [], 'device': 'cuda:0'}):
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            assert obs['policy'].shape == (args.num_envs, 235)
            robot = env.scene['robot']
            result = dict(task=task, num_envs=args.num_envs, seed=42, warmup=args.warmup,
                          steps=args.steps, repeats=args.repeats, physics_dt=cfg.sim.dt,
                          decimation=cfg.decimation, checkpoint=str(args.checkpoint),
                          checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                          joint_names=robot.joint_names, body_names=robot.body_names,
                          observation_terms=env.observation_manager.active_terms,
                          versions={p: importlib.metadata.version(p) for p in
                                    ('isaaclab', 'ovphysx', 'warp-lang', 'torch', 'rsl-rl-lib')},
                          completion_includes_gpu_wait=True, profile=False, resets=0,
                          terrain_type=cfg.scene.terrain.terrain_type,
                          native_actuators=cfg.sim.use_newton_actuators,
                          sdk_environment={k: v for k, v in os.environ.items()
                                           if k.startswith('PHYSX_') or k == 'OVPHYSX_LIB'},
                          scope='Isaac Lab OVPhysX; terrain initial level 0, curriculum disabled; overrides recorded',
                          elapsed_seconds=[])
            done_counts = torch.zeros((), dtype=torch.int64, device='cuda:0')
            for repeat in range(-1, args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for step in range(args.warmup if repeat == -1 else args.steps):
                    mean = actor(torch.cat([obs['policy']], dim=-1))
                    action = torch.distributions.Normal(mean, std.expand_as(mean)).sample()
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done_counts += (terminated | truncated).sum()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                assert torch.isfinite(obs['policy']).all()
                assert torch.isfinite(robot.data.joint_pos.torch).all()
                assert torch.isfinite(robot.data.root_pos_w.torch).all()
                if repeat >= 0:
                    result['elapsed_seconds'].append(elapsed)
                print(f'phase={repeat} seconds={elapsed:.6f}', flush=True)
            result['completion_seconds'] = statistics.median(result['elapsed_seconds'])
            # Bit-level fingerprint of the final state, outside every timing window.
            result['state_sha256'] = {
                name: hashlib.sha256(getattr(robot.data, name).torch.detach().cpu().numpy().tobytes()).hexdigest()
                for name in ('root_pos_w', 'root_quat_w', 'root_lin_vel_w', 'root_ang_vel_w', 'joint_pos', 'joint_vel')}
            result['resets'] = done_counts.item()
            result['root_z_quantiles'] = torch.quantile(robot.data.root_pos_w.torch[:, 2],
                                                       torch.tensor([0., .5, 1.], device='cuda')).tolist()
            # A profiled window is separate from the timing windows and is never mixed with them.
            if args.profile_steps:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                for _ in range(args.profile_steps):
                    mean = actor(obs['policy'])
                    action = torch.distributions.Normal(mean, std.expand_as(mean)).sample()
                    obs, reward, terminated, truncated, _ = env.step(action)
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            # Explicit reset regression is outside every timing window.
            result['explicit_reset_checks'] = []
            for cycle in range(args.reset_checks):
                ids = torch.arange(cycle % 2, args.num_envs, 2, device='cuda', dtype=torch.int32)
                if cycle % 3 == 0:
                    ids = torch.arange(args.num_envs, device='cuda', dtype=torch.int32)
                obs, _ = env.reset(env_ids=ids)
                assert (env.episode_length_buf[ids.long()] == 0).all()
                for _ in range(16):
                    obs, reward, terminated, truncated, _ = env.step(actor(obs['policy']))
                    assert all(torch.isfinite(x).all() for x in
                               (obs['policy'], reward, robot.data.root_pos_w.torch,
                                robot.data.joint_pos.torch, robot.data.joint_vel.torch))
                torch.cuda.synchronize()
                result['explicit_reset_checks'].append(dict(cycle=cycle, count=len(ids), steps=16))
            result['loaded_physics_libraries'] = {}
            for line in Path('/proc/self/maps').read_text().splitlines():
                path = line.split()[-1]
                if path.startswith('/') and '.so' in path and 'physx' in Path(path).name.lower():
                    library = Path(path)
                    if library.is_file() and path not in result['loaded_physics_libraries']:
                        result['loaded_physics_libraries'][path] = hashlib.sha256(library.read_bytes()).hexdigest()
            assert any('libPhysXGpu' in p for p in result['loaded_physics_libraries'])
            result['validation_passed'] = True
            args.output.write_text(json.dumps(result, indent=2) + '\n')
    finally:
        env.close()
