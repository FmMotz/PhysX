# Validation materials for the convex core vs triangle mesh culling PR

These files reproduce the checks described in the pull requests for

- `gpu-convexcore-trimesh-triangle-cull`: per-triangle culling in `convexCoreTrimeshNphase_Kernel32`
- `fix-querybv32-bounds-lifetime`: the `TestBvh::queryBV32` reference lifetime fix

Both branches are single commits on top of PhysX 5.12.0 (`b97365c6ddb32fda8469bc788ec18024862b0368`). "stock" below means that commit without changes.

| File | Purpose |
| --- | --- |
| `build.sh` | Builds `libPhysXGpu_64.so` from a checkout, and optionally the static SDK and the contact test |
| `convex_core_mesh_contacts.cpp` | Standalone GPU contact test (general and boundary scenes) |
| `compare_boundary.py` | Compares two contact outputs grouped by (step, core) |
| `bench_task.py` | Isaac Lab timing with seeded random actions (ANYmal-D, Cassie, any rough velocity task) |
| `bench_lab.py` | Isaac Lab Go2 timing with a fixed policy checkpoint (not included) |

Tested with CUDA 12.8, GCC 11 and CMake 3.22/3.28 on Linux, RTX 5070 Ti (SM120) and RTX 4090 (SM89).

## 1. Build

```bash
git clone https://github.com/NVIDIA-Omniverse/PhysX.git stock && git -C stock checkout b97365c6
git -C stock remote add fork https://github.com/FmMotz/PhysX.git && git -C stock fetch fork
git -C stock worktree add ../cull fork/gpu-convexcore-trimesh-triangle-cull
git -C stock worktree add ../lifetime fork/fix-querybv32-bounds-lifetime

# Static SDK, contact test and PhysXGpu from stock
PHYSX_SDK=$PWD/stock/physx BUILD_DIR=$PWD/build-stock CUDA_ARCH=89 WITH_TEST=1 ./build.sh
# PhysXGpu from each branch (copy each library out before building the next one)
PHYSX_SDK=$PWD/cull/physx BUILD_DIR=$PWD/build-cull CUDA_ARCH=89 ./build.sh
PHYSX_SDK=$PWD/lifetime/physx BUILD_DIR=$PWD/build-lifetime CUDA_ARCH=89 ./build.sh
```

Omit `CUDA_ARCH` to build all architectures listed by the upstream CMake. Put each `libPhysXGpu_64.so` in its own directory, for example `lib/stock`, `lib/cull`, `lib/lifetime`.

## 2. Contact test

The test links the static SDK from stock and loads `libPhysXGpu_64.so` through `LD_LIBRARY_PATH`, so only the GPU library differs between runs.

```bash
for v in stock cull lifetime; do
  LD_LIBRARY_PATH=lib/$v build-stock/convex_core_mesh_contacts general_$v.bin
  LD_LIBRARY_PATH=lib/$v build-stock/convex_core_mesh_contacts boundary_$v.bin boundary
done
sha256sum general_*.bin boundary_*.bin
python3 compare_boundary.py boundary_stock.bin boundary_cull.bin
```

The general scene lowers kinematic cores of all six types towards three meshes and drops dynamic cores (240 steps). The boundary scene moves tilted flat cylinders and cones by 1e-7 m per step between two walls in one BVH leaf (600 steps) and prints, per case, the number of steps with contact on each wall. Contacts are extracted through the simulation callback, up to 64 points per pair and callback, after contact reduction.

## 3. Isaac Lab timing

Isaac Lab `main` at `e3a79743e3119b82ebb72e8e3e2410c9275cc168` pins `ovphysx==0.7.0.74035321` (PhysX 5.12.0).

```bash
git clone https://github.com/isaac-sim/IsaacLab.git && git -C IsaacLab checkout e3a79743e3119b82ebb72e8e3e2410c9275cc168
uv sync --project IsaacLab --frozen --extra ovphysx --no-dev
SITE=IsaacLab/.venv/lib/python3.12/site-packages
for v in stock cull lifetime; do
  mkdir -p runtime/$v && cp -a $SITE/ovphysx runtime/$v/
  cp lib/$v/libPhysXGpu_64.so runtime/$v/ovphysx/plugins/libPhysXGpu_64.so
done
OVSTAGE=$SITE/ovstage/bin
OVPHYSX_LIB=$PWD/runtime/cull/ovphysx/lib/libovphysx.so LD_LIBRARY_PATH=$OVSTAGE:$OVSTAGE/plugins \
  IsaacLab/.venv/bin/python bench_task.py --task Isaac-Velocity-Rough-AnymalD --output anymal_cull.json
```

`bench_task.py` uses 4096 environments, seed 42, terrain level 0 without curriculum, 300 warmup control steps and three 120-step windows, with actions from a CUDA generator seeded with 7. The output JSON holds the window times, their median, the reset count and SHA-256 hashes of the final root and joint states.

`bench_lab.py` runs the Go2 rough task with a fixed policy (`--checkpoint`, an RSL-RL actor with 235 inputs; the checkpoint used for the reported numbers is not published) and `--no-native-actuators`, with 1000 warmup control steps.

Run each variant in separate processes and alternate their order. The reported numbers are medians over three processes per variant.
