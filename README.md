# Validation materials for the GPU convex core vs triangle mesh PRs

These files reproduce the checks described in the pull requests for

- `gpu-convexcore-trimesh-triangle-cull`: per-triangle culling in `convexCoreTrimeshNphase_Kernel32`
- `fix-querybv32-bounds-lifetime`: the `TestBvh::queryBV32` reference lifetime fix
- `gpu-convexcore-trimesh-triangle-core`: the `eTRIANGLE` convex core that replaces the per-triangle `PointsCore` (two commits on top of `gpu-convexcore-trimesh-triangle-cull`)

The first two branches are single commits on top of PhysX 5.12.0 (`b97365c6ddb32fda8469bc788ec18024862b0368`). "stock" below means that commit without changes.

| File | Purpose |
| --- | --- |
| `build.sh` | Builds `libPhysXGpu_64.so` from a checkout, and optionally the static SDK and the contact test |
| `convex_core_mesh_contacts.cpp` | Standalone GPU contact test (general and boundary scenes) |
| `compare_boundary.py` | Compares two contact outputs grouped by (step, core) |
| `bench_task.py` | Isaac Lab timing with seeded random actions (ANYmal-D, Cassie, any rough velocity task) |
| `bench_lab.py` | Isaac Lab Go2 timing with a fixed policy checkpoint (not included) |
| `triangle_core_support_test.cpp` | Host-side differential test of `TriangleCore` against the identity `PointsCore` it replaces |

Tested with CUDA 12.8, GCC 11 and CMake 3.22/3.28 on Linux, RTX 5070 Ti (SM120) and RTX 4090 (SM89).

## 1. Build

```bash
git clone https://github.com/NVIDIA-Omniverse/PhysX.git stock && git -C stock checkout b97365c6
git -C stock remote add fork https://github.com/FmMotz/PhysX.git && git -C stock fetch fork
git -C stock worktree add ../cull fork/gpu-convexcore-trimesh-triangle-cull
git -C stock worktree add ../lifetime fork/fix-querybv32-bounds-lifetime
git -C stock worktree add ../core fork/gpu-convexcore-trimesh-triangle-core

# Static SDK, contact test and PhysXGpu from stock
PHYSX_SDK=$PWD/stock/physx BUILD_DIR=$PWD/build-stock CUDA_ARCH=89 WITH_TEST=1 ./build.sh
# PhysXGpu from each branch (copy each library out before building the next one)
PHYSX_SDK=$PWD/cull/physx BUILD_DIR=$PWD/build-cull CUDA_ARCH=89 ./build.sh
PHYSX_SDK=$PWD/lifetime/physx BUILD_DIR=$PWD/build-lifetime CUDA_ARCH=89 ./build.sh
PHYSX_SDK=$PWD/core/physx BUILD_DIR=$PWD/build-core CUDA_ARCH=89 ./build.sh
```

Omit `CUDA_ARCH` to build all architectures listed by the upstream CMake. Put each `libPhysXGpu_64.so` in its own directory, for example `lib/stock`, `lib/cull`, `lib/lifetime`, `lib/core`.

## 2. Contact test

The test links the static SDK from stock and loads `libPhysXGpu_64.so` through `LD_LIBRARY_PATH`, so only the GPU library differs between runs.

```bash
for v in stock cull lifetime core; do
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
for v in stock cull lifetime core; do
  mkdir -p runtime/$v && cp -a $SITE/ovphysx runtime/$v/
  cp lib/$v/libPhysXGpu_64.so runtime/$v/ovphysx/plugins/libPhysXGpu_64.so
done
OVSTAGE=$SITE/ovstage/bin
OVPHYSX_LIB=$PWD/runtime/cull/ovphysx/lib/libovphysx.so LD_LIBRARY_PATH=$OVSTAGE:$OVSTAGE/plugins \
  IsaacLab/.venv/bin/python bench_task.py --task Isaac-Velocity-Rough-AnymalD --output anymal_cull.json
```

`bench_task.py` uses 4096 environments, seed 42, terrain level 0 without curriculum, 300 warmup control steps and three 120-step windows, with actions from a CUDA generator seeded with 7. The output JSON holds the window times, their median, the reset count and SHA-256 hashes of the final root and joint states.

`bench_lab.py` runs the Go2 rough task with a fixed policy (`--checkpoint`, an RSL-RL actor with 235 inputs; the checkpoint used for the reported numbers is not published) and `--no-native-actuators`, with 1000 warmup control steps.

Run each variant in separate processes and alternate their order. The culling PR reports medians over three processes per variant; the triangle-core PR reports medians over six processes per variant on SM120 and three on SM89.

## 4. Support function test (`gpu-convexcore-trimesh-triangle-core`)

`triangle_core_support_test.cpp` includes `GuConvexSupport.h` from the branch checkout and calls `localSupport` and `contactFace` for `Type::eTRIANGLE` and for `Type::ePOINTS` with the core the stock kernel built (three `float4` vertices, unit scale, identity rotation). It runs 34 fixed cases (unique maxima, exact ties between two and three vertices, directions along the face normal and edges, zero, sub-epsilon and denormal directions, the `FLT_EPSILON` boundary on a triangle away from the origin, repeated and collinear vertices, large, tiny and negative-zero coordinates) followed by 200000 random cases. Coordinates stay below `FLT_MAX / 2`, where the identity rotation of `PointsCore` would overflow. The GPU libraries are built with `-ftz=true`, so on the device the `PointsCore` path also flushed subnormal coordinates; the host test does not reproduce that.

```bash
SDK=$PWD/core/physx
g++ -std=c++14 -O2 -I$SDK/include -I$SDK/source/geomutils/include -I$SDK/source/common/include \
  -I$SDK/source/common/src triangle_core_support_test.cpp -o triangle_core_support_test
./triangle_core_support_test            # 200000 random cases
./triangle_core_support_test 1000000    # more random cases
./triangle_core_support_test --strict   # also fail on sign-of-zero differences
```

Expected output (the counts depend on the random case count):

```
cases: 34 fixed + 200000 random, non-zero supports 194510, face counts 0/1/2/3+: 140623/0/30190/29221
sign-of-zero only differences in support points: 17244 (without a -0.0 vertex: 0)
PASS: TriangleCore matches identity PointsCore (bit for bit except the sign of zero)
```

Contact faces (count, normal and points) must match bit for bit, since both cores compute them through the same `PointsCore` code. Support points match bit for bit with one documented exception: the identity quaternion rotation in `PointsCore` turns a `-0.0` vertex component into `+0.0`, while `TriangleCore` returns the stored vertex. The test counts those support points separately and fails if such a difference occurs for a triangle without a `-0.0` vertex component. This test is compiled for the host with g++; the kernels are compiled by nvcc, and their device behaviour is covered by the GPU contact test above on the enumerated scenes (byte-identical outputs between `stock`, `cull` and `core`).
