#!/usr/bin/env bash
# Build PhysXGpu (and optionally the static SDK plus the contact test) from a PhysX checkout.
#
#   PHYSX_SDK=/path/to/PhysX/physx BUILD_DIR=/path/to/build ./build.sh          # PhysXGpu only
#   PHYSX_SDK=... BUILD_DIR=... WITH_TEST=1 ./build.sh                          # also static SDK and test
#
# Optional: CUDA_HOME (default /usr/local/cuda), CUDA_ARCH (for example 89; default: all upstream
# architectures), BUILD_TYPE (default checked), JOBS (default 8), CC/CXX (default gcc-11/g++-11).
# The ovphysx wheel loads a PhysXGpu library built with PX_CHECKED=1 and OmniPVD enabled; use the
# defaults when the library is swapped into that wheel.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sdk="${PHYSX_SDK:?set PHYSX_SDK to the physx directory of a PhysX checkout}"
build="${BUILD_DIR:?set BUILD_DIR}"
cuda="${CUDA_HOME:-/usr/local/cuda}"
type="${BUILD_TYPE:-checked}"
cc="${CC:-/usr/bin/gcc-11}"
cxx="${CXX:-/usr/bin/g++-11}"
cmake -S "$sdk" -B "$build" -DCMAKE_BUILD_TYPE="$type" \
  -DCMAKE_C_COMPILER="$cc" -DCMAKE_CXX_COMPILER="$cxx" \
  -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$cxx" \
  -DPX_GENERATE_GPU_PROJECTS=ON -DPX_GENERATE_STATIC_LIBRARIES=ON \
  -DPX_BUILDSNIPPETS=OFF -DPX_BUILDPVDRUNTIME=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
if [[ -n "${CUDA_ARCH:-}" ]]; then
  # Restrict the generated CUDA flags to one architecture to shorten local builds.
  python3 - "$build" "$CUDA_ARCH" <<'P'
from pathlib import Path
import re, sys
root, arch = Path(sys.argv[1]), sys.argv[2]
if not arch.isdecimal():
    raise ValueError("CUDA_ARCH must be an SM number")
for path in root.rglob("flags.make"):
    source = path.read_text()
    if "--generate-code" not in source:
        continue
    source = re.sub(r'"?--generate-code=[^\s"]+"?', "", source)
    source = source.replace("CUDA_FLAGS = ", f"CUDA_FLAGS = --generate-code=arch=compute_{arch},code=[compute_{arch},sm_{arch}] ")
    path.write_text(source)
P
fi
targets=(PhysXGpu)
[[ -n "${WITH_TEST:-}" ]] && targets+=(PhysXExtensions PhysXPvdSDK)
cmake --build "$build" --target "${targets[@]}" --parallel "${JOBS:-8}"
lib="$sdk/lib/bin/linux.x86_64/$type"
echo "PhysXGpu: $lib/libPhysXGpu_64.so"
if [[ -n "${WITH_TEST:-}" ]]; then
  "$cxx" -std=c++14 -DPX_PHYSX_STATIC_LIB -DPX_CHECKED -I"$sdk/include" -I"$cuda/include" \
    "$here/convex_core_mesh_contacts.cpp" -L"$lib" -Wl,--start-group -lPhysXExtensions_static_64 \
    -lPhysX_static_64 -lPhysXCommon_static_64 -lPhysXFoundation_static_64 -lPhysXPvdSDK_static_64 \
    -lPhysXCudaContextManager_static_64 -lPVDRuntime_static_64 -Wl,--end-group -ldl -lpthread \
    -o "$build/convex_core_mesh_contacts"
  echo "test: $build/convex_core_mesh_contacts"
fi
