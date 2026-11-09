"""Compare two boundary-scene outputs grouped by (step, core); records are 4 uint32 then 7 float32."""
import struct
import sys
from collections import defaultdict

def load(path):
    data = open(path, "rb").read()
    size = struct.calcsize("<4I7f")
    groups = defaultdict(list)
    for i in range(0, len(data), size):
        r = struct.unpack_from("<4I7f", data, i)
        groups[(r[0], r[1])].append(r[2:])
    return {k: sorted(v) for k, v in groups.items()}

a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) | set(b))
count_diff = [k for k in keys if len(a.get(k, [])) != len(b.get(k, []))]
value_diff = [k for k in keys if k not in count_diff and a[k] != b[k]]
print(f"groups {len(keys)} count_differs {len(count_diff)} values_differ {len(value_diff)}")
for k in (count_diff + value_diff)[:6]:
    print("  step", k[0], "core", k[1], "points", len(a.get(k, [])), len(b.get(k, [])))
worst = 0.0
for k in value_diff:
    for x, y in zip(a[k], b[k]):
        worst = max(worst, max(abs(p - q) for p, q in zip(x[2:], y[2:])))
print("max value difference in equal-count groups", worst)
