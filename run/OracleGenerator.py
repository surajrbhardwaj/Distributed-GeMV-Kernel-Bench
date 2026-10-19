import numpy as np
import sys

if len(sys.argv) < 3:
    print("Usage: python3 oracleGenerator.py <M> <N>")
    sys.exit(1)

M = int(sys.argv[1])
N = int(sys.argv[2])

# 1. Read input vector (M and N must be equal for chained GeMV)
vec = np.fromfile("in/vector.vec", dtype=np.float32)
if vec.size != N:
    raise ValueError(f"Vector size ({vec.size}) does not match N ({N})")

# 2. Iterate through matrix1.mat to matrix20.mat
for i in range(1, 21):
    mat_path = f"in/matrix{i}.mat"

    # Read row-major binary matrix
    mat = np.fromfile(mat_path, dtype=np.float32).reshape((M, N))


    # Compute M x v
    res = np.add.reduce(mat * vec, axis=1)

    vec = res.astype(np.float32)

# 3. Save expected output vector
vec.tofile("out/oracle.vec")
print("Reference calculation complete. Saved to oracle.vec")
