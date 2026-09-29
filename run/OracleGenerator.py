import numpy as np
import sys

if len(sys.argv) < 3:
    print("Usage: python3 oracleGenerator.py <M> <N>")
    sys.exit(1)

M = int(sys.argv[1])
N = int(sys.argv[2])

# 1. Read input vector (M and N must be equal for chained GeMV)
vec = np.fromfile("inputs/vector.vec", dtype=np.float16)
if vec.size != N:
    raise ValueError(f"Vector size ({vec.size}) does not match N ({N})")

# 2. Iterate through matrix1.mat to matrix20.mat
for i in range(1, 21):
    mat_path = f"inputs/matrix{i}.mat"

    # Read row-major binary matrix
    mat = np.fromfile(mat_path, dtype=np.float16).reshape((M, N))

    # Mirror your kernel's precision model:
    # Kernel accumulates in FP32, then casts back down to FP16
    vec_fp32 = vec.astype(np.float32)
    mat_fp32 = mat.astype(np.float32)

    # Compute M x v
    res_fp32 = np.add.reduce(mat_fp32 * vec_fp32, axis=1)

    # Downcast to float16 for the next iteration (matching: results[w] = static_cast<std::float16_t>(sum))
    vec = res_fp32.astype(np.float16)

# 3. Save expected output vector
vec.tofile("out/oracle.vec")
print("Reference calculation complete. Saved to oracle.vec")
