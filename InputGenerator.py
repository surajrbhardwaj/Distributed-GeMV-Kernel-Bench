import argparse
import numpy as np
import struct

parser = argparse.ArgumentParser(description="Process integer matix dimensions -M and -N.")

# Add -M (requires an integer)
parser.add_argument(
    "-M",
    type=int,
    required=True,
    help="First integer value"
)

# Add -N (requires an integer)
parser.add_argument(
    "-N",
    type=int,
    required=True,
    help="Second integer value"
)

# Add --matrixFile (requires a string)
parser.add_argument(
    "--matrixFile",
    type=str,
    required=True,
    help="Output file for generated matrix"
)

# Add --vectorFile (requires a string)
parser.add_argument(
    "--vectorFile",
    type=str,
    required=True,
    help="Output file for generated vector"
)

args = parser.parse_args()

M = args.M
N = args.N

with open(args.matrixFile, "wb") as f:
    f.write(struct.pack("<II", M, N))
    for _ in range(M):
        row = np.random.uniform(-1.0, 1.0, size=N).astype(np.float16)
        row.tofile(f)

with open(args.vectorFile, "wb") as f:
    f.write(struct.pack("<I", N))
    np.random.uniform(-1.0, 1.0, size=N).astype(np.float16).tofile(f)