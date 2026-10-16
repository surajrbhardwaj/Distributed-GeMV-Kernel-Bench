import argparse
import numpy as np
import struct

parser = argparse.ArgumentParser(description="Process integer matix dimensions -M and -N.")

# Add --type (requires a string)
parser.add_argument(
    "--type",
    type=str,
    required=True,
    help="Matrix or Vector"
)

# Add -M (requires an integer)
parser.add_argument(
    "-M",
    type=int,
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
    "--fileName",
    type=str,
    required=True,
    help="Output file for generated data"
)

if __name__ == "__main__":
    args = parser.parse_args()

    matrixOrVector = args.type.lower()
    if matrixOrVector == 'matrix':
        M = args.M
        N = args.N

        limit = np.sqrt(3.0 / N)
        with open(args.fileName, "wb") as f:
            for _ in range(M):
                row = np.random.uniform(-1*limit, limit, size=N).astype(np.float32)
                row.tofile(f)

    elif matrixOrVector == "vector":
        N = args.N
        with open(args.fileName, "wb") as f:
            np.random.uniform(-1.0, 1.0, size=N).astype(np.float32).tofile(f)

    else:
        print("invalid input --type must be either matrix or vector")
