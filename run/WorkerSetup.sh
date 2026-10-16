set -e

REPO_DIR="$HOME/Distributed-LLM-Inference-Research"
REPO_URL="https://github.com/surajrbhardwaj/Distributed-LLM-Inference-Research.git"

echo "=== [1/6] Installing apt packages on $(hostname) ==="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential gcc g++ openmpi-bin libopenmpi-dev trace-cmd cmake python3 python3-pip python3-venv git

echo "=== [2/6] Configuring sudoers for trace-cmd ==="
echo "$USER ALL=(ALL) NOPASSWD: /usr/bin/trace-cmd" | sudo tee /etc/sudoers.d/tracecmd > /dev/null
sudo chmod 0440 /etc/sudoers.d/tracecmd

echo "=== [3/6] Setting up repository ==="
cd "$HOME"
if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  cd "$REPO_DIR" && git pull || true
fi

cd "$REPO_DIR"

echo "=== [4/6] Python venv and numpy ==="
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install numpy -q

echo "=== [5/6] Generating 20 input matrices ==="
mkdir -p in
for i in {1..20}; do
  if [ ! -f "in/matrix${i}.mat" ]; then
    echo "  Generating in/matrix${i}.mat..."
    .venv/bin/python3 run/InputGenerator.py --type matrix -M 32768 -N 32768 --fileName "in/matrix${i}.mat"
  fi
done

echo "=== [6/6] Building CMake target ==="
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

echo "=== Node $(hostname) fully configured ==="
EOF
