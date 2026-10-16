cat << 'EOF' > ~/setup_cluster.sh
#!/bin/bash
set -e

# --- Configuration ---
REPO_URL="https://github.com/surajrbhardwaj/Distributed-LLM-Inference-Research.git"
REPO_DIR="$HOME/Distributed-LLM-Inference-Research"
NODES=("mpi-node-0" "mpi-node-1" "mpi-node-2" "mpi-node-3")

for node in "${NODES[@]}"; do
  echo "============================================================"
  echo ">>> [1/6] Initializing and Provisioning: $node"
  echo "============================================================"

  ssh "$node" "bash -s" << REMOTE_SCRIPT
    set -e

    # 1. System packages: C++ toolchain, OpenMPI, trace-cmd, Python, and CMake
    echo "--> Installing system packages (build-essential, OpenMPI, trace-cmd, CMake)..."
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      build-essential \
      gcc \
      g++ \
      openmpi-bin \
      libopenmpi-dev \
      trace-cmd \
      cmake \
      python3 \
      python3-pip \
      python3-venv \
      git

    # 2. Configure passwordless sudo for trace-cmd
    echo "--> Configuring passwordless sudoers entry for /usr/bin/trace-cmd..."
    echo "\$USER ALL=(ALL) NOPASSWD: /usr/bin/trace-cmd" | sudo tee /etc/sudoers.d/tracecmd > /dev/null
    sudo chmod 0440 /etc/sudoers.d/tracecmd

    # 3. Clone or pull latest repository
    if [ ! -d "$REPO_DIR" ]; then
      echo "--> Cloning repository into $REPO_DIR..."
      git clone "$REPO_URL" "$REPO_DIR"
    else
      echo "--> Existing repository detected. Pulling latest commits..."
      cd "$REPO_DIR" && git pull || true
    fi

    cd "$REPO_DIR"

    # 4. Set up Python virtual environment and install numpy
    if [ ! -d ".venv" ]; then
      echo "--> Initializing Python venv in $REPO_DIR/.venv..."
      python3 -m venv .venv
    fi

    echo "--> Updating pip and installing numpy..."
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install numpy -q

    # 5. Generate input matrices
    echo "--> Creating input directory and generating 20 matrices (32768x32768)..."
    mkdir -p in
    for i in {1..20}; do
      if [ ! -f "in/matrix\${i}.mat" ]; then
        echo "    Generating in/matrix\${i}.mat..."
        .venv/bin/python3 run/InputGenerator.py --type matrix -M 32768 -N 32768 --fileName "in/matrix\${i}.mat"
      fi
    done
    if [ ! -f "in/matrix\${i}.mat" ]; then
        echo "    Generating in/vector.vec..."
        .venv/bin/python3 run/InputGenerator.py --type vector -N 32768 --fileName "in/vector.vec"
    fi

    # 6. Configure and compile project via CMake
    echo "--> Configuring and compiling CMake project (Release mode)..."
    cmake -B build -S .
    cmake --build build -j

    echo "--> Node \$(hostname) successfully provisioned!"
REMOTE_SCRIPT

done

echo "============================================================"
echo "Cluster bootstrap complete across all nodes."
echo "============================================================"
EOF

chmod +x ~/setup_cluster.sh
