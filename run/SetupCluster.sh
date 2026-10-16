#!/bin/bash
set -e

NODES=("mpi-node-0" "mpi-node-1" "mpi-node-2" "mpi-node-3")

for node in "${NODES[@]}"; do
  echo "=========================================================="
  echo ">>> Dispatching setup to $node ..."
  echo "=========================================================="
  
  if [ "$node" == "$(hostname)" ]; then
    ~/WorkerSetup.sh
  else
    scp ~/WorkerSetup.sh "$node:~/"
    ssh "$node" "~/WorkerSetup.sh"
  fi
done

echo "=========================================================="
echo "Cluster bootstrap complete across all 4 nodes."
echo "=========================================================="
