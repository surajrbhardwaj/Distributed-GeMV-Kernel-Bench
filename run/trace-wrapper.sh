#!/usr/bin/env bash

RANK=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}
NODE=$(hostname)
OUTPUT_DIR="./out"
OUTPUT_FILE="${OUTPUT_DIR}/trace_${NODE}_rank_${RANK}.dat"

mkdir -p "${OUTPUT_DIR}"

unset DISPLAY
unset XAUTHORITY

# Forward PMIX and OpenMPI socket variables directly through sudo
exec sudo \
    OMPI_COMM_WORLD_RANK="$OMPI_COMM_WORLD_RANK" \
    OMPI_COMM_WORLD_SIZE="$OMPI_COMM_WORLD_SIZE" \
    PMIX_SERVER_URI="$PMIX_SERVER_URI" \
    PMIX_RANK="$PMIX_RANK" \
    PMIX_SERVER_URI2="$PMIX_SERVER_URI2" \
    PMIX_SECURITY_MODE="$PMIX_SECURITY_MODE" \
    OPAL_PREFIX="$OPAL_PREFIX" \
    /usr/bin/trace-cmd record \
      -e sched:sched_switch \
      -e sched:sched_wakeup \
      -e exceptions:page_fault_user \
      -e tcp:tcp_probe \
      -e net:net_dev_xmit \
      -e net:netif_receive_skb \
      -o "${OUTPUT_FILE}" \
      -F "$@"
