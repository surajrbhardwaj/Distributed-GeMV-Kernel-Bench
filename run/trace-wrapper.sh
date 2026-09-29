#!/usr/bin/env bash

RANK=$OMPI_COMM_WORLD_RANK
OUTPUT_FILE="./out/trace_rank_${RANK}.dat"

unset DISPLAY
unset XAUTHORITY

exec sudo trace-cmd record -e sched:sched_switch -e sched:sched_wakeup -e exceptions:page_fault_user -e tcp:tcp_probe -e net:net_dev_xmit -e net:netif_receive_skb -o "${OUTPUT_FILE}" -F "$@"
