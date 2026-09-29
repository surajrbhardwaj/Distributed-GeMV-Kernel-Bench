#!/bin/bash

# check that all the files currently exist
required_files=(
    "./build/GeMV_Serial"
    "./build/GeMV_TensorParallel"
    "./build/GeMV_PipelineParallel"
    "./build/CompareResults"
    "./out/oracle.vec"
    "./in/vector.vec"
    ./in/matrix{1..20}.mat
)
missing_files=()
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

# TODO: if the missing files are only the input files then run the python script to generate the input files
if (( ${#missing_files[@]} > 0 )); then
    echo "Error: Missing required files:" >&2
    for file in "${missing_files[@]}"; do
        echo "  - $file" >&2
    done
    exit 1
fi

sudo trace-cmd record -e sched_switch -e sched_wakeup -e page_fault_user -o ./out/serial_trace_test.dat -F taskset -c 0,2,4,6,8,10 sh -c './build/GeMV_Serial 32768 32768 > ./out/serial.out 2>&1' &> /dev/null

# compare the results
result="$(./build/CompareResults)"
if [[ "$result" == "No Discrepancies" ]]; then
    # generate the performance report
    echo "successful run"
else
    echo "$result"
fi

# 
# # ./build/GeMV_TensorParallel 32768 32768
# result="$(./build/CompareResults)"
# 
# if [[ "$result" == "No Discrepancies" ]]; then
    # # generate the performance report
# else
    # echo "$result"
# fi
# 
# 
# ./build/GeMV_PipelineParallel 32768 32768
# result="$(./build/CompareResults)"
# 
# if [[ "$result" == "No Discrepancies" ]]; then
    # # generate the performance report
# else
    # echo "$result"
# fi

# rm output.vec
# remove any temporary files here
