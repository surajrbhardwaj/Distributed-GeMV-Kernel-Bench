# Distributed Out-of-Core GeMV & Pipeline Parallelism Benchmarking

A high-performance systems engineering study evaluating the trade-offs between **out-of-core single-node memory-mapped I/O (`mmap`)** and **distributed memory parallelism (MPI)** for large matrix-vector multiplications (GeMV) representative of autoregressive LLM inference workloads.

The project profiles memory hierarchy boundaries, kernel page-fault overhead, and network socket transmission latencies using Linux `ftrace` (`trace-cmd`) and OpenMPI across multi-node topologies.

---

## 🎯 Architecture & Motivation

Autoregressive Large Language Model (LLM) inference is predominantly **memory-bandwidth bound** during the generation phase, where parameter weights ($W$) are streamed once per token generation step to compute activations:

$$y = W \cdot x$$

When model weights exceed host DRAM capacity:
1. **Single-Node Out-of-Core Execution:** Uses OS virtual memory paging via `mmap()` to stream parameters from persistent storage (NVMe SSD / block storage). Execution throughput becomes constrained by major page faults, kernel virtual memory contention (`mmap_lock`), and physical drive read bandwidth (~1.5–7.0 GB/s on physical NVMe; ~180–300 MB/s on virtualized cloud storage).
2. **Distributed Parallel Execution:** Partitions weight matrices across a cluster using Tensor Parallelism (column/row splitting with `MPI_Allreduce`) or Pipeline Parallelism (stage partitioning with peer-to-peer point-to-point communication). Execution latency trades storage page-fault overhead for interconnect fabric latency (TCP/IP socket transmission, NIC queue serialization, and MPI rendezvous overhead).

This repository quantifies the crossover point: **At what matrix dimensions and cluster topologies does distributed network communication outperform single-node disk-paged memory mapping?**

---

## 🏗️ Repository Layout

```text
.
├── src/
│   ├── gemv_serial.cpp             # Single-node baseline using mmap() demand paging
│   ├── gemv_tensor_parallel.cpp    # Distributed Tensor Parallelism (Row/Col split + MPI_Allreduce)
│   └── gemv_pipeline_parallel.cpp  # Distributed Pipeline Parallelism (Staged forward pass)
├── build/                          # Compiled binaries
├── run/
│   ├── trace-wrapper.sh            # ftrace launcher isolating rank threads via trace-cmd
│   └── run_benchmarks.sh           # Automated scaling sweep across matrix dimensions
├── out/                            # Raw trace-cmd .dat files and consolidated traces
├── scripts/
│   ├── parse_traces.py             # Parses ftrace outputs (page faults, switch events, TCP latency)
│   └── plot_results.py             # Generates bandwidth and scaling comparison plots
└── README.md
```
## 📊 Empirical Benchmarking Results

Evaluated on an HPC cluster comparing single-node memory-mapped I/O (`mmap`) against a 4-rank distributed pipeline parallel configuration across scaling matrix workloads.

| Workload | Architecture | Job Wall Time | Kernel System Time | RAM Footprint | Achieved Memory BW | Relative Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **12 Matrices** | Serial (`mmap`) | 14.68 s | 35.11 s | 4.20 GB | 3.51 GB/s | **1.00× (Baseline)** |
| **12 Matrices** | Pipeline (4 Ranks) | 21.12 s | 4.40 – 18.91 s | 12.76 GB | 2.44 – 11.54 GB/s | **0.70× (1.44× slower)** |
| **20 Matrices** | Serial (`mmap`) | 65.16 s | 142.89 s | 4.20 GB | 1.32 GB/s | **1.00× (Baseline)** |
| **20 Matrices** | Pipeline (4 Ranks) | **15.39 s** | **7.14 – 14.43 s** | **21.15 GB** | **5.58 – 7.07 GB/s** | **4.23× (Speedup)** |

### Architectural Takeaways
1. **The Paging Cliff:** Scaling the serial workload from 12 to 20 matrices caused an exponential wall-clock degradation from 14.68s to 65.16s (4.44×), characterized by 142.89s of kernel `System Time` spent inside page-fault handlers and memory bandwidth collapsing to 1.32 GB/s.
2. **Amortization of Interconnect Latency:** At lower memory pressure (12 matrices), pipeline bubble bubbles and MPI overhead (up to 11.05s) make serial execution preferable. At 20 matrices, distributed DRAM aggregation amortizes communication costs, delivering a **4.23× speedup** and **4.6× higher memory throughput**.
