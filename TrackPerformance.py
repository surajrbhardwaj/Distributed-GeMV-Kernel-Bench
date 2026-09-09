#!/usr/bin/env python3
import sys
import os
import re
from datetime import datetime

MD_FILE = "performance.md"

METRIC_CONFIG = {
    "wall_time_sec": {
        "title": "Wall-Clock Duration",
        "unit": "s",
        "lower_is_better": True,
        "desc": "Real-world elapsed execution time.",
        "analysis": {
            "improved": "Faster end-to-end execution; lower overall latency.",
            "regressed": "Run took longer; possible pipeline stalls or sync overhead."
        }
    },
    "task_clock_ms": {
        "title": "Task Clock",
        "unit": "ms",
        "lower_is_better": True,
        "desc": "Total active CPU time consumed across all worker threads.",
        "analysis": {
            "improved": "Reduced core computation overhead or thread spinning.",
            "regressed": "Increased CPU work; potential spin-lock contention."
        }
    },
    "ipc": {
        "title": "Instructions Per Cycle (IPC)",
        "unit": "insn/cyc",
        "lower_is_better": False,
        "desc": "Efficiency of CPU pipeline execution per clock tick.",
        "analysis": {
            "improved": "CPU pipelines are better fed; fewer stalls on memory accesses.",
            "regressed": "Compute cores are idling more waiting on data from memory."
        }
    },
    "llc_load_misses": {
        "title": "LLC (L3) Load Misses",
        "unit": "events",
        "lower_is_better": True,
        "desc": "Last-Level Cache misses that force physical DRAM bus traffic.",
        "analysis": {
            "improved": "Fewer main memory fetches; better working set fit in cache.",
            "regressed": "Higher main memory traffic; cache capacity exceeded."
        }
    },
    "est_dram_read_mb": {
        "title": "Estimated DRAM Fetched",
        "unit": "MB",
        "lower_is_better": True,
        "desc": "Volume of raw data pulled across DRAM bus (LLC misses × 64B).",
        "analysis": {
            "improved": "Lower RAM throughput demand; reduced memory pressure.",
            "regressed": "Higher RAM bandwidth consumed."
        }
    },
    "l1_hit_rate_pct": {
        "title": "L1 Cache Hit Rate",
        "unit": "%",
        "lower_is_better": False,
        "desc": "Percentage of memory accesses served immediately by L1 data cache.",
        "analysis": {
            "improved": "Better temporal locality; vector stays hot in L1 registers/cache.",
            "regressed": "Higher L1 eviction rates; potential cache thrashing."
        }
    },
    "major_faults": {
        "title": "Major Page Faults",
        "unit": "faults",
        "lower_is_better": True,
        "desc": "Instances where data had to be read from swap/disk storage.",
        "analysis": {
            "improved": "Eliminated disk swapping bottlenecks.",
            "regressed": "Kernel was forced to read pages from disk/swap."
        }
    },
    "minor_faults": {
        "title": "Minor Page Faults",
        "unit": "faults",
        "lower_is_better": True,
        "desc": "OS virtual memory page table allocations without disk I/O.",
        "analysis": {
            "improved": "Fewer dynamic memory mappings.",
            "regressed": "More frequent page allocation activity."
        }
    },
    "context_switches": {
        "title": "Context Switches",
        "unit": "events",
        "lower_is_better": True,
        "desc": "Times threads were descheduled or yielded to OS kernel.",
        "analysis": {
            "improved": "Smoother thread execution; fewer sync locks or sleeps.",
            "regressed": "Increased synchronization contention or oversubscription."
        }
    }
}

def parse_perf_text(filepath: str) -> dict:
    raw = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "<not counted>" in line or "<not supported>" in line:
                continue

            time_m = re.search(r"([\d\.]+)\s+seconds time elapsed", line)
            if time_m:
                raw["wall_time_sec"] = float(time_m.group(1))
                continue

            m = re.search(r"^\s*([\d\.,]+)\s+(?:msec\s+)?([a-zA-Z0-9_\-\.\/]+)", line)
            if m:
                val = float(m.group(1).replace(",", ""))
                event = m.group(2)
                raw[event] = val

    metrics = {
        "wall_time_sec": raw.get("wall_time_sec", 0.0),
        "task_clock_ms": raw.get("task-clock", 0.0),
        "context_switches": int(raw.get("context-switches", 0)),
        "cpu_migrations": int(raw.get("cpu-migrations", 0)),
        "page_faults": int(raw.get("page-faults", 0)),
        "major_faults": int(raw.get("major-faults", 0)),
        "minor_faults": int(raw.get("minor-faults", 0)),
        "cycles": raw.get("cpu_core/cycles/", 0.0),
        "instructions": raw.get("cpu_core/instructions/", 0.0),
        "l1_loads": raw.get("cpu_core/L1-dcache-loads/", 0.0),
        "l1_misses": raw.get("cpu_core/L1-dcache-load-misses/", 0.0),
        "llc_loads": raw.get("cpu_core/LLC-loads/", 0.0),
        "llc_misses": raw.get("cpu_core/LLC-load-misses/", 0.0),
    }

    # Derived Calculations
    if metrics["cycles"] > 0:
        metrics["ipc"] = round(metrics["instructions"] / metrics["cycles"], 2)
    if metrics["l1_loads"] > 0:
        metrics["l1_hit_rate_pct"] = round(((metrics["l1_loads"] - metrics["l1_misses"]) / metrics["l1_loads"]) * 100.0, 2)
    if metrics["llc_loads"] > 0:
        metrics["llc_hit_rate_pct"] = round(((metrics["llc_loads"] - metrics["llc_misses"]) / metrics["llc_loads"]) * 100.0, 2)

    metrics["llc_load_misses"] = int(metrics["llc_misses"])
    metrics["est_dram_read_mb"] = round((metrics["llc_misses"] * 64) / (1024 * 1024), 2)
    return metrics

def extract_previous_metrics_from_md(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    
    prev = {}
    with open(filepath, "r") as f:
        content = f.read()

    # Read hidden metadata block
    m = re.search(r"<!-- METRICS_RAW:\s*(\{.*?\})\s*-->", content, re.DOTALL)
    if m:
        try:
            import json
            return json.loads(m.group(1))
        except Exception:
            pass
    return prev

def generate_markdown(current: dict, prev: dict) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    has_prev = bool(prev)

    lines = [
        "# Kernel Performance Profile",
        f"\n**Last Profiled:** {now_str}\n"
    ]

    # Delta Section
    if has_prev:
        lines.append("## Iteration Delta & Comparison\n")
        lines.append("| Metric | Previous | Current | Delta (%) | Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")

        analysis_bullets = []

        for key, conf in METRIC_CONFIG.items():
            if key not in current or key not in prev:
                continue

            c_val = current[key]
            p_val = prev[key]

            if p_val == 0:
                delta_str = "—"
                status = "Neutral"
            else:
                delta = ((c_val - p_val) / p_val) * 100.0
                delta_str = f"{'+' if delta > 0 else ''}{delta:.2f}%"

                # Check if improvement
                if (delta < 0 and conf["lower_is_better"]) or (delta > 0 and not conf["lower_is_better"]):
                    status = "✅ Improved"
                    if abs(delta) >= 1.0:
                        analysis_bullets.append(f"* **{conf['title']} ({delta_str}):** {conf['analysis']['improved']}")
                elif abs(delta) < 0.5:
                    status = "⚪ Unchanged"
                else:
                    status = "⚠️ Regressed"
                    if abs(delta) >= 1.0:
                        analysis_bullets.append(f"* **{conf['title']} ({delta_str}):** {conf['analysis']['regressed']}")

            lines.append(f"| {conf['title']} | {p_val:,.2f} {conf['unit']} | {c_val:,.2f} {conf['unit']} | {delta_str} | {status} |")

        if analysis_bullets:
            lines.append("\n**Key Impact Analysis:**")
            lines.extend(analysis_bullets)
        lines.append("\n---\n")

    # Current Snapshot Table
    lines.append("## Current Execution Metrics\n")
    lines.append("| Metric | Value | What It Measures |")
    lines.append("| :--- | :--- | :--- |")

    for key, conf in METRIC_CONFIG.items():
        if key in current:
            val = current[key]
            lines.append(f"| **{conf['title']}** | {val:,.2f} {conf['unit']} | {conf['desc']} |")

    # Store current machine state in invisible markdown comment for subsequent runs
    import json
    lines.append(f"\n<!-- METRICS_RAW:\n{json.dumps(current, indent=2)}\n-->")
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 TrackPerfomance.py <perf_output.txt>")
        sys.exit(1)

    perf_file = sys.argv[1]
    current_metrics = parse_perf_text(perf_file)
    previous_metrics = extract_previous_metrics_from_md(MD_FILE)

    md_content = generate_markdown(current_metrics, previous_metrics)

    with open(MD_FILE, "w") as f:
        f.write(md_content)