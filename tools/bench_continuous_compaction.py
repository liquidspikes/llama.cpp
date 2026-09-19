#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for Continuous Compaction Architecture
Measures:
1. AMD XDNA 2 NPU (FastFlowLM @ port 52625) Compactor Throughput & TTFT.
2. Primary Cluster Decode & Prefill Speed: Lean Compacted Context vs. Bloated Context.
3. Multi-Session Concurrency: 1, 2, and 4 concurrent agentic sessions.
"""

import sys
import os
import time
import json
import threading
import requests
from typing import Dict, Any, List

NPU_URL = "http://127.0.0.1:52625/v1/chat/completions"
GPU_URL = "http://127.0.0.1:8004/v1/chat/completions"
PROXY_URL = "http://127.0.0.1:13306/v1/chat/completions"

def generate_tool_churn_text(target_words: int) -> str:
    """Generates realistic synthetic agent tool logs (bash, tests, diffs)."""
    blocks = [
        "Running pytest on tests/test_cluster.py: PASSED [100%]. 42 passed in 1.45s.\n",
        "git status: On branch main, Changes not staged: modified: src/kernel.c, modified: src/driver.py\n",
        "gcc -O3 -mavx512f -c src/kernel.c -o build/kernel.o: Build successful, 0 warnings.\n",
        "curl -s http://127.0.0.1:8004/slots: slot 0 idle, slot 1 idle, slot 2 idle, slot 3 idle.\n",
        "Traceback analysis: resolved off-by-one pointer index in memory_allocator_v2.\n"
    ]
    repeated = []
    curr = 0
    idx = 0
    while curr < target_words:
        b = f"[{idx}] " + blocks[idx % len(blocks)]
        repeated.append(b)
        curr += len(b.split())
        idx += 1
    return "".join(repeated)

def bench_npu_compactor():
    print("\n" + "=" * 70)
    print(" 1. BENCHMARK: AMD XDNA 2 NPU AUXILIARY COMPACTOR (FastFlowLM)")
    print("=" * 70)
    test_sizes = [
        ("Short Activity (500 words ~ 650 tokens)", 500),
        ("Medium Activity (1,500 words ~ 2,000 tokens)", 1500),
        ("Large Activity (3,000 words ~ 4,000 tokens)", 3000)
    ]
    results = []
    for label, n_words in test_sizes:
        activity_text = generate_tool_churn_text(n_words)
        payload = {
            "model": "qwen3:0.6b",
            "messages": [
                {"role": "system", "content": "You are a concise context compactor. Summarize key technical findings in bullet points."},
                {"role": "user", "content": f"Summarize this agent activity trace:\n\n{activity_text}"}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }
        t0 = time.time()
        try:
            r = requests.post(NPU_URL, json=payload, timeout=45)
            dt = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                usage = data.get("usage", {})
                prefill_tps = usage.get("prefill_speed_tps", 0.0)
                decode_tps = usage.get("decoding_speed_tps", 0.0)
                prompt_tok = usage.get("prompt_tokens", 0)
                comp_tok = usage.get("completion_tokens", 0)
                summary_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                print(f" • {label}:")
                print(f"   - Wall-Clock Time:    {dt:.2f} s")
                print(f"   - Prompt Tokens:      {prompt_tok} tok")
                print(f"   - Generated Summary:  {comp_tok} tok ({len(summary_text)} chars)")
                print(f"   - NPU Prefill Speed:  {prefill_tps:.1f} tok/s")
                print(f"   - NPU Decode Speed:   {decode_tps:.1f} tok/s")
                results.append({
                    "label": label,
                    "wall_time_s": dt,
                    "prompt_tokens": prompt_tok,
                    "comp_tokens": comp_tok,
                    "prefill_tps": prefill_tps,
                    "decode_tps": decode_tps,
                    "summary_preview": summary_text[:120].replace("\n", " ")
                })
            else:
                print(f" • {label}: Failed with status {r.status_code}")
        except Exception as e:
            print(f" • {label}: Error: {e}")
    return results

def bench_lean_vs_bloated():
    print("\n" + "=" * 70)
    print(" 2. BENCHMARK: PRIMARY CLUSTER DECODE SPEED (Lean Compacted vs Bloated Context)")
    print("=" * 70)

    # 1. Lean Compacted Prompt (~500 tokens: compact rolling summary + recent turn)
    lean_prompt = (
        "[CONTEXT COMPACTED SUMMARY]:\n"
        "- Project: Strix Halo RPC-Tensor 2-Node Cluster Optimization\n"
        "- Files: cluster-dashboard.py, speculative_npu_proxy.py, continuous_compactor.py\n"
        "- Decisions: Disabled ngram-mod, enabled NPU auxiliary compactor on port 52625.\n"
        "- Status: Tests pass. Next task: Verify multi-session decode speed.\n\n"
        "User: Explain in 3 bullet points how continuous compaction prevents memory bandwidth degradation."
    )

    # 2. Bloated Context (~16,000 tokens of raw bash, compiler logs, and repeated diffs)
    bloated_raw = generate_tool_churn_text(12000)
    bloated_prompt = (
        f"Raw History Dumps:\n{bloated_raw}\n\n"
        "User: Explain in 3 bullet points how continuous compaction prevents memory bandwidth degradation."
    )

    test_cases = [
        ("Lean Compacted Context (Zero-Latency Compaction State)", lean_prompt),
        ("Bloated Raw Context (Uncompacted Tool Dumps)", bloated_prompt)
    ]

    results = []
    for label, prompt in test_cases:
        payload = {
            "model": "default",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.7,
            "stream": False
        }
        t0 = time.time()
        try:
            r = requests.post(GPU_URL, json=payload, timeout=120)
            dt = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                timings = data.get("timings", {})
                prompt_tok = timings.get("prompt_n", 0)
                prompt_ms = timings.get("prompt_ms", 0.0)
                prompt_tps = timings.get("prompt_per_second", 0.0)
                pred_tok = timings.get("predicted_n", 0)
                pred_ms = timings.get("predicted_ms", 0.0)
                pred_tps = timings.get("predicted_per_second", 0.0)

                print(f" • {label}:")
                print(f"   - Prompt Length:      {prompt_tok} tokens")
                print(f"   - Prefill Time:       {prompt_ms / 1000.0:.2f} s ({prompt_tps:.1f} tok/s)")
                print(f"   - Generated Tokens:   {pred_tok} tokens")
                print(f"   - Decode Speed:       {pred_tps:.2f} tok/s")
                print(f"   - Total Elapsed:      {dt:.2f} s")
                results.append({
                    "label": label,
                    "prompt_tokens": prompt_tok,
                    "prefill_s": prompt_ms / 1000.0,
                    "prefill_tps": prompt_tps,
                    "decode_tokens": pred_tok,
                    "decode_tps": pred_tps,
                    "total_s": dt
                })
            else:
                print(f" • {label}: Failed with status {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f" • {label}: Error: {e}")
    return results

def bench_concurrency(concurrency_levels=[1, 2, 4]):
    print("\n" + "=" * 70)
    print(" 3. BENCHMARK: MULTI-SESSION CONCURRENCY (Interleaved Agentic Workload)")
    print("=" * 70)

    concurrency_results = {}

    for n_sessions in concurrency_levels:
        print(f"\n--- Testing {n_sessions} Concurrent Session(s) ---")
        session_outputs = [None] * n_sessions
        session_durations = [0.0] * n_sessions
        session_tokens = [0] * n_sessions

        def run_session(s_id):
            prompt = (
                f"[Session {s_id}] You are an autonomous coding agent.\n"
                f"Summary: Modified module_{s_id}.py. Verified all unit tests pass.\n"
                f"Write a concise 4-step action plan for deploying module_{s_id}."
            )
            payload = {
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 120,
                "temperature": 0.7,
                "stream": False
            }
            t0 = time.time()
            try:
                r = requests.post(GPU_URL, json=payload, timeout=120)
                dt = time.time() - t0
                if r.status_code == 200:
                    data = r.json()
                    timings = data.get("timings", {})
                    pred_n = timings.get("predicted_n", 0)
                    session_tokens[s_id] = pred_n
                    session_durations[s_id] = dt
                    session_outputs[s_id] = data
            except Exception as e:
                print(f"Session {s_id} error: {e}")

        threads = []
        wall_start = time.time()
        for i in range(n_sessions):
            t = threading.Thread(target=run_session, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        wall_time = time.time() - wall_start
        total_tokens = sum(session_tokens)
        aggregate_tps = total_tokens / wall_time if wall_time > 0 else 0.0
        avg_session_dur = sum(session_durations) / len(session_durations) if session_durations else 0.0

        print(f" • Concurrency: {n_sessions} Session(s)")
        print(f"   - Total Tokens Generated: {total_tokens} tokens")
        print(f"   - Wall-Clock Time:        {wall_time:.2f} s")
        print(f"   - Aggregate Throughput:   {aggregate_tps:.2f} tok/s")
        print(f"   - Avg Session Latency:    {avg_session_dur:.2f} s")
        for i in range(n_sessions):
            if session_outputs[i]:
                timings = session_outputs[i].get("timings", {})
                dec_spd = timings.get("predicted_per_second", 0.0)
                print(f"     * Session {i}: {session_tokens[i]} tokens in {session_durations[i]:.2f}s ({dec_spd:.2f} tok/s)")

        concurrency_results[n_sessions] = {
            "n_sessions": n_sessions,
            "total_tokens": total_tokens,
            "wall_time_s": wall_time,
            "aggregate_tps": aggregate_tps,
            "avg_session_latency_s": avg_session_dur
        }

    return concurrency_results

if __name__ == "__main__":
    out_dir = "/home/alexzimmerman/benchmark_results"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = int(time.time())
    out_file = os.path.join(out_dir, f"continuous_compaction_bench_{timestamp}.json")

    npu_res = bench_npu_compactor()
    lean_bloat_res = bench_lean_vs_bloated()
    concurr_res = bench_concurrency([1, 2, 4])

    all_results = {
        "timestamp": timestamp,
        "npu_compactor": npu_res,
        "lean_vs_bloated": lean_bloat_res,
        "concurrency": concurr_res
    }

    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f" BENCHMARK COMPLETE — Results saved to {out_file}")
    print("=" * 70)
