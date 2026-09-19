#!/usr/bin/env python3
"""
Large Prompt Logic & Deductive Integrity Verification
Tests the Dual-Node Strix Halo Cluster under:
1. 27,000+ Token Dense Enterprise RTGS Audit Context & Regulatory Rule Deduction
2. Formal Propositional Logic (Modus Tollens & Multi-Premise Truth Satisfaction)
"""

import sys
import time
import json
import re
import urllib.request

sys.stdout.reconfigure(line_buffering=True)

PROXY_URL = "http://127.0.0.1:13306/v1/chat/completions"

def query_endpoint(url, payload, headers=None, timeout=600):
    t0 = time.time()
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    
    if "temperature" not in payload:
        payload["temperature"] = 0.6
    if "top_p" not in payload:
        payload["top_p"] = 0.95
    if "repeat_penalty" not in payload:
        payload["repeat_penalty"] = 1.1
    if "max_tokens" not in payload:
        payload["max_tokens"] = 1536
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=req_headers)
    
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    
    wall_sec = time.time() - t0
    choice = res["choices"][0]["message"]
    timings = res.get("timings", {})
    usage = res.get("usage", {})
    
    prompt_n = timings.get("prompt_n", usage.get("prompt_tokens", 0))
    prompt_ms = timings.get("prompt_ms", 0.0)
    prompt_tps = timings.get("prompt_per_second", (prompt_n / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0.0)
    
    pred_n = timings.get("predicted_n", usage.get("completion_tokens", 0))
    pred_ms = timings.get("predicted_ms", 0.0)
    pred_tps = timings.get("predicted_per_second", (pred_n / (pred_ms / 1000.0)) if pred_ms > 0 else 0.0)
    
    content = choice.get("content", "")
    reasoning = choice.get("reasoning_content", "")
    
    if not reasoning and "<think>" in content and "</think>" in content:
        m = re.search(r"<think>([\s\S]*?)</think>", content)
        if m:
            reasoning = m.group(1).strip()
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            
    return {
        "content": content.strip(),
        "reasoning": reasoning.strip(),
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "prompt_tps": prompt_tps,
        "pred_n": pred_n,
        "pred_ms": pred_ms,
        "pred_tps": pred_tps,
        "wall_sec": wall_sec,
        "raw_response": res
    }

def extract_final(text, fallback_text=""):
    for src in [text, fallback_text]:
        if not src:
            continue
        m = re.findall(r"###\s*Final\s*:\s*([^\n\r]+)", src, re.IGNORECASE)
        if m:
            return m[-1].strip().strip("*#.").strip()
        m2 = re.findall(r"(?:final answer|answer is)\s*[:=]?\s*([^\n\r]+)", src, re.IGNORECASE)
        if m2:
            return m2[-1].strip().strip("*#.").strip()
    if text and text.strip():
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1].strip("*#.").strip() if lines else ""
    return ""

def run_large_prompt_test():
    print("=" * 80)
    print(" STRIX HALO CLUSTER: LARGE PROMPT LOGICAL FIDELITY & REASONING BENCHMARK")
    print("=" * 80)
    print("Unified 1M KV Cache | Dual ROCm+RPC over USB4 40 Gbps\n")

    # -------------------------------------------------------------
    # TEST 1: 27,000+ TOKEN RTGS FINANCIAL AUDIT & MULTI-STEP LOGIC
    # -------------------------------------------------------------
    print("Generating ~27,000 token enterprise RTGS audit context...")
    
    header = (
        "GLOBAL SETTLEMENT AND CLEARING PROTOCOL (GSCP v8.4)\n"
        "AND REAL-TIME GROSS SETTLEMENT (RTGS) REGULATORY DIRECTIVE\n"
        "================================================================================\n\n"
        "SECTION 1: OVERVIEW AND SYSTEMIC LIQUIDITY GUIDELINES\n"
        "All participating clearing members maintain real-time settlement reserves. "
        "Transactions are continuously validated against intraday credit lines, net debit caps, "
        "and bilateral clearing limits established by the central clearing authority.\n\n"
    )
    
    log_entry_template = (
        "[2026-09-18T{hour:02d}:{min:02d}:{sec:02d}.{ms:03d}Z] RTGS-NODE-{node:02d} "
        "MSG_ID={msg_id:08d} ROUTING=SWIFT-MT103 ACCT_SRC=ACCT-US-{src:05d} "
        "ACCT_DST=ACCT-EU-{dst:05d} AMOUNT={amt:.2f} CURR=USD STATUS=SETTLED "
        "CONFIRMATION_HASH=0x{hash_val:016x} LATENCY={lat:.2f}ms NONCE={nonce}\n"
    )
    
    entries_part1 = []
    for i in range(1, 51):
        h = (i // 60) % 24
        m = i % 60
        entries_part1.append(log_entry_template.format(
            hour=h, min=m, sec=12, ms=i*2 % 999, node=(i % 4) + 1,
            msg_id=10000000 + i, src=1000 + (i*7 % 5000), dst=6000 + (i*13 % 4000),
            amt=12500.0 + (i * 187.5 % 95000.0), hash_val=0xdeadbeef10000000 + i*1337,
            lat=1.2 + (i % 5)*0.3, nonce=i
        ))
    part1_text = "".join(entries_part1)
    
    policy_needle_1 = (
        "\n\n================================================================================\n"
        "REGULATORY STATUTE SECTION 7.3: TIERED LIQUIDITY HAIRCUTS AND SURCHARGES\n"
        "1. The baseline haircut on uncollateralized intraday overdrafts is exactly 3.50%.\n"
        "2. If an institution incurs an uncollateralized overdraft exceeding $10,000,000 "
        "during the critical settlement window (14:00 to 16:00 UTC), a regulatory penalty multiplier "
        "of exactly 1.75x is applied to the baseline haircut rate.\n"
        "================================================================================\n\n"
    )
    
    entries_part2 = []
    for i in range(51, 101):
        h = (i // 60) % 24
        m = i % 60
        entries_part2.append(log_entry_template.format(
            hour=h, min=m, sec=33, ms=i*3 % 999, node=(i % 4) + 1,
            msg_id=10000000 + i, src=2000 + (i*11 % 5000), dst=7000 + (i*17 % 4000),
            amt=25000.0 + (i * 243.2 % 88000.0), hash_val=0xcafebabe20000000 + i*2741,
            lat=1.4 + (i % 7)*0.2, nonce=i
        ))
    part2_text = "".join(entries_part2)
    
    policy_needle_2 = (
        "\n\n================================================================================\n"
        "REGULATORY STATUTE SECTION 9.4: CROSS-BORDER DOUBLE-SPEND & CLAWBACK SANCTIONS\n"
        "1. Any cross-border leg presenting conflicting cryptographic proofs shall be immediately flagged.\n"
        "2. Transaction ID 'TX-88294-OMEGA' submitted by institution 'ApexLedger' is hereby REJECTED.\n"
        "3. Rejection under Section 9.4 mandates an automated fixed statutory clawback penalty of exactly $145,000.\n"
        "4. COLLATERAL BALANCE RULE: Final Available Collateral = Initial Collateral minus Settled Obligations "
        "minus the Section 9.4 Statutory Clawback minus the Section 7.3 Overdraft Penalty Surcharge.\n"
        "================================================================================\n\n"
    )
    
    entries_part3 = []
    for i in range(101, 151):
        h = (i // 60) % 24
        m = i % 60
        entries_part3.append(log_entry_template.format(
            hour=h, min=m, sec=55, ms=i*5 % 999, node=(i % 4) + 1,
            msg_id=10000000 + i, src=3000 + (i*19 % 5000), dst=8000 + (i*23 % 4000),
            amt=42000.0 + (i * 311.7 % 120000.0), hash_val=0xfeedface30000000 + i*3917,
            lat=1.1 + (i % 4)*0.4, nonce=i
        ))
    part3_text = "".join(entries_part3)
    
    institution_facts = (
        "\n\n================================================================================\n"
        "INCIDENT REPORT: INSTITUTION 'ApexLedger' DAILY CLEARING RECONCILIATION\n"
        "- Initial Morning Collateral Deposited: $12,500,000.00 USD\n"
        "- Approved and Completed Settled Obligations: $4,250,000.00 USD\n"
        "- Incident Timestamp: 14:32:15 UTC (during the 14:00 to 16:00 UTC critical settlement window)\n"
        "- Uncollateralized Intraday Overdraft Amount: $16,000,000.00 USD (exceeds $10,000,000 threshold)\n"
        "- Flagged Violation Event: Cross-border double-spend leg TX-88294-OMEGA\n"
        "================================================================================\n\n"
    )
    
    entries_part4 = []
    for i in range(151, 201):
        h = (i // 60) % 24
        m = i % 60
        entries_part4.append(log_entry_template.format(
            hour=h, min=m, sec=41, ms=i*7 % 999, node=(i % 4) + 1,
            msg_id=10000000 + i, src=4000 + (i*29 % 5000), dst=9000 + (i*31 % 4000),
            amt=38000.0 + (i * 419.1 % 75000.0), hash_val=0xbaadf00d40000000 + i*5113,
            lat=1.3 + (i % 6)*0.25, nonce=i
        ))
    part4_text = "".join(entries_part4)
    
    query = (
        "AUDIT DIRECTIVE AND QUANTITATIVE REASONING TASK:\n"
        "Based STRICTLY on the regulatory statutes (Section 7.3 and Section 9.4) and the incident report for 'ApexLedger' in the documentation above:\n"
        "1. Identify the rejected transaction ID that violated Section 9.4.\n"
        "2. Calculate the exact penalty surcharge fee for ApexLedger's $16,000,000 overdraft:\n"
        "   - Determine the effective haircut percentage rate (base 3.50% multiplied by the Section 7.3 critical window penalty multiplier 1.75x).\n"
        "   - Multiply this effective percentage rate by the $16,000,000 overdraft to find the surcharge dollar fee.\n"
        "3. Apply the Collateral Balance Rule from Section 9.4:\n"
        "   - Final Available Collateral = Initial Collateral ($12,500,000) minus Settled Obligations ($4,250,000) minus Statutory Clawback ($145,000) minus Overdraft Penalty Surcharge.\n"
        "4. Be concise in your reasoning (under 60 words). State the final answer formatted exactly as:\n"
        "### Final: Rejected TX: [TX-ID] | Surcharge: $[number] | Available Balance: $[number]"
    )
    
    full_large_prompt = f"{header}{part1_text}{policy_needle_1}{part2_text}{policy_needle_2}{part3_text}{institution_facts}{part4_text}\n\n{query}"
    
    print(f"Submitting Large Prompt ({len(full_large_prompt)} characters, ~27k tokens) to Port 13306...")
    res_large = query_endpoint(PROXY_URL, {
        "model": "qwen3.8-flash-next-official",
        "messages": [{"role": "user", "content": full_large_prompt}],
        "max_tokens": 1536,
        "temperature": 0.6,
        "top_p": 0.95,
        "repeat_penalty": 1.1
    })
    
    ans_large = extract_final(res_large["content"], res_large["reasoning"])
    full_resp = (res_large["content"] + " " + res_large["reasoning"]).lower()
    
    tx_ok = "tx-88294-omega" in full_resp or "tx-88294-omega" in ans_large.lower()
    surcharge_ok = "980,000" in full_resp or "980000" in full_resp
    balance_ok = "7,125,000" in full_resp or "7125000" in full_resp
    all_large_ok = tx_ok and surcharge_ok and balance_ok
    
    print("-" * 80)
    print(f"[TEST 1: 27k+ TOKEN ENTERPRISE RTGS LOGIC & DEDUCTION]")
    print(f"  Prompt Tokens:  {res_large['prompt_n']} tokens")
    print(f"  Prefill Speed:  {res_large['prompt_tps']:.1f} tok/s ({res_large['prompt_ms']:.1f}ms)")
    print(f"  Decode Speed:   {res_large['pred_tps']:.1f} tok/s ({res_large['pred_n']} tok in {res_large['pred_ms']:.1f}ms)")
    print(f"  Wall Clock:     {res_large['wall_sec']:.2f}s")
    print(f"  Output Content:\n{res_large['content']}")
    if res_large["reasoning"]:
        print(f"  Reasoning Summary:\n{res_large['reasoning'][:300]}...")
    print(f"  Extracted Final: '{ans_large}'")
    print(f"  Verification:   TX_OK={tx_ok} | SURCHARGE_OK={surcharge_ok} | BALANCE_OK={balance_ok}")
    print(f"  Result:         {'PASS (Logic Intact Under Large Prompt)' if all_large_ok else 'FAIL'}")
    print("-" * 80)

    # -------------------------------------------------------------
    # TEST 2: MULTI-STEP PROPOSITIONAL LOGIC (MODUS TOLLENS)
    # -------------------------------------------------------------
    print("\n--> Running [TEST 2]: Complex Multi-Premise Propositional Deduction (Modus Tollens)...")
    formal_logic_prompt = (
        "Consider five logical propositions: P, Q, R, S, and T.\n"
        "Premise 1: If (P and Q), then (not R or S).\n"
        "Premise 2: (not S) is True, and P is True.\n"
        "Premise 3: If T, then R.\n"
        "Premise 4: T is True.\n\n"
        "Deduce the truth value of P, Q, R, S, and T. Be extremely concise in your reasoning (under 50 words). Conclude with:\n"
        "### Final: P: [True/False], Q: [True/False], R: [True/False], S: [True/False], T: [True/False]"
    )
    
    res_logic = query_endpoint(PROXY_URL, {
        "model": "qwen3.8-flash-next-official",
        "messages": [{"role": "user", "content": formal_logic_prompt}],
        "max_tokens": 1024,
        "temperature": 0.6,
        "top_p": 0.95,
        "repeat_penalty": 1.1
    })
    
    ans_logic = extract_final(res_logic["content"], res_logic["reasoning"])
    full_logic = (res_logic["content"] + " " + res_logic["reasoning"]).lower()
    
    p_ok = "p: true" in ans_logic.lower() or "p is true" in full_logic or "p = true" in full_logic
    q_ok = "q: false" in ans_logic.lower() or "q is false" in full_logic or "q = false" in full_logic
    r_ok = "r: true" in ans_logic.lower() or "r is true" in full_logic or "r = true" in full_logic
    s_ok = "s: false" in ans_logic.lower() or "s is false" in full_logic or "s = false" in full_logic
    t_ok = "t: true" in ans_logic.lower() or "t is true" in full_logic or "t = true" in full_logic
    logic_passed = p_ok and q_ok and r_ok and s_ok and t_ok
    
    print("-" * 80)
    print(f"[TEST 2: MULTI-PREMISE PROPOSITIONAL LOGIC (MODUS TOLLENS)]")
    print(f"  Prefill Speed:  {res_logic['prompt_tps']:.1f} tok/s ({res_logic['prompt_n']} tok)")
    print(f"  Decode Speed:   {res_logic['pred_tps']:.1f} tok/s ({res_logic['pred_n']} tok)")
    print(f"  Wall Clock:     {res_logic['wall_sec']:.2f}s")
    print(f"  Output Content:\n{res_logic['content']}")
    if res_logic["reasoning"]:
        print(f"  Reasoning Summary:\n{res_logic['reasoning'][:300]}...")
    print(f"  Extracted Final: '{ans_logic}'")
    print(f"  Variables:      P={p_ok}, Q(Key)={q_ok}, R={r_ok}, S={s_ok}, T={t_ok}")
    print(f"  Result:         {'PASS (Flawless Modus Tollens Proof)' if logic_passed else 'FAIL'}")
    print("=" * 80)
    
    output_path = "/home/alexzimmerman/llama.cpp/benchmark_results/large_prompt_logic_result.json"
    with open(output_path, "w") as f:
        json.dump({
            "test1_large_prompt": {
                "prompt_tokens": res_large["prompt_n"],
                "prefill_tps": res_large["prompt_tps"],
                "decode_tps": res_large["pred_tps"],
                "wall_sec": res_large["wall_sec"],
                "tx_ok": tx_ok,
                "surcharge_ok": surcharge_ok,
                "balance_ok": balance_ok,
                "passed": all_large_ok,
                "content": res_large["content"],
                "reasoning": res_large["reasoning"],
                "final": ans_large
            },
            "test2_formal_logic": {
                "prompt_tokens": res_logic["prompt_n"],
                "prefill_tps": res_logic["prompt_tps"],
                "decode_tps": res_logic["pred_tps"],
                "wall_sec": res_logic["wall_sec"],
                "passed": logic_passed,
                "p_ok": p_ok, "q_ok": q_ok, "r_ok": r_ok, "s_ok": s_ok, "t_ok": t_ok,
                "content": res_logic["content"],
                "reasoning": res_logic["reasoning"],
                "final": ans_logic
            }
        }, f, indent=2)
    print(f"\nDetailed JSON report written to: {output_path}")

if __name__ == "__main__":
    run_large_prompt_test()
