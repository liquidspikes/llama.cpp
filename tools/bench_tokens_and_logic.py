#!/usr/bin/env python3
"""
Comprehensive Benchmark: Speed (Tokens/s) AND Deductive Logic Accuracy
Evaluates the Strix Halo Dual-Node Cluster across:
1. Mathematical Reasoning (Multi-step Arithmetic, Algebra, Modular Cycles)
2. Formal Deductive Logic (Knights & Knaves, Relational Ordering)
3. Multi-Turn Dependent Logic with <think> Evaporation (Turn 1 -> Turn 2)
4. Long-Context Rule Synthesis & Needle Deduction (Testing Continuous Compaction)
5. Algorithmic Code Synthesis & Execution Verification
"""

import os
import sys
import time
import json
import re
import urllib.request
import collections

sys.stdout.reconfigure(line_buffering=True)

PROXY_URL = "http://127.0.0.1:13306/v1/chat/completions"
NPU_URL = "http://127.0.0.1:52625/v1/chat/completions"
GPU_DIRECT_URL = "http://127.0.0.1:8001/v1/chat/completions"

def query_endpoint(url, payload, headers=None, timeout=300):
    t0 = time.time()
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    
    if "chat_template_kwargs" not in payload:
        payload["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": "xhigh"}
    if "temperature" not in payload:
        payload["temperature"] = 1.0
    if "top_p" not in payload:
        payload["top_p"] = 0.95
    if "repeat_penalty" not in payload:
        payload["repeat_penalty"] = 1.0
    if "max_tokens" not in payload:
        payload["max_tokens"] = 6144
    
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
    
    # In peg-native or deepseek format, thinking might be inside content if reasoning_content is empty
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
    source = text.strip() if text and text.strip() else (fallback_text.strip() if fallback_text else "")
    if not source:
        return ""
    m = re.findall(r"###\s*Final\s*:\s*([^\n\r]+)", source, re.IGNORECASE)
    if m:
        return m[-1].strip().strip("*#.").strip()
    m2 = re.findall(r"(?:final answer|answer is|final:?)\s*[:=]?\s*([^\n\r]+)", source, re.IGNORECASE)
    if m2:
        return m2[-1].strip().strip("*#.").strip()
    if text and text.strip():
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1].strip("*#.").strip() if lines else ""
    return ""

def run_logic_benchmark():
    results = []
    print("=" * 75)
    print(" STRIX HALO CLUSTER BENCHMARK: TOKENS/SEC & LOGICAL REASONING")
    print("=" * 75)
    print("Evaluating Math, Formal Logic, Multi-Turn CoT Evaporation & Compaction...\n")

    # -------------------------------------------------------------
    # SUITE 1: MATHEMATICAL & QUANTITATIVE REASONING
    # -------------------------------------------------------------
    math_tests = [
        {
            "id": "MATH-1",
            "name": "Multi-Step Inventory & Percentage Deduction",
            "prompt": (
                "A store has 120 apples. In the morning, they sell 35% of them. In the afternoon, "
                "they receive a delivery of 50 new apples, but 8 of the delivered apples are rotten and discarded. "
                "In the evening, a customer buys 2/3 of all remaining good apples. "
                "How many good apples are left in the store at closing? "
                "Work through the steps briefly, then write your final answer formatted exactly as: ### Final: [number]"
            ),
            "expected": "40",
            "check": lambda ans: "40" in ans
        },
        {
            "id": "MATH-2",
            "name": "Simultaneous Rate & Rational Arithmetic",
            "prompt": (
                "Pipe A can fill a tank in 6 hours. Pipe B can fill the same tank in 9 hours. "
                "Pipe C can empty the tank in 12 hours. If all three pipes are opened simultaneously "
                "when the tank is empty, how many hours will it take to fill the tank completely? "
                "Calculate step-by-step with concise reasoning (under 100 words), then write your final answer as a fraction (e.g. 36/7) or decimal: ### Final: [number]"
            ),
            "expected": "36/7 or 5.14",
            "check": lambda ans: "36/7" in ans or "5.14" in ans or "5.1" in ans
        },
        {
            "id": "MATH-3",
            "name": "Modular Arithmetic & Fermat Cycle",
            "prompt": (
                "Find the remainder when 3^2026 is divided by 7. "
                "Show the modular cycle of powers of 3 modulo 7 step-by-step with concise reasoning (under 100 words), then write: ### Final: [number]"
            ),
            "expected": "4",
            "check": lambda ans: ans.strip() == "4" or "Final: 4" in ans or "is 4" in ans or "4" in ans
        }
    ]

    for t in math_tests:
        print(f"--> Running [{t['id']}]: {t['name']}...")
        payload = {
            "model": "qwen3.8-flash-next-official",
            "messages": [{"role": "user", "content": t["prompt"]}],
            "max_tokens": 3072,
            "temperature": 1.0,
            "top_p": 0.95,
            "repeat_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
            "id_slot": 0
        }
        res = query_endpoint(PROXY_URL, payload)
        final_ans = extract_final(res["content"], res["reasoning"])
        is_correct = t["check"](final_ans)
        
        print(f"    Correct: {is_correct} | Got: '{final_ans}' (Expected: '{t['expected']}')")
        print(f"    Prefill: {res['prompt_tps']:.1f} tok/s ({res['prompt_n']} tok in {res['prompt_ms']:.1f}ms)")
        print(f"    Decode:  {res['pred_tps']:.1f} tok/s ({res['pred_n']} tok in {res['pred_ms']:.1f}ms)")
        print(f"    Wall:    {res['wall_sec']:.2f}s | Reasoning length: {len(res['reasoning'])} chars\n")
        
        results.append({
            "category": "Math & Quantitative",
            "id": t["id"],
            "name": t["name"],
            "correct": is_correct,
            "expected": t["expected"],
            "got": final_ans,
            "prompt_tok": res["prompt_n"],
            "prompt_tps": res["prompt_tps"],
            "pred_tok": res["pred_n"],
            "pred_tps": res["pred_tps"],
            "wall_sec": res["wall_sec"]
        })

    # -------------------------------------------------------------
    # SUITE 2: FORMAL DEDUCTIVE LOGIC & CONSTRAINT PUZZLES
    # -------------------------------------------------------------
    logic_tests = [
        {
            "id": "LOGIC-1",
            "name": "Knights & Knaves Propositional Deduction",
            "prompt": (
                "On an island, Knights always tell the truth and Knaves always lie. "
                "You meet two inhabitants, Alice and Bob. "
                "Alice says: 'At least one of us is a knave.' "
                "Bob says: 'Alice and I are of the same type.' "
                "Determine the identity of each person. Keep your reasoning concise (under 100 words), "
                "then end with: ### Final: Alice is a [Knight/Knave], Bob is a [Knight/Knave]"
            ),
            "expected": "Alice is a Knight, Bob is a Knave",
            "check": lambda ans: "alice is a knight" in ans.lower() and "bob is a knave" in ans.lower()
        },
        {
            "id": "LOGIC-2",
            "name": "5-Way Relational & Temporal Ordering",
            "prompt": (
                "Five runners—Alex, Blake, Casey, Dylan, and Evan—finished a race in 1st through 5th place with no ties.\n"
                "1. Alex finished before Casey.\n"
                "2. Dylan finished between Alex and Casey.\n"
                "3. Blake finished immediately after Dylan.\n"
                "4. Evan finished either first or last.\n"
                "5. Casey did not finish last.\n"
                "Deduce the exact finish order. Keep your reasoning concise (under 150 words). Conclude with: ### Final: 1st: [name], 2nd: [name], 3rd: [name], 4th: [name], 5th: [name]"
            ),
            "expected": "1st: Alex, 2nd: Dylan, 3rd: Blake, 4th: Casey, 5th: Evan",
            "check": lambda ans: (
                "alex" in ans.lower() and "dylan" in ans.lower() and "blake" in ans.lower() and
                "casey" in ans.lower() and "evan" in ans.lower() and
                ans.lower().find("alex") < ans.lower().find("dylan") < ans.lower().find("blake") < ans.lower().find("casey") < ans.lower().find("evan")
            )
        }
    ]

    for t in logic_tests:
        print(f"--> Running [{t['id']}]: {t['name']}...")
        payload = {
            "model": "qwen3.8-flash-next-official",
            "messages": [{"role": "user", "content": t["prompt"]}],
            "max_tokens": 3072,
            "temperature": 1.0,
            "top_p": 0.95,
            "repeat_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
            "id_slot": 0
        }
        res = query_endpoint(PROXY_URL, payload)
        final_ans = extract_final(res["content"], res["reasoning"])
        is_correct = t["check"](final_ans)
        
        print(f"    Correct: {is_correct} | Got: '{final_ans}'")
        print(f"    Prefill: {res['prompt_tps']:.1f} tok/s ({res['prompt_n']} tok in {res['prompt_ms']:.1f}ms)")
        print(f"    Decode:  {res['pred_tps']:.1f} tok/s ({res['pred_n']} tok in {res['pred_ms']:.1f}ms)")
        print(f"    Wall:    {res['wall_sec']:.2f}s\n")
        
        results.append({
            "category": "Formal Deductive Logic",
            "id": t["id"],
            "name": t["name"],
            "correct": is_correct,
            "expected": t["expected"],
            "got": final_ans,
            "prompt_tok": res["prompt_n"],
            "prompt_tps": res["prompt_tps"],
            "pred_tok": res["pred_n"],
            "pred_tps": res["pred_tps"],
            "wall_sec": res["wall_sec"]
        })

    # -------------------------------------------------------------
    # SUITE 3: MULTI-TURN REASONING: <think> EVAPORATION VS PRESERVATION
    # -------------------------------------------------------------
    print("--> Running [MULTI-TURN]: Testing State Continuity under <think> Evaporation...")
    turn1_prompt = (
        "A warehouse starts on Monday with 500 crates of widgets.\n"
        "On Tuesday, shipment Inbound-A arrives with 120 crates, and 80 crates are shipped out to Client-1.\n"
        "On Wednesday, 15% of the total crates currently in stock are damaged in a flood and quarantined (round down to nearest whole crate). Then shipment Inbound-B arrives with 95 crates.\n"
        "What is the exact count of available (non-quarantined) crates at the end of Wednesday?\n"
        "Calculate step-by-step with concise reasoning (under 100 words) and conclude with: ### Final: [number]"
    )
    # Turn 1: 500 + 120 - 80 = 540. 15% of 540 = 81 quarantined. Remaining = 459. + 95 = 554.
    turn1_res = query_endpoint(PROXY_URL, {
        "model": "qwen3.8-flash-next-official",
        "messages": [{"role": "user", "content": turn1_prompt}],
        "max_tokens": 3072,
        "temperature": 1.0,
        "top_p": 0.95,
        "repeat_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
        "id_slot": 1
    })
    t1_ans = extract_final(turn1_res["content"], turn1_res["reasoning"])
    t1_correct = "554" in t1_ans
    print(f"    Turn 1 (Base State): Correct={t1_correct} | Got={t1_ans} (Expected: 554)")
    print(f"    Turn 1 Reasoning Length: {len(turn1_res['reasoning'])} chars")

    turn2_prompt = (
        "Continuing from Wednesday's closing inventory (which had 554 available crates and 81 quarantined crates):\n"
        "On Thursday, the 81 quarantined crates are inspected. It is found that exactly 1/3 of them are undamaged and returned to available stock, while the rest are destroyed.\n"
        "In addition, Client-2 orders 180 crates and Client-3 orders 75 crates, both fulfilled from available stock.\n"
        "How many available crates remain in the warehouse at the end of Thursday?\n"
        "Calculate step-by-step with concise reasoning (under 100 words) and conclude with: ### Final: [number]"
    )
    # Turn 2: 554 + 27 - 180 - 75 = 326.
    
    # Sub-test A: Full Reasoning Preserved (Assistant message contains reasoning + content)
    full_turn1_content = f"<think>\n{turn1_res['reasoning']}\n</think>\n\n{turn1_res['content']}" if turn1_res['reasoning'] else turn1_res['content']
    msgs_preserved = [
        {"role": "user", "content": turn1_prompt},
        {"role": "assistant", "content": full_turn1_content},
        {"role": "user", "content": turn2_prompt}
    ]
    t0 = time.time()
    res_preserved = query_endpoint(PROXY_URL, {
        "model": "qwen3.8-flash-next-official",
        "messages": msgs_preserved,
        "max_tokens": 3072,
        "temperature": 1.0,
        "top_p": 0.95,
        "repeat_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
        "id_slot": 1
    }, headers={"X-Evaporate-Thinking": "false"})
    ans_preserved = extract_final(res_preserved["content"], res_preserved["reasoning"])
    correct_preserved = "326" in ans_preserved

    # Sub-test B: <think> Evaporation (Thinking stripped from history)
    msgs_evaporated = [
        {"role": "user", "content": turn1_prompt},
        {"role": "assistant", "content": full_turn1_content},
        {"role": "user", "content": turn2_prompt}
    ]
    res_evaporated = query_endpoint(PROXY_URL, {
        "model": "qwen3.8-flash-next-official",
        "messages": msgs_evaporated,
        "max_tokens": 3072,
        "temperature": 1.0,
        "top_p": 0.95,
        "repeat_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
        "id_slot": 1
    }, headers={"X-Evaporate-Thinking": "true"})
    ans_evaporated = extract_final(res_evaporated["content"], res_evaporated["reasoning"])
    correct_evaporated = "326" in ans_evaporated

    print(f"\n    [Sub-test A: Thinking Preserved in History]:")
    print(f"      Correct: {correct_preserved} | Got: '{ans_preserved}' (Expected: '326')")
    print(f"      Prompt Tokens: {res_preserved['prompt_n']} | Prompt Time: {res_preserved['prompt_ms']:.1f}ms")
    print(f"      Decode Speed:  {res_preserved['pred_tps']:.1f} tok/s | Wall: {res_preserved['wall_sec']:.2f}s")

    print(f"\n    [Sub-test B: <think> Evaporation Enabled]:")
    print(f"      Correct: {correct_evaporated} | Got: '{ans_evaporated}' (Expected: '326')")
    print(f"      Prompt Tokens: {res_evaporated['prompt_n']} (Saved {res_preserved['prompt_n'] - res_evaporated['prompt_n']} tokens)")
    print(f"      Prompt Time:   {res_evaporated['prompt_ms']:.1f}ms")
    print(f"      Decode Speed:  {res_evaporated['pred_tps']:.1f} tok/s | Wall: {res_evaporated['wall_sec']:.2f}s\n")

    results.append({
        "category": "Multi-Turn Thinking Preserved",
        "id": "TURN-PRESERVED",
        "name": "Turn 2 with full <think> in history",
        "correct": correct_preserved,
        "expected": "326",
        "got": ans_preserved,
        "prompt_tok": res_preserved["prompt_n"],
        "prompt_tps": res_preserved["prompt_tps"],
        "pred_tok": res_preserved["pred_n"],
        "pred_tps": res_preserved["pred_tps"],
        "wall_sec": res_preserved["wall_sec"]
    })

    results.append({
        "category": "<think> Evaporation Optimized",
        "id": "TURN-EVAPORATED",
        "name": "Turn 2 with dead <think> evaporated",
        "correct": correct_evaporated,
        "expected": "326",
        "got": ans_evaporated,
        "prompt_tok": res_evaporated["prompt_n"],
        "prompt_tps": res_evaporated["prompt_tps"],
        "pred_tok": res_evaporated["pred_n"],
        "pred_tps": res_evaporated["pred_tps"],
        "wall_sec": res_evaporated["wall_sec"]
    })

    # -------------------------------------------------------------
    # SUITE 4: CODE & ALGORITHMIC LOGIC SYNTHESIS
    # -------------------------------------------------------------
    print("--> Running [CODE-1]: Sliding Window Maximum (Algorithmic Logic)...")
    code_prompt = (
        "Write a Python function `max_sliding_window(nums: list[int], k: int) -> list[int]` "
        "that finds the maximum value in every sliding window of size k moving from left to right. "
        "Your implementation MUST run in O(n) time using `collections.deque`. "
        "Provide ONLY Python code enclosed in ```python ``` blocks with no markdown outside."
    )
    res_code = query_endpoint(PROXY_URL, {
        "model": "qwen3.8-flash-next-official",
        "messages": [{"role": "user", "content": code_prompt}],
        "max_tokens": 3072,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
        "id_slot": 0
    })
    
    # Extract code and test
    code_match = re.search(r"```python([\s\S]*?)```", res_code["content"])
    raw_code = code_match.group(1).strip() if code_match else res_code["content"]
    
    test_env = {"collections": collections, "deque": collections.deque, "__builtins__": __builtins__}
    exec_passed = False
    error_msg = ""
    try:
        exec(raw_code, test_env)
        fn = test_env.get("max_sliding_window")
        if fn:
            assert fn([1,3,-1,-3,5,3,6,7], 3) == [3,3,5,5,6,7], "Case 1 failed"
            assert fn([1], 1) == [1], "Case 2 failed"
            assert fn([1,-1], 1) == [1,-1], "Case 3 failed"
            assert fn([9,11], 2) == [11], "Case 4 failed"
            assert fn([4,-2], 2) == [4], "Case 5 failed"
            exec_passed = True
        else:
            error_msg = "Function max_sliding_window not defined in code"
    except Exception as err:
        error_msg = str(err)
        
    print(f"    Code Unit Tests: {'PASSED' if exec_passed else 'FAILED (' + error_msg + ')'}")
    print(f"    Decode:  {res_code['pred_tps']:.1f} tok/s ({res_code['pred_n']} tok in {res_code['pred_ms']:.1f}ms)")
    print(f"    Wall:    {res_code['wall_sec']:.2f}s\n")
    
    results.append({
        "category": "Algorithmic Code Logic",
        "id": "CODE-1",
        "name": "Sliding Window Max O(N) Deque",
        "correct": exec_passed,
        "expected": "Pass 5/5 unit tests",
        "got": "Passed 5/5" if exec_passed else f"Failed: {error_msg}",
        "prompt_tok": res_code["prompt_n"],
        "prompt_tps": res_code["prompt_tps"],
        "pred_tok": res_code["pred_n"],
        "pred_tps": res_code["pred_tps"],
        "wall_sec": res_code["wall_sec"]
    })

    # -------------------------------------------------------------
    # SUITE 5: CONTINUOUS COMPACTION NEEDLE & POLICY REASONING
    # -------------------------------------------------------------
    print("--> Running [COMPACTION-LOGIC]: Long-Context Needle & Policy Synthesis...")
    # Generate 15k tokens of architectural context with embedded SLA rules
    filler_chunk = (
        "The cluster storage subsystem manages distributed block extents using non-blocking RDMA descriptors. "
        "Each segment is replicated across two storage controllers with quorum heartbeat intervals set to 250ms. "
        "Network telemetry indicates that packet jitter remains below 12 microseconds under normal operating parameters. "
        "All telemetry packets are stamped with hardware monotonic timers before submission to the ring buffer.\n"
    )
    long_filler_1 = filler_chunk * 120 # ~3,000 tokens
    
    sla_needle = (
        "\n\n==================== SERVICE LEVEL AGREEMENT (POLICY 4.12) ====================\n"
        "SECTION 4.12.1: ACCOUNTS WITH TIER 'PLATINUM' LOCATED IN REGION 'APAC':\n"
        "1. Base compensation penalty multiplier is exactly 4.5x the base hourly rate per hour of unplanned downtime.\n"
        "2. If total continuous incident duration exceeds 180 minutes, the customer is entitled to an automatic "
        "fixed lump-sum outage credit of $50,000, which is added to the penalty multiplier compensation.\n"
        "=================================================================================\n\n"
    )
    long_filler_2 = filler_chunk * 120 # ~3,000 tokens
    
    sla_query = (
        f"{long_filler_1}{sla_needle}{long_filler_2}\n\n"
        "QUESTION:\n"
        "Account 'ZephyrTech' is classified as Tier PLATINUM in region APAC. "
        "Their agreed base hourly fee is $1,200. Yesterday they experienced an unplanned continuous outage of 240 minutes (4.0 hours).\n"
        "Based on POLICY 4.12 in the text above, calculate the total outage credit owed to ZephyrTech (hourly multiplier penalty + lump-sum credit).\n"
        "Keep your reasoning concise (under 100 words). Show your calculation and conclude with: ### Final: $<number>"
    )
    # 4 hours * $1200 * 4.5 = $21,600. Duration 240 > 180 min -> +$50,000. Total = $71,600.
    
    res_needle = query_endpoint(PROXY_URL, {
        "model": "qwen3.8-flash-next-official",
        "messages": [{"role": "user", "content": sla_query}],
        "max_tokens": 3072,
        "temperature": 1.0,
        "top_p": 0.95,
        "repeat_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
        "id_slot": 0
    })
    ans_needle = extract_final(res_needle["content"], res_needle["reasoning"])
    correct_needle = "71600" in ans_needle.replace(",", "").replace("$", "") or "71,600" in ans_needle or "71600" in res_needle["content"] or "71600" in res_needle["reasoning"]
    
    print(f"    SLA Policy Needle Deduction: Correct={correct_needle} | Got='{ans_needle}' (Expected: '$71,600')")
    print(f"    Prompt Tokens: {res_needle['prompt_n']} | Prefill Speed: {res_needle['prompt_tps']:.1f} tok/s")
    print(f"    Decode Speed:  {res_needle['pred_tps']:.1f} tok/s | Wall: {res_needle['wall_sec']:.2f}s\n")

    results.append({
        "category": "Long-Context Policy Deduction",
        "id": "NEEDLE-SLA",
        "name": "6k+ Token Policy Needle Deduction",
        "correct": correct_needle,
        "expected": "$71,600",
        "got": ans_needle,
        "prompt_tok": res_needle["prompt_n"],
        "prompt_tps": res_needle["prompt_tps"],
        "pred_tok": res_needle["pred_n"],
        "pred_tps": res_needle["pred_tps"],
        "wall_sec": res_needle["wall_sec"]
    })

    # -------------------------------------------------------------
    # SUMMARY & REPORT GENERATION
    # -------------------------------------------------------------
    print("=" * 75)
    print(" BENCHMARK COMPLETE: SUMMARY OF TOKENS/SEC & LOGICAL ACCURACY")
    print("=" * 75)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["correct"])
    accuracy = (passed_tests / total_tests) * 100.0
    avg_decode = sum(r["pred_tps"] for r in results) / total_tests
    
    print(f"Total Logical Tests: {total_tests}")
    print(f"Logic Accuracy:      {passed_tests}/{total_tests} ({accuracy:.1f}%)")
    print(f"Average Decode:      {avg_decode:.2f} tok/s")
    print("-" * 75)
    for r in results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"[{status}] {r['id']:<15} | Got: {r['got'][:25]:<25} | {r['pred_tps']:>5.1f} tok/s | {r['wall_sec']:>5.2f}s | {r['name']}")
    print("=" * 75)
    
    output_path = "/home/alexzimmerman/llama.cpp/benchmark_results/tokens_and_logic_benchmark.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "accuracy_pct": accuracy,
            "average_decode_tps": avg_decode,
            "results": results
        }, f, indent=2)
    print(f"\nDetailed JSON report saved to: {output_path}")

if __name__ == "__main__":
    run_logic_benchmark()
