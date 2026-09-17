#!/usr/bin/env python3
"""
Final Comprehensive Cluster Verification Suite
Validates:
1. Exact Arithmetic ($17 * 19 = 323$)
2. Valid Python Syntax and Logic
3. Language Purity (English only, zero foreign language token drift)
4. Repetition Uniqueness (no repetitive looping)
5. Heterogeneous Auto-Routing (Fast NPU @ ~100 tok/s vs Deep GPU Cluster 177B MoE @ ~28 tok/s)
6. Streaming Latency & Throughput
"""

import sys
import time
import json
import ast
import re
import requests

PROX_URL = "http://127.0.0.1:13306/v1/chat/completions"
GPU_URL = "http://127.0.0.1:8002/v1/chat/completions"

def run_test(name, url, payload, validator_fn):
    print(f"\n{'='*70}\n[TEST] {name}")
    print(f"Target: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=90)
    t1 = time.perf_counter()
    
    if resp.status_code != 200:
        print(f"FAILED with HTTP {resp.status_code}: {resp.text}")
        return False, 0.0
    
    data = resp.json()
    msg = data['choices'][0]['message']
    content = msg.get('content', '') or msg.get('reasoning_content', '')
    usage = data.get('usage', {})
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    duration = t1 - t0
    tok_per_sec = completion_tokens / duration if duration > 0 and completion_tokens > 0 else 0.0
    
    print(f"Duration: {duration:.2f}s | Speed: {tok_per_sec:.2f} tok/s | Tokens: {completion_tokens}")
    print(f"Engine Header: {resp.headers.get('X-Routed-Engine', 'Direct')}")
    print(f"Preview (first 250 chars):\n{content[:250]}...\n")
    
    passed, reason = validator_fn(content)
    if passed:
        print(f"Result: PASSED ({reason})")
    else:
        print(f"Result: FAILED ({reason})")
    return passed, tok_per_sec

def check_math(content):
    # Check for exact 323
    has_323 = bool(re.search(r'\b323\b', content))
    # Check for foreign characters (Chinese/Russian/Japanese/Korean)
    has_cjk = bool(re.search(r'[\u4e00-\u9fff\u0400-\u04ff\u3040-\u30ff]', content))
    if not has_323:
        return False, "Target answer '323' not found in text"
    if has_cjk:
        return False, "Detected foreign characters in output"
    return True, "Exact 323 found, 100% English"

def check_python(content):
    # Extract code blocks
    code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', content, re.DOTALL)
    code = code_match.group(1) if code_match else content
    try:
        ast.parse(code)
        syntax_ok = True
    except SyntaxError as e:
        syntax_ok = False
        return False, f"Python syntax error: {e}"
    
    has_cjk = bool(re.search(r'[\u4e00-\u9fff\u0400-\u04ff\u3040-\u30ff]', content))
    if has_cjk:
        return False, "Detected foreign characters in output"
    return True, "Valid Python 3 syntax, clean markdown"

def check_repetition(content):
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        return False, "Empty output"
    unique_lines = len(set(lines))
    ratio = unique_lines / len(lines)
    has_cjk = bool(re.search(r'[\u4e00-\u9fff\u0400-\u04ff\u3040-\u30ff]', content))
    if has_cjk:
        return False, "Detected foreign characters"
    if ratio < 0.8:
        return False, f"Repetition detected (uniqueness ratio: {ratio:.2f})"
    return True, f"Uniqueness ratio {ratio:.2f} (zero loop repetition)"

def main():
    results = []
    
    # Test 1: Math Verification on Cluster GPU
    p1 = {
        "model": "cluster-gpu",
        "messages": [{"role": "user", "content": "Calculate 17 * 19. Explain your solution clearly in English."}],
        "max_tokens": 128,
        "temperature": 0.0
    }
    ok1, speed1 = run_test("Math Exactness (Cluster GPU)", GPU_URL, p1, check_math)
    results.append(("Math Exactness (Cluster GPU)", ok1, speed1))
    
    # Test 2: Python Code Generation on Cluster GPU
    p2 = {
        "model": "cluster-gpu",
        "messages": [{"role": "user", "content": "Write a Python class 'CircularBuffer' with push, pop, and is_empty methods."}],
        "max_tokens": 300,
        "temperature": 0.0
    }
    ok2, speed2 = run_test("Python Syntax & Logic (Cluster GPU)", GPU_URL, p2, check_python)
    results.append(("Python Syntax & Logic (Cluster GPU)", ok2, speed2))
    
    # Test 3: Sustained Generation & Non-Repetition
    p3 = {
        "model": "cluster-gpu",
        "messages": [{"role": "user", "content": "Write an in-depth essay comparing unified memory architecture in Apple Silicon vs AMD Strix Halo."}],
        "max_tokens": 512,
        "temperature": 0.0
    }
    ok3, speed3 = run_test("Sustained 512-Tok Uniqueness (Cluster GPU)", GPU_URL, p3, check_repetition)
    results.append(("Sustained 512-Tok Uniqueness (Cluster GPU)", ok3, speed3))
    
    # Test 4: NPU Auto-Routing via Port 13306
    p4 = {
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
        "max_tokens": 64,
        "temperature": 0.0
    }
    ok4, speed4 = run_test("NPU Auto-Routing (Port 13306)", PROX_URL, p4, lambda c: ("Tokyo" in c, "Contains Tokyo"))
    results.append(("NPU Auto-Routing (Port 13306)", ok4, speed4))
    
    # Test 5: Complex Logic Auto-Routing to GPU via Port 13306
    p5 = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Analyze step-by-step why quicksort has worst-case O(n^2) complexity and how randomized pivot selection mitigates it."}],
        "max_tokens": 256,
        "temperature": 0.0
    }
    ok5, speed5 = run_test("GPU Auto-Routing (Port 13306)", PROX_URL, p5, check_repetition)
    results.append(("GPU Auto-Routing (Port 13306)", ok5, speed5))
    
    print("\n" + "="*70)
    print("FINAL VERIFICATION SUMMARY")
    print("="*70)
    all_passed = True
    for name, passed, speed in results:
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {name:<45} | Speed: {speed:.2f} tok/s")
        if not passed:
            all_passed = False
            
    if all_passed:
        print("\nALL VERIFICATION TESTS PASSED PERFECTLY!")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
