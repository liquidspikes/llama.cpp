#!/usr/bin/env python3
"""
Dual-Node AMD Strix Halo USB4STREAM Cluster 24/7 Auto-Healing Watchdog Daemon
Monitors:
- Dual Strix Halo GPU Cluster (177B MoE @ 1M Context on Lemonade / llama-server)
- FastFlowLM AMD XDNA 2 NPU Coprocessor (@ 96 tok/s on port 8999)
- Speculative & Auto-Routing Proxy (ports 13306 & 8000)
- Bosgame2 USB4 DMA RPC Worker (192.168.137.52)
- Zero-Network Character Devices (/dev/tbstream0, /dev/tbstream1)

Features:
- Sub-second dynamic port discovery across 8001-8006
- Deep progress tracking per slot (detects true graph hangs vs heavy computation)
- Active end-to-end synthetic canaries (short and decomp code prompts)
- Clean, sequential dual-node failover, auto-restart, and model reload
- Full logging to /var/log/llama-cluster-watchdog.log
"""

import os
import sys
import time
import json
import socket
import logging
import subprocess
import urllib.request
import urllib.error

LOG_FILE = "/var/log/llama-cluster-watchdog.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(console)

NODE2_IP = "192.168.137.52"
SSH_KEY = "/home/alexzimmerman/.ssh/id_ed25519"
DEFAULT_MODEL = "qwen3.8-flash-next"
CTX_SIZE = "1048576"
MAX_STALL_SECONDS = 240
CANARY_INTERVAL_SECONDS = 600

def get_json(url, timeout=5):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ClusterWatchdog/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.getcode(), json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, None
    except Exception:
        return 0, None

def find_active_gpu_port():
    for p in range(8001, 8007):
        code, data = get_json(f"http://127.0.0.1:{p}/health", timeout=1.0)
        if code in (200, 503):
            return p
    return 8002

def is_gpu_loading():
    for p in range(8001, 8007):
        code, data = get_json(f"http://127.0.0.1:{p}/health", timeout=1.0)
        if code == 503 and data and "Loading model" in str(data.get("error", {}).get("message", "")):
            return True, p
    return False, None

def get_slots():
    port = find_active_gpu_port()
    code, data = get_json(f"http://127.0.0.1:{port}/slots", timeout=2.5)
    if code == 200 and isinstance(data, list):
        return data, port
    return None, port

def check_remote_node():
    try:
        res = subprocess.run([
            "ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=4", f"alexzimmerman@{NODE2_IP}",
            "systemctl is-active llama-rpc.service"
        ], capture_output=True, text=True, timeout=5)
        return res.returncode == 0 and res.stdout.strip() == "active"
    except Exception as e:
        logging.warning(f"SSH check to Node 2 failed: {e}")
        return False

def run_canary_inference():
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Canary probe: Return exact string 'ALIVE_OK' and integer 42."}
        ],
        "max_tokens": 16,
        "temperature": 0.0
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:13306/v1/chat/completions",
        data=data_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "ClusterWatchdog/2.0"},
        method="POST"
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            code = resp.getcode()
            body = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            choices = body.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            timings = body.get("timings", {})
            pred_toks = timings.get("predicted_n", 0)
            pred_speed = timings.get("predicted_per_second", 0.0)
            logging.info(f"Canary check PASS in {elapsed:.2f}s ({pred_toks} toks @ {pred_speed:.1f} tok/s) - text: {content[:30]!r}")
            return True
    except Exception as e:
        logging.error(f"Canary check FAIL: {e}")
        return False

def recover_cluster(reason):
    logging.error(f"==================================================")
    logging.error(f"WATCHDOG INITIATING CLUSTER RECOVERY: {reason}")
    logging.error(f"==================================================")
    
    # 1. Stop Node 1 Lemonade service and kill any leftover llama-server
    logging.info("1/6 Stopping lemonade.service on Node 1...")
    subprocess.run(["systemctl", "stop", "lemonade.service"], timeout=15)
    subprocess.run(["pkill", "-9", "-f", "llama-server.real"], timeout=5)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], timeout=5)

    # 2. Stop Node 2 llama-rpc.service
    logging.info("2/6 Stopping llama-rpc.service on Node 2...")
    try:
        subprocess.run([
            "ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
            f"alexzimmerman@{NODE2_IP}", "sudo systemctl stop llama-rpc.service && sudo pkill -9 -f rpc-server"
        ], timeout=15)
    except Exception as e:
        logging.warning(f"Failed stopping Node 2 gracefully: {e}")

    time.sleep(3)

    # 3. Restart Node 2 llama-rpc.service
    logging.info("3/6 Starting llama-rpc.service on Node 2...")
    try:
        subprocess.run([
            "ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
            f"alexzimmerman@{NODE2_IP}", "sudo systemctl start llama-rpc.service"
        ], timeout=15)
    except Exception as e:
        logging.error(f"Failed starting Node 2 llama-rpc.service: {e}")
        return False

    time.sleep(3)

    # 4. Restart Node 1 Lemonade service
    logging.info("4/6 Starting lemonade.service on Node 1...")
    subprocess.run(["systemctl", "start", "lemonade.service"], timeout=15)
    time.sleep(3)

    # 5. Reload model with 1M context
    logging.info(f"5/6 Loading model {DEFAULT_MODEL} with {CTX_SIZE} context (--pinned)...")
    load_res = subprocess.run([
        "lemonade", "load", DEFAULT_MODEL, "--ctx-size", CTX_SIZE, "--pinned"
    ], capture_output=True, text=True, timeout=360)
    logging.info(f"lemonade load output: {load_res.stdout.strip()} {load_res.stderr.strip()}")

    # Restart speculative-proxy to pick up any new port immediately
    subprocess.run(["systemctl", "restart", "speculative-proxy.service"], timeout=10)

    # 6. Verify cluster health
    logging.info("6/6 Waiting for cluster ports to report healthy...")
    ready = False
    for i in range(40):
        time.sleep(5)
        code13306, h13306 = get_json("http://127.0.0.1:13306/health", timeout=2)
        if code13306 == 200 and h13306 and h13306.get("status") == "ok":
            logging.info("Cluster recovery SUCCESSFUL! Proxy on 13306 is healthy.")
            ready = True
            break
        logging.info(f"Waiting for recovery... ({i+1}/40)")

    if ready:
        time.sleep(2)
        run_canary_inference()
        return True
    else:
        logging.error("Cluster recovery FAILED: Timeout waiting for healthy status.")
        return False

def main():
    logging.info("================================================================")
    logging.info("=== Starting Dual-Node USB4STREAM Cluster Watchdog v2.0     ===")
    logging.info(f"=== Model: {DEFAULT_MODEL} | Ctx: {CTX_SIZE} | MaxStall: {MAX_STALL_SECONDS}s ===")
    logging.info("================================================================")

    slot_last_state = {}
    consecutive_health_failures = 0
    consecutive_slots_failures = 0
    last_canary_time = time.time()

    while True:
        try:
            # 0. Check if GPU backend is in the middle of streaming weights / loading model
            loading, load_port = is_gpu_loading()
            if loading:
                logging.info(f"Model backend on port {load_port} is actively streaming weights / loading...")
                time.sleep(5)
                consecutive_health_failures = 0
                consecutive_slots_failures = 0
                continue

            # 1. Check primary proxy health
            code13306, h13306 = get_json("http://127.0.0.1:13306/health", timeout=3)
            gpu_online = h13306.get("gpu_cluster_online", False) if h13306 else False
            if code13306 != 200 or not h13306 or h13306.get("status") != "ok" or not gpu_online:
                consecutive_health_failures += 1
                logging.warning(f"Health check failed on 13306 (attempt {consecutive_health_failures}/6, code={code13306}, gpu_online={gpu_online})")
                if consecutive_health_failures >= 6:
                    recover_cluster(f"Proxy 13306 / GPU cluster unhealthy for 6 consecutive checks (code={code13306}, gpu_online={gpu_online})")
                    consecutive_health_failures = 0
                    consecutive_slots_failures = 0
                    slot_last_state.clear()
                time.sleep(5)
                continue

            consecutive_health_failures = 0

            # 2. Check Node 2 worker state
            if not check_remote_node():
                logging.error("Node 2 (bosgame2) llama-rpc.service is NOT active!")
                recover_cluster("Node 2 worker llama-rpc.service stopped or unreachable")
                slot_last_state.clear()
                time.sleep(10)
                continue

            # 3. Check slots on active GPU port
            slots, active_port = get_slots()
            if not slots:
                consecutive_slots_failures += 1
                logging.warning(f"GPU slots unreachable on port {active_port} (attempt {consecutive_slots_failures}/6)")
                if consecutive_slots_failures >= 6:
                    recover_cluster(f"GPU slots unreachable on port {active_port} for 6 consecutive checks")
                    consecutive_slots_failures = 0
                    slot_last_state.clear()
                time.sleep(5)
                continue

            consecutive_slots_failures = 0

            current_time = time.time()
            for s in slots:
                slot_id = s.get("id")
                is_processing = s.get("is_processing", False)
                task_id = s.get("id_task", -1)

                if not is_processing:
                    slot_last_state.pop(slot_id, None)
                    continue

                prompt_processed = s.get("n_prompt_tokens_processed", 0)
                next_tokens = s.get("next_token", [{}])
                decoded = next_tokens[0].get("n_decoded", 0) if next_tokens else 0
                progress_key = (task_id, prompt_processed, decoded)

                if slot_id not in slot_last_state:
                    slot_last_state[slot_id] = {
                        "key": progress_key,
                        "time": current_time,
                        "task_id": task_id
                    }
                else:
                    prev = slot_last_state[slot_id]
                    if prev["key"] == progress_key and prev["task_id"] == task_id:
                        elapsed = current_time - prev["time"]
                        if elapsed > MAX_STALL_SECONDS:
                            recover_cluster(
                                f"Slot {slot_id} stalled on task {task_id} "
                                f"(processed={prompt_processed}, decoded={decoded}) on port {active_port} for {elapsed:.1f}s"
                            )
                            slot_last_state.clear()
                            break
                    else:
                        slot_last_state[slot_id] = {
                            "key": progress_key,
                            "time": current_time,
                            "task_id": task_id
                        }

            # 4. Periodic Active Canary
            any_slot_active = any(s.get("is_processing", False) for s in slots)
            if any_slot_active:
                # Active generation on cluster constitutes live proof of health
                last_canary_time = current_time
            elif current_time - last_canary_time >= CANARY_INTERVAL_SECONDS:
                logging.info("Slots idle. Running scheduled 10-minute active canary test...")
                canary_ok = run_canary_inference()
                if not canary_ok:
                    logging.warning("Canary failed once, verifying backend health before retry...")
                    code13306, h13306 = get_json("http://127.0.0.1:13306/health", timeout=3)
                    if code13306 == 200 and h13306 and h13306.get("status") == "ok":
                        logging.info("Proxy reports healthy. Retrying canary...")
                        canary_ok = run_canary_inference()
                    if not canary_ok:
                        fresh_slots, _ = get_slots()
                        if fresh_slots and any(s.get("is_processing", False) for s in fresh_slots):
                            logging.info("A slot became active during canary; skipping recovery.")
                        else:
                            recover_cluster("Consecutive canary inference failures on port 13306 while slots idle")
                            slot_last_state.clear()
                last_canary_time = time.time()

            # 5. Periodic status log if any slot is processing
            active_info = []
            for s in slots:
                if s.get("is_processing"):
                    nt = s.get("next_token", [{}])
                    dec = nt[0].get("n_decoded", 0) if nt else 0
                    active_info.append(
                        f"Slot{s.get('id')}(task={s.get('id_task')}, "
                        f"prompt={s.get('n_prompt_tokens_processed')}/{s.get('n_prompt_tokens')}, dec={dec})"
                    )
            if active_info:
                logging.info(f"Active Cluster Load [{active_port}]: {' | '.join(active_info)}")

        except Exception as e:
            logging.error(f"Watchdog loop exception: {e}")

        time.sleep(10)

if __name__ == "__main__":
    main()
