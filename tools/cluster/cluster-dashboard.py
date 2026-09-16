#!/usr/bin/env python3
"""
Dual-Node Strix Halo Cluster Dashboard & Real-Time "Stream of Consciousness" Monitor
- Web Dashboard on Port 9090 (with live 50ms SSE Synaptic Stream)
- Transparent Streaming API Proxy on Port 13305 -> Backend Lemonade on Port 13306
"""

import http.server
import socketserver
import json
import os
import subprocess
import glob
import time
import socket
import urllib.request
import urllib.error
import http.client
import fcntl
import struct
import ctypes
import re
import threading
import pwd
import select

DASHBOARD_PORT = 9090
PROXY_PORT = 13305
BACKEND_PORT = 13306

_prev_net_stats = {}
_prev_io_stats = {}
_prev_time = {}

LOCK = threading.Lock()
PROMPT_HISTORY = []
ACTIVE_PROMPT = None

_LLM_CACHE = None
_LLM_CACHE_TIME = 0.0
_LLM_CACHE_TTL = 1.5

def get_cpu_info():
    try:
        model = "Unknown AMD Processor"
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    model = line.split(":", 1)[1].strip()
                    break
        with open("/proc/stat") as f:
            fields = [float(x) for x in f.readline().strip().split()[1:5]]
            idle = fields[3]
            total = sum(fields)
            usage = 0.0
            if hasattr(get_cpu_info, "last_idle") and hasattr(get_cpu_info, "last_total"):
                d_idle = idle - get_cpu_info.last_idle
                d_total = total - get_cpu_info.last_total
                if d_total > 0:
                    usage = round((1.0 - d_idle / d_total) * 100.0, 1)
            get_cpu_info.last_idle = idle
            get_cpu_info.last_total = total
        
        temp = None
        for tf in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
            try:
                val = int(open(tf).read().strip()) / 1000.0
                if 20 <= val <= 110:
                    temp = val
                    break
            except:
                pass

        cores = os.cpu_count() or 16
        return {"model": model, "cores": cores, "usage_percent": usage, "temperature_c": temp}
    except Exception as e:
        return {"error": str(e)}

def get_mem_info():
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    mem[parts[0].strip()] = int(parts[1].strip().split()[0]) * 1024
        sys_total = mem.get("MemTotal", 0) / (1024**3)
        sys_avail = mem.get("MemAvailable", 0) / (1024**3)
        sys_used = sys_total - sys_avail
        
        vram_used = 0.0
        vram_total = 64.0
        for card in glob.glob("/sys/class/drm/card[0-9]/device"):
            vf = os.path.join(card, "mem_info_vram_used")
            vtf = os.path.join(card, "mem_info_vram_total")
            if os.path.exists(vf):
                vram_used = int(open(vf).read().strip()) / (1024**3)
            if os.path.exists(vtf):
                vram_total = int(open(vtf).read().strip()) / (1024**3)
            if os.path.exists(vf):
                break

        unified_total = 128.0
        unified_used = round(sys_used + vram_used, 2)
        unified_free = round(max(0.0, unified_total - unified_used), 2)
        unified_percent = round((unified_used / unified_total * 100.0), 1)

        return {
            "unified_total_gb": unified_total,
            "unified_used_gb": unified_used,
            "unified_free_gb": unified_free,
            "unified_percent": unified_percent,
            "system_total_gb": round(sys_total, 2),
            "system_used_gb": round(sys_used, 2),
            "system_available_gb": round(sys_avail, 2),
            "system_percent": round((sys_used / sys_total * 100.0) if sys_total else 0, 1),
            "vram_used_gb": round(vram_used, 2),
            "vram_total_gb": round(vram_total, 2),
            "total_gb": unified_total,
            "used_gb": unified_used,
            "available_gb": unified_free,
            "usage_percent": unified_percent
        }
    except Exception as e:
        return {"error": str(e)}

def get_disk_info():
    try:
        st = os.statvfs("/")
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        used = total - free
        return {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "usage_percent": round((used / total * 100.0) if total else 0, 1)
        }
    except Exception as e:
        return {"error": str(e)}

def get_gpu_info():
    try:
        for card in glob.glob("/sys/class/drm/card[0-9]/device"):
            busy_f = os.path.join(card, "gpu_busy_percent")
            vram_u_f = os.path.join(card, "mem_info_vram_used")
            vram_t_f = os.path.join(card, "mem_info_vram_total")
            gtt_u_f = os.path.join(card, "mem_info_gtt_used")
            gtt_t_f = os.path.join(card, "mem_info_gtt_total")
            
            if os.path.exists(busy_f):
                busy = int(open(busy_f).read().strip())
                vram_u = int(open(vram_u_f).read().strip()) / (1024**3)
                vram_t = int(open(vram_t_f).read().strip()) / (1024**3)
                gtt_u = int(open(gtt_u_f).read().strip()) / (1024**3) if os.path.exists(gtt_u_f) else 0.0
                gtt_t = int(open(gtt_t_f).read().strip()) / (1024**3) if os.path.exists(gtt_t_f) else 0.0
                
                sclk = None
                mclk = None
                sclk_f = os.path.join(card, "pp_dpm_sclk")
                mclk_f = os.path.join(card, "pp_dpm_mclk")
                if os.path.exists(sclk_f):
                    for l in open(sclk_f):
                        if "*" in l: sclk = l.split()[1]
                if os.path.exists(mclk_f):
                    for l in open(mclk_f):
                        if "*" in l: mclk = l.split()[1]

                temp = None
                power = None
                for hw in glob.glob(os.path.join(card, "hwmon/hwmon*")):
                    tf = os.path.join(hw, "temp1_input")
                    pf = os.path.join(hw, "power1_average")
                    if not os.path.exists(pf): pf = os.path.join(hw, "power1_input")
                    if os.path.exists(tf): temp = int(open(tf).read().strip()) / 1000.0
                    if os.path.exists(pf): power = round(int(open(pf).read().strip()) / 1000000.0, 1)

                return {
                    "model": "AMD Radeon 8060S Graphics (gfx1151)",
                    "busy_percent": busy,
                    "vram_used_gb": round(vram_u, 2),
                    "vram_total_gb": round(vram_t, 2),
                    "vram_percent": round((vram_u / vram_t * 100.0) if vram_t else 0, 1),
                    "gtt_used_gb": round(gtt_u, 2),
                    "gtt_total_gb": round(gtt_t, 2),
                    "temperature_c": temp,
                    "power_watts": power,
                    "sclk": sclk or "2900MHz",
                    "mclk": mclk or "1000MHz"
                }
        return {"error": "No AMD GPU device found"}
    except Exception as e:
        return {"error": str(e)}

def get_npu_info():
    try:
        accel_dev = "/dev/accel/accel0"
        sys_dev = "/sys/class/accel/accel0/device"
        if not os.path.exists(accel_dev):
            return {"available": False, "status": "Not Detected", "utilization_percent": 0.0, "busy_percent": 0.0}
        
        fw = open(os.path.join(sys_dev, "fw_version")).read().strip() if os.path.exists(os.path.join(sys_dev, "fw_version")) else "1.1.2.65"
        pwr = open(os.path.join(sys_dev, "power_state")).read().strip() if os.path.exists(os.path.join(sys_dev, "power_state")) else "D0"
        
        h_clk = 1800
        mp_clk = 1267
        tops_max = 58
        tops_curr = 58
        tasks_curr = 0
        tasks_max = 16
        cols = 8
        rows = 6
        tiles = 48
        
        try:
            IOC_NR = (3 << 30) | (16 << 16) | (ord('d') << 8) | (0x40 + 7)
            fd = os.open(accel_dev, os.O_RDWR)
            buf_clock = bytearray(64)
            buf_addr = ctypes.addressof((ctypes.c_char * 64).from_buffer(buf_clock))
            fcntl.ioctl(fd, IOC_NR, struct.pack('IIQ', 3, 64, buf_addr))
            mp_clk = struct.unpack('I', buf_clock[16:20])[0]
            h_clk = struct.unpack('I', buf_clock[40:44])[0]
            
            buf_res = bytearray(40)
            buf_addr = ctypes.addressof((ctypes.c_char * 40).from_buffer(buf_res))
            fcntl.ioctl(fd, IOC_NR, struct.pack('IIQ', 12, 40, buf_addr))
            clk_max, tops_max_val, task_max_val, tops_curr_val, task_curr_val = struct.unpack('QQQQQ', buf_res)
            tops_max = tops_max_val
            tops_curr = tops_curr_val
            tasks_curr = task_curr_val
            tasks_max = task_max_val
            
            buf_aie = bytearray(64)
            buf_addr = ctypes.addressof((ctypes.c_char * 64).from_buffer(buf_aie))
            fcntl.ioctl(fd, IOC_NR, struct.pack('IIQ', 1, 64, buf_addr))
            col_size, cols_val, rows_val = struct.unpack('IHH', buf_aie[:8])
            cols = cols_val
            rows = rows_val
            tiles = cols * rows
            
            os.close(fd)
        except Exception:
            pass

        # Inspect FLM process state for active inference compute load
        flm_busy = False
        try:
            for p_stat in glob.glob("/proc/[0-9]*/stat"):
                try:
                    with open(p_stat, "r") as sf:
                        content = sf.read()
                        if "flm-real" in content or "flm" in content:
                            p_state = content.split()[2]
                            if p_state == "R":
                                flm_busy = True
                                break
                except Exception:
                    pass
        except Exception:
            pass

        # Calculate hardware utilization percentage
        if tasks_curr > 0:
            task_ratio = tasks_curr / max(1, tasks_max)
            if flm_busy:
                util_pct = min(100.0, max(50.0, (task_ratio * 100.0) + 35.0))
            else:
                util_pct = min(100.0, task_ratio * 100.0)
        elif flm_busy:
            util_pct = 75.0
        else:
            util_pct = 0.0

        util_pct = round(util_pct, 1)
        status_label = "Inference Active" if flm_busy or util_pct > 35 else ("Allocated (D0)" if tasks_curr > 0 else "Ready (Standby D3hot)" if "D3" in pwr else "Ready (D0)")

        return {
            "available": True,
            "model": "AMD XDNA 2 Neural Processing Unit",
            "architecture": "XDNA 2 (Strix Halo)",
            "tops_max": tops_max,
            "tops_curr": tops_curr,
            "h_clock_mhz": h_clk,
            "mp_clock_mhz": mp_clk,
            "tasks_active": tasks_curr,
            "tasks_max": tasks_max,
            "utilization_percent": util_pct,
            "busy_percent": util_pct,
            "tiles": tiles,
            "cols": cols,
            "rows": rows,
            "firmware": fw,
            "device_node": "/dev/accel/accel0",
            "power_state": pwr,
            "status": status_label
        }
    except Exception as e:
        return {"available": False, "error": str(e), "utilization_percent": 0.0, "busy_percent": 0.0}

def get_wirespeeds():
    global _prev_net_stats, _prev_time
    now = time.time()
    dt = now - _prev_time.get("net", now - 1.0)
    if dt <= 0: dt = 1.0
    _prev_time["net"] = now

    interfaces = {}
    for iface in glob.glob("/sys/class/net/*"):
        name = os.path.basename(iface)
        if name == "lo": continue
        
        speed_f = os.path.join(iface, "speed")
        rx_f = os.path.join(iface, "statistics/rx_bytes")
        tx_f = os.path.join(iface, "statistics/tx_bytes")
        
        speed_mbps = 1000
        try:
            if os.path.exists(speed_f):
                val = int(open(speed_f).read().strip())
                if val > 0: speed_mbps = val
        except:
            if "thunderbolt" in name: speed_mbps = 20000

        rx_bytes = int(open(rx_f).read().strip()) if os.path.exists(rx_f) else 0
        tx_bytes = int(open(tx_f).read().strip()) if os.path.exists(tx_f) else 0

        prev = _prev_net_stats.get(name, {"rx": rx_bytes, "tx": tx_bytes})
        rx_rate_bytes = max(0, rx_bytes - prev["rx"]) / dt
        tx_rate_bytes = max(0, tx_bytes - prev["tx"]) / dt

        _prev_net_stats[name] = {"rx": rx_bytes, "tx": tx_bytes}

        rx_mbps = (rx_rate_bytes * 8.0) / 1_000_000.0
        tx_mbps = (tx_rate_bytes * 8.0) / 1_000_000.0
        total_mbps = rx_mbps + tx_mbps
        util_pct = round(min(100.0, (total_mbps / speed_mbps) * 100.0), 2)

        interfaces[name] = {
            "capacity_mbps": speed_mbps,
            "rx_mbps": round(rx_mbps, 2),
            "tx_mbps": round(tx_mbps, 2),
            "rx_mbs": round(rx_rate_bytes / (1024**2), 2),
            "tx_mbs": round(tx_rate_bytes / (1024**2), 2),
            "util_percent": util_pct
        }

    llm_pids = []
    for p in glob.glob('/proc/[0-9]*'):
        try:
            comm = open(f'{p}/comm').read().strip()
            if comm.startswith(('llama-server', 'rpc-server', 'ggml-rpc-server')):
                llm_pids.append(int(os.path.basename(p)))
        except Exception:
            pass
    
    total_io_r = 0
    total_io_w = 0
    for pid in llm_pids:
        io_f = f"/proc/{pid}/io"
        if os.path.exists(io_f):
            try:
                for line in open(io_f):
                    if line.startswith("rchar:"): total_io_r += int(line.split()[1])
                    if line.startswith("wchar:"): total_io_w += int(line.split()[1])
            except:
                pass

    prev_io = _prev_io_stats.get("llm", {"r": total_io_r, "w": total_io_w, "peak": 0.0})
    dma_r_bytes = max(0, total_io_r - prev_io["r"]) / dt
    dma_w_bytes = max(0, total_io_w - prev_io["w"]) / dt
    curr_rate_mbs = round((dma_r_bytes + dma_w_bytes) / (1024**2), 2)
    peak_mbs = max(prev_io.get("peak", 0.0), curr_rate_mbs)
    _prev_io_stats["llm"] = {"r": total_io_r, "w": total_io_w, "peak": peak_mbs}

    dma_mbps = ((dma_r_bytes + dma_w_bytes) * 8.0) / 1_000_000.0
    usb4_capacity = 80000
    usb4_util = round(min(100.0, (dma_mbps / usb4_capacity) * 100.0), 3)

    load_pct = 0
    if curr_rate_mbs > 50.0:
        try:
            import json
            with open("/tmp/dma_metrics.json") as m:
                dstats = json.load(m)
                tx_b = dstats.get("tx_bytes", 0)
                load_pct = min(100.0, (tx_b / (64.0 * 1024 * 1024 * 1024)) * 100.0)
        except:
            pass

    activity_mode = "Idle"
    if curr_rate_mbs > 50.0:
        activity_mode = "Prompt Prefill / Matrix Ingestion"
    elif curr_rate_mbs > 1.0:
        activity_mode = "Activation Stream (Decoding @ 22 t/s)"

    interfaces["usb4stream_dma"] = {
        "capacity_mbps": usb4_capacity,
        "capacity_str": "80 Gbps (Dual 40G USB4 PCIe DMA)",
        "rate_mbs": curr_rate_mbs,
        "peak_mbs": peak_mbs,
        "activity_mode": activity_mode,
        "load_progress": load_pct,
        "rate_mbps": round(dma_mbps, 2),
        "util_percent": usb4_util
    }

    return interfaces

_NET_IPS_CACHE = None
_NET_IPS_TIME = 0.0

def get_net_ips():
    global _NET_IPS_CACHE, _NET_IPS_TIME
    now = time.time()
    if _NET_IPS_CACHE is not None and (now - _NET_IPS_TIME) < 5.0:
        return dict(_NET_IPS_CACHE)
    ips = {}
    try:
        out = subprocess.check_output(["ip", "-o", "-4", "addr", "show"], text=True, timeout=1.0)
        for line in out.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 4:
                iface = parts[1]
                addr = parts[3].split("/")[0]
                if iface != "lo": ips[iface] = addr
        _NET_IPS_CACHE = ips
        _NET_IPS_TIME = now
    except Exception as e:
        ips["error"] = str(e)
    return ips

def get_os_info():
    os_name = "Linux"
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_name = line.split("=", 1)[1].strip().strip('"')
    except: pass
    
    kernel = os.uname().release
    uptime_str = "--"
    try:
        with open("/proc/uptime") as f:
            secs = float(f.readline().split()[0])
            mins, sec = divmod(int(secs), 60)
            hours, min_ = divmod(mins, 60)
            days, hr = divmod(hours, 24)
            uptime_str = f"{days}d {hr}h {min_}m {sec}s" if days > 0 else f"{hr}h {min_}m {sec}s"
    except: pass

    return {
        "hostname": socket.gethostname(),
        "os": os_name,
        "kernel": kernel,
        "uptime": uptime_str
    }

def get_usb4_status():
    s0 = os.path.exists("/dev/tbstream0")
    s1 = os.path.exists("/dev/tbstream1")
    return {
        "configured": s0 and s1,
        "devices": ["/dev/tbstream0", "/dev/tbstream1"] if (s0 and s1) else []
    }

def detect_llm():
    info = {
        "server_online": False,
        "role": "Offline",
        "model_name": "None Loaded",
        "model_path": "",
        "model_details": "No active model loaded",
        "split_mode": "none",
        "split_mode_label": "STANDBY",
        "ctx_size": 0,
        "devices": "",
        "status_tag": "OFFLINE",
        "port": 8001
    }

    # Scan /proc directly: zero subprocesses, immune to shell/python wrappers
    l_proc, r_proc, lem_proc = None, None, None
    for p in glob.glob('/proc/[0-9]*'):
        try:
            comm = open(f'{p}/comm').read().strip()
            if comm.startswith('llama-server') and not l_proc:
                cmd = open(f'{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='ignore').strip()
                l_proc = (os.path.basename(p), cmd)
            elif comm.startswith(('rpc-server', 'ggml-rpc-server')) and not r_proc:
                cmd = open(f'{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='ignore').strip()
                r_proc = (os.path.basename(p), cmd)
            elif comm.startswith('lemond') and not lem_proc:
                cmd = open(f'{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='ignore').strip()
                lem_proc = (os.path.basename(p), cmd)
        except Exception:
            pass

    if l_proc:
        pid, line = l_proc
        info["server_online"] = True
        info["role"] = "Primary Master (llama-server)"

        m = re.search(r'(?:-m|--model)\s+([^\s]+)', line)
        if m:
            info["model_path"] = m.group(1)
            raw_name = os.path.basename(m.group(1))
            if raw_name.endswith(".gguf"):
                raw_name = raw_name[:-5]
            info["model_name"] = raw_name

        c = re.search(r'(?:-c|--ctx-size)\s+(\d+)', line)
        if c:
            info["ctx_size"] = int(c.group(1))

        sm = re.search(r'(?:-sm|--split-mode)\s+([^\s]+)', line)
        if sm:
            info["split_mode"] = sm.group(1)

        dev = re.search(r'(?:--device|-dev)\s+([^\s]+)', line)
        if dev:
            info["devices"] = dev.group(1)

        ts = re.search(r'(?:-ts|--tensor-split)\s+([^\s]+)', line)
        ts_str = f" · TS {ts.group(1)}" if ts else ""

        port_match = re.search(r'--port\s+(\d+)', line)
        port = int(port_match.group(1)) if port_match else 8001
        info["port"] = port

        if info["split_mode"] == "rpc-tensor":
            info["split_mode_label"] = "HYBRID TP (80G DMA RING)"
            mode_desc = f"Hybrid Tensor Parallelism{ts_str}"
        elif info["split_mode"] == "layer":
            info["split_mode_label"] = "LAYER PIPELINE (DUAL DMA)"
            mode_desc = f"Pipeline Parallelism{ts_str}"
        elif info["split_mode"] == "row":
            info["split_mode_label"] = "ROW TENSOR PARALLEL"
            mode_desc = f"Row Parallelism{ts_str}"
        else:
            info["split_mode_label"] = "LOCAL APU"
            mode_desc = "Local Single-Node APU"

        ctx_str = f"{info['ctx_size'] // 1024}k Context" if info["ctx_size"] >= 1024 else f"{info['ctx_size']} Context"
        dev_str = f" · Dev: {info['devices']}" if info["devices"] else ""
        info["model_details"] = f"{ctx_str} · {mode_desc}{dev_str}"

        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={"User-Agent": "Dashboard"})
            with urllib.request.urlopen(req, timeout=0.3) as s_resp:
                if s_resp.status == 200:
                    info["status_tag"] = "READY"
                else:
                    info["status_tag"] = "LOADING"
        except urllib.error.HTTPError as he:
            if he.code == 503:
                info["status_tag"] = "LOADING (503)"
            else:
                info["status_tag"] = f"HTTP {he.code}"
        except Exception:
            info["status_tag"] = "INITIALIZING"

        return info

    if r_proc:
        pid, line = r_proc
        info["server_online"] = True
        info["role"] = "Compute Worker (rpc-server)"
        info["split_mode"] = "rpc-worker"
        info["split_mode_label"] = "RPC WORKER (DMA)"
        info["model_name"] = "RPC Worker Node"
        info["model_details"] = "Awaiting DMA activation / tensor chunks"
        info["status_tag"] = "LISTENING"
        return info

    if lem_proc:
        pid, line = lem_proc
        info["server_online"] = True
        info["role"] = "Orchestrator Standby (lemond)"
        info["model_name"] = "Lemonade Orchestrator"
        info["model_details"] = "Listening on 0.0.0.0:13306 · Model Dormant"
        info["split_mode_label"] = "LEMONADE STANDBY"
        info["status_tag"] = "STANDBY"
        return info

    return info

def get_llm_status():
    global ACTIVE_PROMPT, PROMPT_HISTORY, _LLM_CACHE, _LLM_CACHE_TIME
    now = time.time()
    if _LLM_CACHE is None or (now - _LLM_CACHE_TIME) > _LLM_CACHE_TTL:
        _LLM_CACHE = detect_llm()
        _LLM_CACHE_TIME = now

    status = dict(_LLM_CACHE)
    status["model"] = f"{status['model_name']} ({status['model_details']})" if status['model_name'] != "None Loaded" else "None Loaded"
    
    with LOCK:
        if ACTIVE_PROMPT:
            status["active_prompt"] = dict(ACTIVE_PROMPT)
        else:
            status["active_prompt"] = None
        status["recent_prompts"] = [dict(p) for p in PROMPT_HISTORY]

    return status

def get_local_processes():
    kfd_map = {}
    for p in glob.glob('/sys/class/kfd/kfd/proc/*'):
        pid_s = os.path.basename(p)
        if pid_s.isdigit():
            tot = sum(int(open(vf).read().strip()) for vf in glob.glob(f'{p}/vram_*'))
            if tot > 0:
                kfd_map[int(pid_s)] = round(tot / (1024**3), 2)

    procs = []
    for p in glob.glob('/proc/[0-9]*'):
        try:
            pid = int(os.path.basename(p))
            comm = open(f'{p}/comm').read().strip()
            ai_keys = ['llama', 'rpc-server', 'lemond', 'lemonade', 'grok', 'muse', 'quality_sweep']
            is_ai = any(k in comm.lower() for k in ai_keys)
            vram = kfd_map.get(pid, 0.0)
            if not is_ai and vram == 0:
                continue

            cmd = open(f'{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='ignore').strip()
            threads = 1
            rss_kb = 0
            user = 'unknown'
            try:
                for line in open(f'{p}/status'):
                    if line.startswith('Threads:'):
                        threads = int(line.split(':')[1].strip())
                    elif line.startswith('VmRSS:'):
                        rss_kb = int(line.split(':')[1].split()[0])
                    elif line.startswith('Uid:'):
                        uid = int(line.split(':')[1].split()[0])
                        try:
                            user = pwd.getpwuid(uid).pw_name
                        except Exception:
                            user = str(uid)
            except Exception:
                pass

            role = 'Worker'
            if 'llama-server' in comm:
                role = 'Master Inference'
            elif 'rpc-server' in comm:
                role = 'RPC Worker'
            elif 'lemonade' in comm or 'lemond' in comm:
                role = 'API Gateway'
            elif 'grok' in comm:
                role = 'Grok Agent'
            elif 'muse' in comm:
                role = 'Muse Agent'

            procs.append({
                'pid': pid,
                'user': user,
                'role': role,
                'comm': comm,
                'threads': threads,
                'rss_gb': round(rss_kb / (1024**2), 2),
                'vram_gb': vram,
                'cmd': cmd[:120]
            })
        except Exception:
            pass

    procs.sort(key=lambda x: (x['vram_gb'], x['rss_gb']), reverse=True)
    return procs[:15]

def get_all_local_stats():
    return {
        "system": get_os_info(),
        "network_ips": get_net_ips(),
        "wirespeeds": get_wirespeeds(),
        "cpu": get_cpu_info(),
        "memory": get_mem_info(),
        "disk": get_disk_info(),
        "gpu": get_gpu_info(),
        "npu": get_npu_info(),
        "usb4stream": get_usb4_status(),
        "llm": get_llm_status(),
        "processes": get_local_processes(),
        "timestamp": time.time()
    }

def find_llama_port():
    global _LLM_CACHE
    if _LLM_CACHE and _LLM_CACHE.get("port"):
        port = _LLM_CACHE["port"]
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return port
        except Exception:
            pass

    for p in glob.glob('/proc/[0-9]*'):
        try:
            comm = open(f'{p}/comm').read().strip()
            if comm.startswith('llama-server'):
                cmd = open(f'{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='ignore').strip()
                m = re.search(r'--port\s+(\d+)', cmd)
                if m:
                    p_port = int(m.group(1))
                    try:
                        with socket.create_connection(("127.0.0.1", p_port), timeout=0.2):
                            return p_port
                    except Exception:
                        pass
        except Exception:
            pass

    try:
        with socket.create_connection(("127.0.0.1", BACKEND_PORT), timeout=0.2):
            return BACKEND_PORT
    except Exception:
        pass

    return 8001

# ==================== STREAMING API PROXY HANDLER (PORT 13305) ====================

class StreamingAPIProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        target_port = BACKEND_PORT
        lp = find_llama_port()

        if self.path.startswith("/api/v1/health"):
            if lp and lp != BACKEND_PORT:
                model_name = (_LLM_CACHE.get("model_name") if _LLM_CACHE else None) or "qwen3.8-flash-next-official"
                ctx_size = (_LLM_CACHE.get("ctx_size") if _LLM_CACHE else 1048576) or 1048576
                data = json.dumps({
                    "status": "ok",
                    "model_loaded": model_name,
                    "all_models_loaded": [{
                        "model_name": model_name,
                        "status": "ready",
                        "device": "gpu",
                        "recipe": "llamacpp",
                        "ctx_size": ctx_size,
                        "checkpoint": "/home/alexzimmerman/models/qwen3.8-flash-next-official",
                        "pinned": True
                    }],
                    "version": "11.9.0",
                    "websocket_port": 9000
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
                return

        if self.path.startswith("/v1/models") or self.path.startswith("/models") or self.path == "/health":
            if lp and lp != BACKEND_PORT:
                target_port = lp
        try:
            conn = http.client.HTTPConnection("127.0.0.1", target_port, timeout=10)
            headers = {k: v for k, v in self.headers.items() if k.lower() != 'host'}
            conn.request("GET", self.path, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ['transfer-encoding', 'content-length']:
                    self.send_header(k, v)
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            conn.close()
        except Exception as e:
            try:
                self.send_response(502)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(f"Proxy Backend Error: {str(e)}".encode())
            except: pass

    def do_POST(self):
        global ACTIVE_PROMPT, PROMPT_HISTORY
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        messages = []
        raw_prompt = ""
        default_model = (_LLM_CACHE.get("model_name") if _LLM_CACHE else None) or "Cluster Model"
        model_name = default_model
        tool_names = []
        try:
            body_json = json.loads(body.decode('utf-8'))
            messages = body_json.get("messages", [])
            raw_prompt = body_json.get("prompt", "")
            model_name = body_json.get("model", model_name)
            tools = body_json.get("tools", [])
            if tools and isinstance(tools, list):
                for t in tools:
                    if isinstance(t, dict):
                        fn = t.get("function", {})
                        if fn.get("name"): tool_names.append(fn.get("name"))
        except:
            pass

        question_text = ""
        system_text = ""
        formatted_turns = []
        total_input_words = 0

        if raw_prompt:
            if isinstance(raw_prompt, list):
                question_text = "\n".join(str(p) for p in raw_prompt)
            else:
                question_text = str(raw_prompt)
            total_input_words = len(question_text.split())
            formatted_turns.append(f"PROMPT:\n{question_text}")

        for m in messages:
            role = m.get("role", "unknown").upper()
            content = m.get("content", "")
            if isinstance(content, list):
                content_str = "\n".join([c.get("text", "") for c in content if isinstance(c, dict)])
            else:
                content_str = str(content) if content is not None else ""

            # Check for tool call turns
            if m.get("tool_calls"):
                for tc in m.get("tool_calls"):
                    fn = tc.get("function", {})
                    content_str += f"\n[TOOL CALL: {fn.get('name')}({fn.get('arguments')})]"

            total_input_words += len(content_str.split())

            if role == "USER":
                question_text = content_str
                formatted_turns.append(f"👤 USER:\n{content_str}")
            elif role == "SYSTEM":
                system_text = content_str
                formatted_turns.append(f"⚙️ SYSTEM:\n{content_str}")
            elif role == "ASSISTANT":
                formatted_turns.append(f"🤖 ASSISTANT:\n{content_str}")
            elif role == "TOOL":
                tool_name = m.get("name", "tool")
                formatted_turns.append(f"🛠️ TOOL RESULT ({tool_name}):\n{content_str}")
            else:
                formatted_turns.append(f"[{role}]:\n{content_str}")

        if not question_text and messages:
            # If no user message was explicitly tagged, take the last message
            last_m = messages[-1]
            question_text = str(last_m.get("content", "(API Multi-Turn Request)"))

        full_context_str = "\n\n".join(formatted_turns)

        prompt_record = {
            "id": int(time.time() * 1000),
            "time": time.strftime("%H:%M:%S"),
            "model": model_name,
            "system_prompt": system_text,
            "question": question_text or "(API Completion Prompt)",
            "full_context": full_context_str,
            "tools_available": tool_names,
            "thinking": "",
            "answer": "",
            "tool_calls_invoked": [],
            "status": "Deliberating...",
            "phase": "inner_monologue",
            "ttft": None,
            "tps": None,
            "tokens_think": 0,
            "tokens_answer": 0,
            "tokens_in": total_input_words,
            "tokens_out": 0,
            "start_time": time.time(),
            "duration": 0
        }

        with LOCK:
            ACTIVE_PROMPT = prompt_record

        target_port = BACKEND_PORT
        target_path = self.path
        if "/chat/completions" in self.path or "/completions" in self.path:
            lp = find_llama_port()
            if lp and lp != BACKEND_PORT:
                target_port = lp
            if target_path.startswith("/api/v1/"):
                target_path = target_path.replace("/api/v1/", "/v1/")
            try:
                bj = json.loads(body)
                if not bj.get("model"):
                    try:
                        if target_port == BACKEND_PORT:
                            with urllib.request.urlopen(f"http://127.0.0.1:{target_port}/api/v1/health", timeout=1) as hr:
                                hd = json.loads(hr.read().decode())
                            lm = hd.get("model_loaded")
                            bj["model"] = lm if lm else "qwen3.8-flash-next-official"
                        else:
                            with urllib.request.urlopen(f"http://127.0.0.1:{target_port}/v1/models", timeout=1) as hr:
                                hd = json.loads(hr.read().decode())
                            lm = (hd.get("data") or [{}])[0].get("id")
                            bj["model"] = lm if lm else "qwen3.8-flash-next-official"
                    except Exception:
                        bj["model"] = "qwen3.8-flash-next-official"
                ctk = bj.get("chat_template_kwargs")
                if not isinstance(ctk, dict):
                    ctk = {}
                if thinking_enabled:
                    # Official Qwen Thinking Mode parameters:
                    # temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
                    bj["temperature"] = 1.0 if bj.get("temperature") in (0.7, None) else bj["temperature"]
                    bj.setdefault("top_p", 0.95)
                    bj.setdefault("top_k", 20)
                    bj.setdefault("min_p", 0.0)
                    bj.setdefault("presence_penalty", 0.0)
                    bj.setdefault("repeat_penalty", 1.0)
                    bj.setdefault("frequency_penalty", 0.0)
                    bj.setdefault("dry_multiplier", 0.0)
                    ctk.setdefault("enable_thinking", True)
                    ctk.setdefault("preserve_thinking", False)
                    ctk.setdefault("reasoning_effort", "xhigh")
                else:
                    # Official Qwen Instruct (non-thinking) mode parameters:
                    # temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=0.0 (0.0 prevents proper-noun distortion like 'Capcomprehensive', callers can still override), repetition_penalty=1.0
                    bj["temperature"] = 0.7 if bj.get("temperature") in (1.0, None) else bj["temperature"]
                    bj.setdefault("top_p", 0.80)
                    bj.setdefault("top_k", 20)
                    bj.setdefault("min_p", 0.0)
                    bj.setdefault("presence_penalty", 0.0)
                    bj.setdefault("repeat_penalty", 1.0)
                    bj.setdefault("frequency_penalty", 0.0)
                    bj.setdefault("dry_multiplier", 0.8)
                    bj.setdefault("dry_base", 1.75)
                    bj.setdefault("dry_allowed_length", 2)
                    ctk["enable_thinking"] = False
                    ctk["preserve_thinking"] = False

                bj["chat_template_kwargs"] = ctk
                bj.setdefault("cache_prompt", True)

                mt = bj.get("max_tokens", bj.get("n_predict", None))
                if mt is None or int(mt) <= 0:
                    bj["max_tokens"] = 8192
                    bj.pop("n_predict", None)
                elif int(mt) < 4096:
                    bj["max_tokens"] = 8192
                body = json.dumps(bj).encode()
            except Exception:
                pass

        try:
            conn = http.client.HTTPConnection("127.0.0.1", target_port, timeout=1800)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "content-length")}
            headers["Content-Length"] = str(len(body))
            conn.request("POST", target_path, body=body, headers=headers)
            resp = conn.getresponse()

            self.send_response(resp.status)
            is_sse = False
            for k, v in resp.getheaders():
                if k.lower() == 'content-type' and 'event-stream' in v:
                    is_sse = True
                if k.lower() not in ['transfer-encoding', 'content-length']:
                    self.send_header(k, v)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            first_token_time = None
            token_count = 0
            cur_tool_call = {"name": "", "arguments": ""}

            if is_sse:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    try:
                        self.wfile.write(line)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break

                    line_str = line.decode('utf-8', errors='ignore')
                    if line_str.startswith("data: "):
                        payload = line_str[6:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                r_delta = delta.get("reasoning_content", "")
                                c_delta = delta.get("content", "")
                                t_calls = delta.get("tool_calls", [])
                                
                                if first_token_time is None and (r_delta or c_delta or t_calls):
                                    first_token_time = time.time()
                                    prompt_record["ttft"] = f"{round(first_token_time - prompt_record['start_time'], 2)}s"

                                if r_delta:
                                    prompt_record["thinking"] += r_delta
                                    prompt_record["status"] = "Deep Reasoning Active"
                                    prompt_record["phase"] = "inner_monologue"
                                    prompt_record["tokens_think"] += 1
                                    token_count += 1

                                if c_delta:
                                    prompt_record["answer"] += c_delta
                                    prompt_record["status"] = "Generating Output"
                                    prompt_record["phase"] = "output_generation"
                                    prompt_record["tokens_answer"] += 1
                                    token_count += 1

                                if t_calls:
                                    prompt_record["status"] = "Invoking Tool..."
                                    prompt_record["phase"] = "output_generation"
                                    for tc in t_calls:
                                        fn = tc.get("function", {})
                                        fn_name = fn.get("name", "")
                                        fn_args = fn.get("arguments", "")
                                        if fn_name and not cur_tool_call["name"]:
                                            cur_tool_call["name"] = fn_name
                                            prompt_record["answer"] += f"\n\n🛠️ [TOOL CALL]: {fn_name}\n"
                                        if fn_args:
                                            cur_tool_call["arguments"] += fn_args
                                            prompt_record["answer"] += fn_args
                                        token_count += 1
                                        prompt_record["tokens_answer"] += 1
                                
                                prompt_record["tokens_out"] = token_count
                                elapsed = time.time() - (first_token_time or prompt_record["start_time"])
                                if elapsed > 0 and token_count > 0:
                                    prompt_record["tps"] = f"{round(token_count / elapsed, 1)} t/s"
                        except:
                            pass
            else:
                data = resp.read()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                try:
                    res_json = json.loads(data.decode('utf-8'))
                    choices = res_json.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        prompt_record["thinking"] = msg.get("reasoning_content", "") or ""
                        ans_text = msg.get("content", "") or ""
                        if msg.get("tool_calls"):
                            for tc in msg.get("tool_calls"):
                                fn = tc.get("function", {})
                                ans_text += f"\n\n🛠️ [TOOL CALL]: {fn.get('name')}\nArguments: {fn.get('arguments')}"
                        prompt_record["answer"] = ans_text
                    usage = res_json.get("usage", {})
                    prompt_record["tokens_in"] = usage.get("prompt_tokens", total_input_words)
                    prompt_record["tokens_out"] = usage.get("completion_tokens", 0)
                    timings = res_json.get("timings", {})
                    if timings:
                        prompt_record["tps"] = f"{round(timings.get('predicted_per_second', 0), 1)} t/s"
                        prompt_record["ttft"] = f"{round(timings.get('prompt_ms', 0) / 1000.0, 2)}s"
                except:
                    pass

            prompt_record["status"] = "Synthesis Completed"
            prompt_record["phase"] = "completed"
            prompt_record["duration"] = round(time.time() - prompt_record["start_time"], 2)
            conn.close()

        except Exception as e:
            prompt_record["status"] = f"Interrupted: {str(e)}"
            prompt_record["phase"] = "error"
            try:
                self.send_response(500)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(str(e).encode())
            except:
                pass
        finally:
            with LOCK:
                PROMPT_HISTORY.append(prompt_record)
                if len(PROMPT_HISTORY) > 50:
                    PROMPT_HISTORY.pop(0)
                if ACTIVE_PROMPT == prompt_record:
                    ACTIVE_PROMPT = None

    def do_DELETE(self):
        self._proxy_generic("DELETE")

    def do_PUT(self):
        self._proxy_generic("PUT")

    def _proxy_generic(self, method):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None
        target_port = BACKEND_PORT
        try:
            conn = http.client.HTTPConnection("127.0.0.1", target_port, timeout=60)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in ('host', 'content-length')}
            if body:
                headers["Content-Length"] = str(len(body))
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ['transfer-encoding', 'content-length']:
                    self.send_header(k, v)
            data = resp.read()
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            conn.close()
        except Exception as e:
            try:
                self.send_response(502)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(f"Proxy Backend Error: {str(e)}".encode())
            except: pass

    def log_message(self, format, *args):
        pass

# ==================== WEB DASHBOARD HTML & HANDLER (PORT 9090) ====================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dual-Node Strix Halo Cluster · Live Stream of Consciousness</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    if (window.tailwind) {
      tailwind.config = {
        darkMode: 'class',
        theme: {
          extend: {
            colors: {
              brand: { 500: '#ef4444', 600: '#dc2626' }
            }
          }
        }
      };
    }
  </script>
  <style>
    @keyframes pulse-fast { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
    @keyframes synaptic-glow { 
      0%, 100% { border-color: rgba(168, 85, 247, 0.4); box-shadow: 0 0 15px rgba(168, 85, 247, 0.15); } 
      50% { border-color: rgba(236, 72, 153, 0.8); box-shadow: 0 0 25px rgba(236, 72, 153, 0.3); } 
    }
    @keyframes cursor-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
    
    .pulse-online { animation: pulse-fast 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
    .pulse-active { animation: pulse-fast 0.6s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
    .synaptic-active { animation: synaptic-glow 1.5s ease-in-out infinite; }
    .live-cursor { display: inline-block; width: 8px; height: 16px; background-color: #a855f7; animation: cursor-blink 0.8s infinite; vertical-align: middle; margin-left: 2px; }
    .live-cursor-emerald { display: inline-block; width: 8px; height: 16px; background-color: #10b981; animation: cursor-blink 0.8s infinite; vertical-align: middle; margin-left: 2px; }
    
    pre, code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    .custom-scroll::-webkit-scrollbar { width: 6px; }
    .custom-scroll::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.6); }
    .custom-scroll::-webkit-scrollbar-thumb { background: rgba(51, 65, 85, 0.8); border-radius: 4px; }

    /* Offline / Local Fallback Styling */
    body { background-color: #020617; color: #f8fafc; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
    .hidden { display: none !important; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-4 md:p-8 font-sans antialiased selection:bg-purple-500 selection:text-white">
  <div class="max-w-7xl mx-auto space-y-6">
    
    <!-- Header -->
    <header class="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-slate-800 gap-4">
      <div>
        <div class="flex items-center gap-3">
          <span class="inline-block w-3.5 h-3.5 rounded-full bg-emerald-500 pulse-online"></span>
          <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-pink-400 to-amber-300">
            Live Stream of Consciousness
          </h1>
          <span class="px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-purple-500/20 text-purple-300 border border-purple-500/30">NEURAL COGNITION 20FPS</span>
        </div>
        <p class="text-sm text-slate-400 mt-1">Dual APU Architecture · 256GB Unified Memory · Dual 58 TOPS XDNA 2 NPU · Real-Time Inner Monologue & Synthesized Output</p>
      </div>
      <div class="flex items-center gap-4 text-xs font-mono text-slate-400 bg-slate-900/90 px-4 py-2.5 rounded-xl border border-slate-800">
        <div>STREAM RATE: <span class="text-purple-400 font-bold">50ms (SSE)</span></div>
        <div>UPDATED: <span id="last-updated" class="text-slate-200">--:--:--</span></div>
      </div>
    </header>

    <!-- Cluster-Wide Summary Banner -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div class="bg-slate-900/90 rounded-xl p-4 border border-slate-800 shadow-lg">
        <div class="text-xs text-slate-400 font-mono flex items-center justify-between">
          <span>CLUSTER UNIFIED MEMORY</span>
          <span class="text-purple-400 font-bold">256 GB POOL</span>
        </div>
        <div class="mt-2 flex items-baseline gap-2">
          <span id="cluster-mem-val" class="text-2xl font-bold text-white font-mono">-- / 256 GB</span>
          <span id="cluster-mem-pct" class="text-xs font-bold text-purple-400 font-mono">--%</span>
        </div>
        <div class="w-full bg-slate-800 rounded-full h-2 mt-2.5 overflow-hidden">
          <div id="cluster-mem-bar" class="bg-gradient-to-r from-purple-500 to-indigo-500 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
        </div>
      </div>

      <div class="bg-slate-900/90 rounded-xl p-4 border border-slate-800 shadow-lg">
        <div class="text-xs text-slate-400 font-mono flex items-center justify-between">
          <span>CLUSTER NPU ACCELERATION</span>
          <span id="cluster-npu-pct" class="text-emerald-400 font-bold font-mono">0.0%</span>
        </div>
        <div class="mt-2 flex items-baseline justify-between gap-2">
          <span id="cluster-npu-val" class="text-xl font-bold text-white font-mono">116 TOPS (96 Tiles)</span>
          <span id="cluster-npu-tasks" class="text-xs text-emerald-400 font-mono font-semibold">Ready</span>
        </div>
        <div class="w-full bg-slate-800 rounded-full h-2 mt-2 overflow-hidden">
          <div id="cluster-npu-bar" class="bg-gradient-to-r from-emerald-500 to-teal-400 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
        </div>
        <div id="cluster-npu-sub" class="text-xs text-slate-400 font-mono mt-1 flex justify-between">
          <span>Dual XDNA 2 Engines</span>
          <span class="text-emerald-400">1800 MHz Pinned</span>
        </div>
      </div>

      <div class="bg-slate-900/90 rounded-xl p-4 border border-slate-800 shadow-lg">
        <div class="text-xs text-slate-400 font-mono flex items-center justify-between">
          <span>INTERCONNECT LINK</span>
          <span id="cluster-split-badge" class="text-cyan-400 font-bold">80 Gbps</span>
        </div>
        <div class="mt-2">
          <div id="cluster-split-label" class="text-base font-bold text-white font-mono truncate">Dual USB4 DMA</div>
          <div class="text-xs text-slate-400 font-mono mt-0.5">Zero-Loss NVMe Tensor Stream</div>
        </div>
      </div>

      <div class="bg-slate-900/90 rounded-xl p-4 border border-slate-800 shadow-lg">
        <div class="text-xs text-slate-400 font-mono flex items-center justify-between">
          <span>COGNITIVE ENGINE</span>
          <span id="cluster-model-badge" class="text-amber-400 font-bold">DETECTING...</span>
        </div>
        <div class="mt-2">
          <div id="cluster-model-name" class="text-base font-bold text-amber-300 truncate">Detecting Model...</div>
          <div id="cluster-model-details" class="text-xs text-slate-400 font-mono mt-0.5 truncate">Detecting topology &amp; context...</div>
        </div>
      </div>
    </div>

    <!-- ==================== STREAM OF CONSCIOUSNESS MAIN TERMINAL ==================== -->
    <div id="consciousness-card" class="bg-slate-900/95 rounded-2xl p-6 border border-slate-800 shadow-2xl space-y-6 transition-all duration-500">
      
      <!-- Top Consciousness Status Header -->
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div class="flex items-center gap-3.5">
          <div id="synapse-icon-box" class="p-3 rounded-2xl bg-gradient-to-br from-purple-600/20 via-pink-500/20 to-amber-400/20 text-purple-300 border border-purple-500/30">
            <svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
            </svg>
          </div>
          <div>
            <div class="flex items-center gap-2.5">
              <h2 class="text-xl font-black text-white tracking-wide">Neural Cognitive Stream</h2>
              <span id="synapse-pulse-dot" class="w-2.5 h-2.5 rounded-full bg-slate-600"></span>
            </div>
            <p class="text-xs text-slate-400 font-mono mt-0.5">Capturing internal deliberation tokens and conscious response emissions</p>
          </div>
        </div>
        
        <!-- Brain Activity Badge -->
        <div id="brain-state-badge" class="px-4 py-2 text-xs font-mono font-bold rounded-xl bg-slate-800 text-slate-400 border border-slate-700 flex items-center gap-2.5 shadow-inner">
          <span class="w-2 h-2 rounded-full bg-slate-500"></span> COGNITIVE ENGINE DORMANT (Listening on 13305)
        </div>
      </div>

      <!-- Consciousness Visual Grid (Prompt Input, Inner Monologue, Conscious Output) -->
      <div class="space-y-5">

        <!-- Interactive Inference Test Stimulus Dispatcher -->
        <div class="bg-slate-950/80 rounded-xl border border-purple-900/40 p-4 space-y-3 shadow-inner">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2 text-xs font-mono font-bold text-purple-300">
              <span class="p-1 rounded bg-purple-500/20 text-purple-300">⚡</span>
              <span>DISPATCH TEST PROMPT / INFERENCE STIMULUS</span>
            </div>
            <div class="flex items-center gap-2">
              <span id="stimulus-status" class="text-xs font-mono text-slate-400">Target: Port 13305 (Proxy) / 13306 (Engine)</span>
            </div>
          </div>
          <div class="flex flex-col sm:flex-row gap-2">
            <select id="mode-select" class="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-purple-500 shrink-0">
              <option value="thinking" selected>🧠 Thinking (xhigh, temp=1.0)</option>
              <option value="instruct">⚡ Instruct (direct, temp=0.7)</option>
            </select>
            <input id="prompt-input" type="text" placeholder="Enter test prompt (e.g. 'What is the capital of France?')..." class="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3.5 py-2 text-sm font-mono text-slate-100 placeholder-slate-500 focus:outline-none focus:border-purple-500 transition-colors" />
            <button id="send-prompt-btn" onclick="sendTestPrompt()" class="px-5 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white text-xs font-mono font-bold rounded-lg transition-all shadow-md active:scale-95 flex items-center justify-center gap-1.5 shrink-0">
              <span>⚡ Send Stimulus</span>
            </button>
          </div>
          <div class="flex flex-wrap items-center gap-2 text-[11px] font-mono text-slate-400">
            <span class="text-slate-500">Presets:</span>
            <button onclick="setPrompt('What is the capital of France?')" class="px-2 py-0.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 hover:border-slate-700 transition-colors">Capital of France</button>
            <button onclick="setPrompt('Explain how 80Gbps USB4 DMA enables zero-loss tensor parallelism across two AMD Strix Halo APUs.')" class="px-2 py-0.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 hover:border-slate-700 transition-colors">80G USB4 DMA</button>
            <button onclick="setPrompt('Write a fast Python script to benchmark memory throughput.')" class="px-2 py-0.5 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800 hover:border-slate-700 transition-colors">Python Benchmark</button>
          </div>
        </div>
        
        <!-- 1. Incoming Stimulus (Question / Task Prompt / Multi-Turn Context) -->
        <div class="bg-slate-950/80 rounded-xl border border-slate-800 p-4 space-y-2">
          <div class="flex items-center justify-between text-xs font-mono">
            <span class="text-amber-400 font-bold flex items-center gap-2">
              <span class="p-1 rounded bg-amber-500/20 text-amber-300">📥</span> INCOMING PROMPT / STIMULUS
            </span>
            <div class="flex items-center gap-3">
              <span id="live-tools-tag" class="hidden px-2 py-0.5 rounded text-[10px] bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">TOOLS ATTACHED</span>
              <span id="live-model-tag" class="text-slate-500 font-mono">Standby</span>
            </div>
          </div>
          <div id="live-question" class="bg-slate-900/90 p-3.5 rounded-lg border border-slate-800 text-sm font-mono text-slate-200 whitespace-pre-wrap max-h-48 overflow-y-auto custom-scroll leading-relaxed">
(No active request. Waiting for API prompt to trigger neural cognition...)
          </div>
          <div id="live-context-drawer" class="hidden pt-2">
            <div class="text-[11px] font-bold text-amber-300 font-mono mb-1">📜 FULL MULTI-TURN CONTEXT &amp; SYSTEM INSTRUCTIONS:</div>
            <div id="live-full-context" class="bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs font-mono text-slate-400 whitespace-pre-wrap max-h-60 overflow-y-auto custom-scroll"></div>
          </div>
          <button id="toggle-context-btn" onclick="toggleFullContext()" class="hidden text-xs font-mono text-amber-400 hover:text-amber-300 underline pt-1">
            ▼ View Full System Prompt &amp; Conversation Context (<span id="context-words-count">0</span> words)
          </button>
        </div>

        <!-- 2. The Inner Monologue (<think> Reasoning Stream) -->
        <div id="think-container" class="bg-slate-950/90 rounded-xl border border-purple-900/40 p-4 space-y-2 transition-all">
          <div class="flex items-center justify-between text-xs font-mono">
            <span class="text-purple-400 font-bold flex items-center gap-2">
              <span class="p-1 rounded bg-purple-500/20 text-purple-300">🧠</span> INNER MONOLOGUE &amp; CHAIN-OF-THOUGHT (&lt;think&gt;)
            </span>
            <div class="flex items-center gap-3 text-[11px]">
              <span id="live-think-count" class="text-purple-300/80">0 thought tokens</span>
              <span id="live-ttft-tag" class="text-pink-400 font-bold">TTFT: --</span>
            </div>
          </div>
          <div id="live-thinking" class="bg-gradient-to-b from-purple-950/20 to-slate-950 p-4 rounded-lg border border-purple-900/30 text-xs font-mono text-purple-200/95 whitespace-pre-wrap max-h-72 overflow-y-auto custom-scroll leading-relaxed tracking-wide">
(Inner monologue stream will materialize here in real time...)
          </div>
        </div>

        <!-- 3. The Conscious Output (Answer / Decompiled Code) -->
        <div id="output-container" class="bg-slate-950/90 rounded-xl border border-emerald-900/40 p-4 space-y-2 transition-all">
          <div class="flex items-center justify-between text-xs font-mono">
            <span class="text-emerald-400 font-bold flex items-center gap-2">
              <span class="p-1 rounded bg-emerald-500/20 text-emerald-300">⚡</span> CONSCIOUS SYNTHESIS &amp; OUTPUT EMISSION
            </span>
            <div class="flex items-center gap-3 text-[11px]">
              <span id="live-ans-count" class="text-emerald-300/80">0 output tokens</span>
              <span id="live-tps-tag" class="text-emerald-400 font-bold font-mono">-- t/s</span>
            </div>
          </div>
          <div id="live-answer" class="bg-gradient-to-b from-emerald-950/20 to-slate-950 p-4 rounded-lg border border-emerald-900/30 text-sm font-mono text-emerald-200 whitespace-pre-wrap max-h-96 overflow-y-auto custom-scroll leading-relaxed">
(Conscious response will stream here...)
          </div>
        </div>

        <!-- History container for cognitive sessions & prompt archive -->
        <div class="pt-2">
          <div class="text-xs font-bold text-slate-400 font-mono mb-2 flex items-center justify-between">
            <span>📜 RECENT COGNITIVE TRACES &amp; PROMPTS</span>
            <span class="text-[11px] text-slate-500">Live Memory</span>
          </div>
          <div id="history-container" class="space-y-3"></div>
        </div>

      </div>
    </div>

    <!-- ==================== HARDWARE TELEMETRY ==================== -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      
      <!-- Node 1 (Local Master) -->
      <div id="node1-card" class="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 shadow-xl space-y-5">
        <div class="flex items-center justify-between pb-4 border-b border-slate-800">
          <div class="flex items-center gap-3">
            <span class="px-2.5 py-1 text-xs font-bold rounded-lg bg-red-500/20 text-red-400 border border-red-500/30">NODE 1</span>
            <h2 id="node1-host" class="text-xl font-bold text-white">bosgame1</h2>
          </div>
          <div id="node1-role" class="text-xs px-2.5 py-1 rounded-md bg-emerald-500/20 text-emerald-300 font-mono font-bold">PRIMARY MASTER</div>
        </div>

        <div id="node1-error-banner" class="hidden text-xs bg-red-950/70 border border-red-500/50 text-red-300 p-3 rounded-xl font-mono"></div>

        <div class="grid grid-cols-2 gap-3 text-xs bg-slate-950/70 p-3.5 rounded-xl border border-slate-800/80 font-mono">
          <div><span class="text-slate-500">OS:</span> <span id="node1-os" class="font-medium text-slate-200">--</span></div>
          <div><span class="text-slate-500">Kernel:</span> <span id="node1-kernel" class="text-slate-300">--</span></div>
          <div><span class="text-slate-500">Ethernet IP:</span> <span id="node1-ip-eth" class="text-amber-300 font-bold">--</span></div>
          <div><span class="text-slate-500">Wi-Fi IP:</span> <span id="node1-ip-wifi" class="text-emerald-300 font-bold">--</span></div>
          <div><span class="text-slate-500">Uptime:</span> <span id="node1-uptime" class="text-slate-300">--</span></div>
        </div>

        <div class="bg-gradient-to-br from-purple-950/40 via-slate-950 to-slate-900 p-4 rounded-xl border border-purple-900/40 space-y-3">
          <div class="flex justify-between items-center">
            <div class="text-sm font-semibold text-purple-300">Overall Unified Memory (128 GB LPDDR5X)</div>
            <span id="node1-unified-pct" class="text-purple-300 font-bold text-xs font-mono">--%</span>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">Total Unified RAM Used</span>
              <span id="node1-unified-text" class="text-purple-300 font-bold">-- / 128.0 GB</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
              <div id="node1-unified-bar" class="bg-gradient-to-r from-purple-500 via-indigo-500 to-cyan-500 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
          <div class="grid grid-cols-3 gap-2 pt-1 text-[11px] font-mono text-center">
            <div class="bg-slate-900/90 py-1.5 px-2 rounded-lg border border-slate-800">
              <span class="text-slate-400 block text-[10px]">GPU VRAM</span>
              <span id="node1-sub-vram" class="text-cyan-400 font-bold">-- GB</span>
            </div>
            <div class="bg-slate-900/90 py-1.5 px-2 rounded-lg border border-slate-800">
              <span class="text-slate-400 block text-[10px]">HOST RAM</span>
              <span id="node1-sub-sys" class="text-purple-400 font-bold">-- GB</span>
            </div>
            <div class="bg-slate-900/90 py-1.5 px-2 rounded-lg border border-slate-800">
              <span class="text-slate-400 block text-[10px]">AVAILABLE</span>
              <span id="node1-sub-free" class="text-emerald-400 font-bold">-- GB</span>
            </div>
          </div>
        </div>

        <!-- Node 1 CPU (Zen 5) -->
        <div class="bg-gradient-to-br from-slate-950 to-slate-900 p-4 rounded-xl border border-slate-800 space-y-3">
          <div class="flex justify-between items-center">
            <div class="text-sm font-semibold text-slate-200">AMD Ryzen AI Max+ 395 CPU (Zen 5)</div>
            <div class="text-xs font-mono text-slate-400">
              <span id="node1-cpu-cores" class="text-blue-400">32 Threads</span> | 
              <span id="node1-cpu-temp" class="text-amber-400">--°C</span>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">CPU Usage</span>
              <span id="node1-cpu-usage" class="text-blue-400 font-bold">0%</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
              <div id="node1-cpu-usage-bar" class="bg-blue-500 h-2.5 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
        </div>

        <div class="bg-gradient-to-br from-slate-950 to-slate-900 p-4 rounded-xl border border-slate-800 space-y-3">
          <div class="flex justify-between items-center">
            <div class="text-sm font-semibold text-slate-200">AMD Radeon 8060S GPU (40 CUs)</div>
            <div class="text-xs font-mono text-slate-400">
              <span id="node1-gpu-clock" class="text-amber-400">2900MHz</span> | 
              <span id="node1-gpu-temp" class="text-red-400">--°C</span> | 
              <span id="node1-gpu-power" class="text-cyan-400">--W</span>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">Compute Load</span>
              <span id="node1-gpu-busy" class="text-emerald-400 font-bold">0%</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
              <div id="node1-gpu-busy-bar" class="bg-emerald-500 h-2.5 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">Unified VRAM (GTT Allocation)</span>
              <span id="node1-gpu-vram" class="text-cyan-400 font-bold">-- / 120 GB</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              <div id="node1-gpu-vram-bar" class="bg-cyan-500 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
        </div>

        <div class="bg-slate-950/80 p-4 rounded-xl border border-slate-800 space-y-3">
          <div class="text-xs font-bold text-slate-300 flex items-center justify-between">
            <span>🧠 AMD XDNA 2 NPU Coprocessor</span>
            <div class="text-[11px] font-mono">
              <span id="node1-npu-clock" class="text-amber-400">1800MHz</span> | 
              <span id="node1-npu-tops" class="text-emerald-400">58 TOPS</span>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">Hardware Compute Load (<span id="node1-npu-tasks" class="text-slate-300">4/16 Tasks</span>)</span>
              <span id="node1-npu-busy" class="text-emerald-400 font-bold">0%</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
              <div id="node1-npu-busy-bar" class="bg-gradient-to-r from-emerald-500 to-teal-400 h-2.5 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
          <div class="flex justify-between text-xs font-mono text-slate-400">
            <span>Model: <span id="node1-npu-model" class="text-slate-200">qwen3:0.6b (FastFlowLM)</span></span>
            <span id="node1-npu-status" class="text-emerald-400">Active (D0)</span>
          </div>
        </div>

        <div class="bg-slate-950/80 p-4 rounded-xl border border-slate-800 space-y-3">
          <div class="text-xs font-bold text-slate-300 flex items-center justify-between">
            <span>⚡ Wirespeeds & Bandwidth</span>
            <span class="text-emerald-400 font-mono text-[11px]">USB4STREAM Active</span>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">USB4 DMA Tunnel (Dual 40G - 80 Gbps)</span>
              <span id="node1-usb4-speed" class="text-cyan-300 font-bold">0.0 MB/s</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              <div id="node1-usb4-bar" class="bg-cyan-500 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Node 2 (Remote Worker) -->
      <div id="node2-card" class="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 shadow-xl space-y-5">
        <div class="flex items-center justify-between pb-4 border-b border-slate-800">
          <div class="flex items-center gap-3">
            <span class="px-2.5 py-1 text-xs font-bold rounded-lg bg-blue-500/20 text-blue-400 border border-blue-500/30">NODE 2</span>
            <h2 id="node2-host" class="text-xl font-bold text-white">bosgame2</h2>
          </div>
          <div id="node2-role" class="text-xs px-2.5 py-1 rounded-md bg-blue-500/20 text-blue-300 font-mono font-bold">COMPUTE WORKER</div>
        </div>

        <div id="node2-error-banner" class="hidden text-xs bg-red-950/70 border border-red-500/50 text-red-300 p-3 rounded-xl font-mono"></div>

        <div class="grid grid-cols-2 gap-3 text-xs bg-slate-950/70 p-3.5 rounded-xl border border-slate-800/80 font-mono">
          <div><span class="text-slate-500">OS:</span> <span id="node2-os" class="font-medium text-slate-200">--</span></div>
          <div><span class="text-slate-500">Kernel:</span> <span id="node2-kernel" class="text-slate-300">--</span></div>
          <div><span class="text-slate-500">Ethernet IP:</span> <span id="node2-ip-eth" class="text-amber-300 font-bold">192.168.137.52</span></div>
          <div><span class="text-slate-500">Wi-Fi IP:</span> <span id="node2-ip-wifi" class="text-emerald-300 font-bold">--</span></div>
          <div><span class="text-slate-500">Uptime:</span> <span id="node2-uptime" class="text-slate-300">--</span></div>
        </div>

        <div class="bg-gradient-to-br from-purple-950/40 via-slate-950 to-slate-900 p-4 rounded-xl border border-purple-900/40 space-y-3">
          <div class="flex justify-between items-center">
            <div class="text-sm font-semibold text-purple-300">Overall Unified Memory (128 GB LPDDR5X)</div>
            <span id="node2-unified-pct" class="text-purple-300 font-bold text-xs font-mono">--%</span>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">Total Unified RAM Used</span>
              <span id="node2-unified-text" class="text-purple-300 font-bold">-- / 128.0 GB</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
              <div id="node2-unified-bar" class="bg-gradient-to-r from-purple-500 via-indigo-500 to-cyan-500 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
          <div class="grid grid-cols-3 gap-2 pt-1 text-[11px] font-mono text-center">
            <div class="bg-slate-900/90 py-1.5 px-2 rounded-lg border border-slate-800">
              <span class="text-slate-400 block text-[10px]">GPU VRAM</span>
              <span id="node2-sub-vram" class="text-cyan-400 font-bold">-- GB</span>
            </div>
            <div class="bg-slate-900/90 py-1.5 px-2 rounded-lg border border-slate-800">
              <span class="text-slate-400 block text-[10px]">HOST RAM</span>
              <span id="node2-sub-sys" class="text-purple-400 font-bold">-- GB</span>
            </div>
            <div class="bg-slate-900/90 py-1.5 px-2 rounded-lg border border-slate-800">
              <span class="text-slate-400 block text-[10px]">AVAILABLE</span>
              <span id="node2-sub-free" class="text-emerald-400 font-bold">-- GB</span>
            </div>
          </div>
        </div>

        <!-- Node 2 CPU (Zen 5) -->
        <div class="bg-gradient-to-br from-slate-950 to-slate-900 p-4 rounded-xl border border-slate-800 space-y-3">
          <div class="flex justify-between items-center">
            <div class="text-sm font-semibold text-slate-200">AMD Ryzen AI Max+ 395 CPU (Zen 5)</div>
            <div class="text-xs font-mono text-slate-400">
              <span id="node2-cpu-cores" class="text-blue-400">32 Threads</span> | 
              <span id="node2-cpu-temp" class="text-amber-400">--°C</span>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">CPU Usage</span>
              <span id="node2-cpu-usage" class="text-blue-400 font-bold">0%</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
              <div id="node2-cpu-usage-bar" class="bg-blue-500 h-2.5 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
        </div>

        <div class="bg-gradient-to-br from-slate-950 to-slate-900 p-4 rounded-xl border border-slate-800 space-y-3">
          <div class="flex justify-between items-center">
            <div class="text-sm font-semibold text-slate-200">AMD Radeon 8060S GPU (40 CUs)</div>
            <div class="text-xs font-mono text-slate-400">
              <span id="node2-gpu-clock" class="text-amber-400">2900MHz</span> | 
              <span id="node2-gpu-temp" class="text-red-400">--°C</span> | 
              <span id="node2-gpu-power" class="text-cyan-400">--W</span>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">Compute Load</span>
              <span id="node2-gpu-busy" class="text-emerald-400 font-bold">0%</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
              <div id="node2-gpu-busy-bar" class="bg-emerald-500 h-2.5 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">Unified VRAM (GTT Allocation)</span>
              <span id="node2-gpu-vram" class="text-cyan-400 font-bold">-- / 120 GB</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              <div id="node2-gpu-vram-bar" class="bg-cyan-500 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
        </div>

        <div class="bg-slate-950/80 p-4 rounded-xl border border-slate-800 space-y-3">
          <div class="text-xs font-bold text-slate-300 flex items-center justify-between">
            <span>🧠 AMD XDNA 2 NPU Coprocessor</span>
            <div class="text-[11px] font-mono">
              <span id="node2-npu-clock" class="text-amber-400">792MHz</span> | 
              <span id="node2-npu-tops" class="text-emerald-400">25 TOPS</span>
            </div>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">Hardware Compute Load (<span id="node2-npu-tasks" class="text-slate-300">0/16 Tasks</span>)</span>
              <span id="node2-npu-busy" class="text-emerald-400 font-bold">0%</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
              <div id="node2-npu-busy-bar" class="bg-gradient-to-r from-emerald-500 to-teal-400 h-2.5 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
          <div class="flex justify-between text-xs font-mono text-slate-400">
            <span>Role: <span id="node2-npu-model" class="text-slate-200">Worker Standby / Draft Target</span></span>
            <span id="node2-npu-status" class="text-emerald-400">Ready (D0)</span>
          </div>
        </div>

        <div class="bg-slate-950/80 p-4 rounded-xl border border-slate-800 space-y-3">
          <div class="text-xs font-bold text-slate-300 flex items-center justify-between">
            <span>⚡ Wirespeeds & Bandwidth</span>
            <span class="text-emerald-400 font-mono text-[11px]">USB4STREAM Active</span>
          </div>
          <div>
            <div class="flex justify-between text-xs mb-1 font-mono">
              <span class="text-slate-400">USB4 DMA Tunnel (Dual 40G - 80 Gbps)</span>
              <span id="node2-usb4-speed" class="text-cyan-300 font-bold">0.0 MB/s</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              <div id="node2-usb4-bar" class="bg-cyan-500 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>

  <script>
    function escapeHtml(str) {
      if (str === null || str === undefined) return '';
      return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }

    function formatToolCalls(text) {
      if (!text) return '';
      let safe = escapeHtml(text);
      safe = safe.replaceAll('🛠️ [TOOL CALL]: ', '<span class="inline-block px-2 py-0.5 my-1 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 font-bold">🛠️ TOOL CALL: </span>');
      return safe;
    }

    function setPrompt(txt) {
      const input = document.getElementById('prompt-input');
      if (input) {
        input.value = txt;
        input.focus();
      }
    }

    async function sendTestPrompt() {
      const input = document.getElementById('prompt-input');
      const btn = document.getElementById('send-prompt-btn');
      const status = document.getElementById('stimulus-status');
      const text = (input ? input.value : '').trim();
      if (!text) return;

      if (btn) {
        btn.disabled = true;
        btn.classList.add('opacity-50', 'cursor-not-allowed');
      }
      if (status) {
        status.innerText = 'Dispatching to :13305 proxy...';
        status.className = 'text-xs font-mono text-amber-400 animate-pulse';
      }

      try {
        activeFastPolling = true;
        if (typeof runPollCycle === 'function') runPollCycle();

        const modeSelect = document.getElementById('mode-select');
        const isThinking = modeSelect ? (modeSelect.value === 'thinking') : true;

        const proxyUrl = `${window.location.protocol}//${window.location.hostname}:13305/v1/chat/completions`;
        const res = await fetch(proxyUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: [{ role: 'user', content: text }],
            stream: true,
            max_tokens: 8192,
            temperature: isThinking ? 1.0 : 0.7,
            top_p: isThinking ? 0.95 : 0.80,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: isThinking ? 0.0 : 1.5,
            repeat_penalty: 1.0,
            chat_template_kwargs: {
              enable_thinking: isThinking,
              preserve_thinking: false,
              reasoning_effort: isThinking ? 'xhigh' : 'low'
            }
          })
        });

        if (!res.ok) {
          if (status) {
            status.innerText = `HTTP ${res.status}: ${res.statusText}`;
            status.className = 'text-xs font-mono text-red-400';
          }
          return;
        }

        if (status) {
          status.innerText = 'Streaming response tokens...';
          status.className = 'text-xs font-mono text-purple-400 font-bold';
        }

        const reader = res.body.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
        }

        if (status) {
          status.innerText = 'Synthesis Completed ✓';
          status.className = 'text-xs font-mono text-emerald-400 font-bold';
        }
      } catch (err) {
        if (status) {
          status.innerText = `Error: ${err.message}`;
          status.className = 'text-xs font-mono text-red-400';
        }
      } finally {
        activeFastPolling = false;
        if (btn) {
          btn.disabled = false;
          btn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
      }
    }

    let activeFastPolling = false;
    let currentPollInterval = 1000;
    let isUpdating = false;
    let pollTimer = null;
    let lastRenderedHistKey = "";

    function scheduleNextPoll(ms) {
      if (pollTimer) clearTimeout(pollTimer);
      pollTimer = setTimeout(runPollCycle, ms);
    }

    async function runPollCycle() {
      if (!isUpdating) {
        isUpdating = true;
        try {
          await updateHardwareStats();
        } catch (e) {
          console.error("Poll error:", e);
        } finally {
          isUpdating = false;
        }
      }
      scheduleNextPoll(currentPollInterval);
    }

    window.addEventListener('beforeunload', function() {
      if (pollTimer) clearTimeout(pollTimer);
    });

    function toggleFullContext() {
      const drawer = document.getElementById('live-context-drawer');
      const btn = document.getElementById('toggle-context-btn');
      if (!drawer || !btn) return;
      if (drawer.classList.contains('hidden')) {
        drawer.classList.remove('hidden');
        btn.innerText = '▲ Hide Full System Prompt & Context';
      } else {
        drawer.classList.add('hidden');
        const countEl = document.getElementById('context-words-count');
        const cnt = countEl ? countEl.innerText : '0';
        btn.innerHTML = `▼ View Full System Prompt &amp; Conversation Context (<span id="context-words-count">${cnt}</span> words)`;
      }
    }

    function renderConsciousness(llm) {
      if (!llm || typeof llm !== 'object') return;

      const card = document.getElementById('consciousness-card');
      const badge = document.getElementById('brain-state-badge');
      const dot = document.getElementById('synapse-pulse-dot');
      
      const qBox = document.getElementById('live-question');
      const fullCtxBox = document.getElementById('live-full-context');
      const ctxBtn = document.getElementById('toggle-context-btn');
      const ctxWords = document.getElementById('context-words-count');
      const toolsTag = document.getElementById('live-tools-tag');

      const thinkBox = document.getElementById('live-thinking');
      const ansBox = document.getElementById('live-answer');
      
      const thinkCount = document.getElementById('live-think-count');
      const ansCount = document.getElementById('live-ans-count');
      const ttftTag = document.getElementById('live-ttft-tag');
      const tpsTag = document.getElementById('live-tps-tag');

      if (llm.model_name && llm.model_name !== "None Loaded") {
        const mNameEl = document.getElementById('cluster-model-name');
        if (mNameEl) mNameEl.innerText = llm.model_name;
        const mDetEl = document.getElementById('cluster-model-details');
        if (mDetEl) mDetEl.innerText = llm.model_details || '';
        const mBadgeEl = document.getElementById('cluster-model-badge');
        if (mBadgeEl) {
          mBadgeEl.innerText = llm.status_tag || 'ONLINE';
          mBadgeEl.className = (llm.status_tag === 'READY' ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold');
        }
      } else if (llm.role) {
        const mNameEl = document.getElementById('cluster-model-name');
        if (mNameEl) mNameEl.innerText = llm.role;
        const mDetEl = document.getElementById('cluster-model-details');
        if (mDetEl) mDetEl.innerText = llm.model_details || 'Standby';
        const mBadgeEl = document.getElementById('cluster-model-badge');
        if (mBadgeEl) {
          mBadgeEl.innerText = llm.status_tag || 'STANDBY';
          mBadgeEl.className = 'text-amber-400 font-bold';
        }
      }
      if (llm.split_mode_label) {
        const splitBadge = document.getElementById('cluster-split-label');
        if (splitBadge) splitBadge.innerText = llm.split_mode_label;
      }
      const liveModelTag = document.getElementById('live-model-tag');
      if (liveModelTag && llm.model_name && llm.model_name !== "None Loaded") {
        liveModelTag.innerText = llm.model_name;
      }

      const activeOrLast = llm.active_prompt || (Array.isArray(llm.recent_prompts) && llm.recent_prompts.length > 0 ? llm.recent_prompts[llm.recent_prompts.length - 1] : null);

      if (llm.active_prompt) {
        const ap = llm.active_prompt;
        if (card) card.classList.add('synaptic-active');
        if (dot) dot.className = "w-2.5 h-2.5 rounded-full bg-purple-400 pulse-active";
        
        if (badge) {
          if (!ap.ttft && (ap.tokens_out === 0 || !ap.tokens_out)) {
            const elapsedSec = ap.start_time ? Math.round(Date.now() / 1000 - ap.start_time) : 0;
            badge.className = "px-4 py-2 text-xs font-mono font-bold rounded-xl bg-amber-500/20 text-amber-300 border border-amber-500/40 flex items-center gap-2.5 pulse-active shadow-lg shadow-amber-950/50";
            badge.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span> ⏳ INGESTING CONTEXT (${elapsedSec}s · ${ap.tokens_in || 'Large'} Tokens Pre-computing)`;
          } else if (ap.phase === "inner_monologue") {
            badge.className = "px-4 py-2 text-xs font-mono font-bold rounded-xl bg-purple-500/20 text-purple-300 border border-purple-500/40 flex items-center gap-2.5 pulse-active shadow-lg shadow-purple-950/50";
            badge.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-purple-400"></span> 🧠 DEEP REASONING ACTIVE (${ap.tps || '--'})`;
          } else if (ap.phase === "output_generation") {
            badge.className = "px-4 py-2 text-xs font-mono font-bold rounded-xl bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center gap-2.5 pulse-active shadow-lg shadow-emerald-950/50";
            badge.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span> ⚡ CONSCIOUS OUTPUT EMITTING (${ap.tps || '--'})`;
          }
        }

        if (qBox) qBox.innerText = ap.question || '(Streaming stimulus...)';
        
        if (ap.full_context && ap.full_context !== ap.question) {
          if (fullCtxBox) fullCtxBox.innerText = ap.full_context;
          if (ctxWords) ctxWords.innerText = ap.tokens_in || 0;
          if (ctxBtn) ctxBtn.classList.remove('hidden');
        } else {
          if (ctxBtn) ctxBtn.classList.add('hidden');
        }

        if (toolsTag) {
          if (ap.tools_available && ap.tools_available.length > 0) {
            toolsTag.classList.remove('hidden');
            toolsTag.innerText = `🛠️ ${ap.tools_available.length} TOOLS ATTACHED (${ap.tools_available.join(', ')})`;
          } else {
            toolsTag.classList.add('hidden');
          }
        }

        if (thinkBox) {
          if (ap.thinking) {
            thinkBox.innerHTML = formatToolCalls(ap.thinking) + (ap.phase === "inner_monologue" ? '<span class="live-cursor"></span>' : '');
            thinkBox.scrollTop = thinkBox.scrollHeight;
          } else {
            thinkBox.innerHTML = '<span class="text-purple-400/60 italic">Initiating neural cognition and strategy planning...</span><span class="live-cursor"></span>';
          }
        }

        if (ansBox) {
          if (ap.answer) {
            ansBox.innerHTML = formatToolCalls(ap.answer) + (ap.phase === "output_generation" ? '<span class="live-cursor-emerald"></span>' : '');
            ansBox.scrollTop = ansBox.scrollHeight;
          } else {
            ansBox.innerHTML = '<span class="text-slate-600 italic">(Waiting for inner monologue resolution...)</span>';
          }
        }

        if (thinkCount) thinkCount.innerText = `${ap.tokens_think || 0} thought tokens`;
        if (ansCount) ansCount.innerText = `${ap.tokens_answer || 0} output tokens`;
        if (ttftTag) ttftTag.innerText = `TTFT: ${ap.ttft || '--'}`;
        if (tpsTag) tpsTag.innerText = `${ap.tps || '--'}`;

      } else {
        if (card) card.classList.remove('synaptic-active');
        if (dot) dot.className = "w-2.5 h-2.5 rounded-full bg-slate-600";
        if (badge) {
          badge.className = "px-4 py-2 text-xs font-mono font-bold rounded-xl bg-slate-800 text-slate-400 border border-slate-700 flex items-center gap-2.5";
          badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-slate-500"></span> COGNITIVE ENGINE DORMANT (Listening on 13305)`;
        }

        if (activeOrLast) {
          if (qBox) qBox.innerText = activeOrLast.question || '';
          if (activeOrLast.full_context && activeOrLast.full_context !== activeOrLast.question) {
            if (fullCtxBox) fullCtxBox.innerText = activeOrLast.full_context;
            if (ctxWords) ctxWords.innerText = activeOrLast.tokens_in || 0;
            if (ctxBtn) ctxBtn.classList.remove('hidden');
          } else {
            if (ctxBtn) ctxBtn.classList.add('hidden');
          }

          if (toolsTag) {
            if (activeOrLast.tools_available && activeOrLast.tools_available.length > 0) {
              toolsTag.classList.remove('hidden');
              toolsTag.innerText = `🛠️ ${activeOrLast.tools_available.length} TOOLS (${activeOrLast.tools_available.join(', ')})`;
            } else {
              toolsTag.classList.add('hidden');
            }
          }

          if (thinkBox) thinkBox.innerHTML = formatToolCalls(activeOrLast.thinking) || '<span class="text-slate-600">(No thinking trace recorded)</span>';
          if (ansBox) ansBox.innerHTML = formatToolCalls(activeOrLast.answer) || '<span class="text-slate-600">(No output recorded)</span>';
          if (thinkCount) thinkCount.innerText = `${activeOrLast.tokens_think || activeOrLast.tokens_out || 0} tokens`;
          if (ansCount) ansCount.innerText = `${activeOrLast.tokens_answer || activeOrLast.tokens_out || 0} tokens`;
          if (ttftTag) ttftTag.innerText = `TTFT: ${activeOrLast.ttft || '--'}`;
          if (tpsTag) tpsTag.innerText = `${activeOrLast.tps || '--'}`;
        }
      }

      // Render Archive (only when history changes to prevent collapsing active user views)
      const histContainer = document.getElementById('history-container');
      if (histContainer && Array.isArray(llm.recent_prompts) && llm.recent_prompts.length > 0) {
        const histKey = llm.recent_prompts.map(p => `${p.id || ''}_${p.time || ''}_${p.tokens_out || 0}`).join('|');
        if (histKey !== lastRenderedHistKey) {
          lastRenderedHistKey = histKey;
          histContainer.innerHTML = llm.recent_prompts.map(p => {
            const qText = String(p.question || '');
            const hasThink = p.thinking && String(p.thinking).trim().length > 0;
            const hasFullCtx = p.full_context && String(p.full_context).trim().length > 0;
            return `
              <div class="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden shadow-md">
                <button onclick="this.nextElementSibling.classList.toggle('hidden')" class="w-full text-left p-4 hover:bg-slate-900/60 transition-colors flex items-center justify-between gap-4">
                  <div class="flex items-center gap-3 truncate">
                    <span class="px-2 py-0.5 rounded text-[10px] font-mono bg-purple-500/20 text-purple-300 border border-purple-500/30 font-bold">${escapeHtml(p.time || '--')}</span>
                    <span class="text-sm font-semibold text-slate-200 truncate">${escapeHtml(qText.substring(0, 90))}${qText.length > 90 ? '...' : ''}</span>
                  </div>
                  <div class="flex items-center gap-3 text-xs font-mono shrink-0">
                    <span class="text-purple-300/80">${p.tokens_out || 0} tokens</span>
                    <span class="text-emerald-400 font-bold">${escapeHtml(p.tps || '--')}</span>
                    <span class="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">✓ ARCHIVED</span>
                  </div>
                </button>
                
                <div class="hidden p-5 border-t border-slate-800/80 bg-slate-950 space-y-4">
                  <div>
                    <div class="text-[11px] font-bold text-amber-400 font-mono mb-1">📝 LATEST USER PROMPT / STIMULUS</div>
                    <div class="bg-slate-900 p-3 rounded-lg border border-slate-800 text-xs font-mono text-slate-200 whitespace-pre-wrap">${escapeHtml(qText)}</div>
                  </div>

                  ${hasFullCtx ? `
                  <div>
                    <div class="text-[11px] font-bold text-amber-300 font-mono mb-1">📜 FULL MULTI-TURN CONTEXT &amp; SYSTEM PROMPT</div>
                    <div class="bg-slate-900/90 p-3 rounded-lg border border-slate-800 text-xs font-mono text-slate-400 whitespace-pre-wrap max-h-72 overflow-y-auto custom-scroll">${escapeHtml(p.full_context)}</div>
                  </div>` : ''}

                  ${hasThink ? `
                  <div>
                    <div class="text-[11px] font-bold text-purple-400 font-mono mb-1">🧠 INNER MONOLOGUE (&lt;think&gt;)</div>
                    <div class="bg-purple-950/20 p-3 rounded-lg border border-purple-900/30 text-xs font-mono text-purple-200 whitespace-pre-wrap max-h-72 overflow-y-auto custom-scroll">${escapeHtml(p.thinking)}</div>
                  </div>` : ''}

                  <div>
                    <div class="text-[11px] font-bold text-emerald-400 font-mono mb-1">⚡ CONSCIOUS SYNTHESIS &amp; TOOL EMISSIONS</div>
                    <div class="bg-emerald-950/20 p-3 rounded-lg border border-emerald-900/30 text-xs font-mono text-emerald-200 whitespace-pre-wrap max-h-96 overflow-y-auto custom-scroll">${escapeHtml(p.answer || '')}</div>
                  </div>

                  <div class="text-[11px] font-mono text-slate-500 flex flex-wrap gap-4 pt-1">
                    <span>Duration: ${escapeHtml(String(p.duration || '--'))}s</span>
                    <span>TTFT: ${escapeHtml(String(p.ttft || '--'))}</span>
                    <span>Speed: ${escapeHtml(String(p.tps || '--'))}</span>
                    <span>Input Tokens: ${p.tokens_in || 0}</span>
                    <span>Model: ${escapeHtml(String(p.model || 'Cluster Model'))}</span>
                  </div>
                </div>
              </div>
            `;
          }).reverse().join('');
        }
      }
    }

    async function updateHardwareStats() {
      try {
        const res = await fetch('/api/stats', { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        if (!data) return;
        
        let totalClusterMemUsed = 0;

        // Symmetric Node Mapping: Node 1 is always bosgame1, Node 2 is always bosgame2
        const isLocalBosgame1 = (data.local?.system?.hostname === 'bosgame1');
        const isLocalBosgame2 = (data.local?.system?.hostname === 'bosgame2');

        let n1 = isLocalBosgame2 ? data.remote : data.local;
        let n2 = isLocalBosgame2 ? data.local : data.remote;

        // Adaptive polling speed: 100ms when inference active, 1000ms when dormant
        if (n1?.llm?.active_prompt || activeFastPolling) {
          currentPollInterval = 100;
        } else {
          currentPollInterval = 1000;
        }

        // Render Node 1 (bosgame1 - Master)
        if (n1) {
          try {
            const errBanner1 = document.getElementById('node1-error-banner');
            const n1Role = document.getElementById('node1-role');

            if (n1.error) {
              if (errBanner1) {
                errBanner1.innerText = `⚠️ bosgame1 Warning: ${n1.error}`;
                errBanner1.classList.remove('hidden');
              }
              if (n1Role) {
                n1Role.innerText = 'OFFLINE';
                n1Role.className = 'text-xs px-2.5 py-1 rounded-md bg-red-500/20 text-red-400 font-mono font-bold border border-red-500/30';
              }
            } else {
              if (errBanner1) errBanner1.classList.add('hidden');
              if (n1Role) {
                n1Role.innerText = (n1.llm?.role ? n1.llm.role.toUpperCase() : 'PRIMARY MASTER');
                n1Role.className = 'text-xs px-2.5 py-1 rounded-md bg-emerald-500/20 text-emerald-300 font-mono font-bold';
              }
              if (n1.system) {
                const h = document.getElementById('node1-host'); if (h) h.innerText = n1.system.hostname || 'bosgame1';
                const o = document.getElementById('node1-os'); if (o) o.innerText = n1.system.os || '--';
                const k = document.getElementById('node1-kernel'); if (k) k.innerText = n1.system.kernel || '--';
                const u = document.getElementById('node1-uptime'); if (u) u.innerText = n1.system.uptime || '--';
              }
              if (n1.network_ips) {
                const eth = document.getElementById('node1-ip-eth'); if (eth) eth.innerText = n1.network_ips.eno1 || '192.168.137.51';
                const wif = document.getElementById('node1-ip-wifi'); if (wif) wif.innerText = n1.network_ips.wlp195s0 || 'N/A';
              }

              if (n1.memory) {
                const uUsed = Number(n1.memory.unified_used_gb ?? n1.memory.used_gb ?? 0);
                const uTotal = Number(n1.memory.unified_total_gb ?? 128.0);
                const uPct = Number(n1.memory.unified_percent ?? n1.memory.usage_percent ?? 0);
                totalClusterMemUsed += uUsed;

                const ut = document.getElementById('node1-unified-text'); if (ut) ut.innerText = `${uUsed.toFixed(1)} / ${uTotal.toFixed(0)} GB`;
                const up = document.getElementById('node1-unified-pct'); if (up) up.innerText = `${uPct}%`;
                const ub = document.getElementById('node1-unified-bar'); if (ub) ub.style.width = `${Math.min(uPct, 100)}%`;

                const vr = document.getElementById('node1-sub-vram'); if (vr) vr.innerText = `${n1.memory.vram_used_gb ?? 0} GB`;
                const sy = document.getElementById('node1-sub-sys'); if (sy) sy.innerText = `${n1.memory.system_used_gb ?? 0} GB`;
                const fr = document.getElementById('node1-sub-free'); if (fr) fr.innerText = `${n1.memory.unified_free_gb ?? 0} GB`;
              }

              if (n1.cpu) {
                const cu = document.getElementById('node1-cpu-usage'); if (cu) cu.innerText = (n1.cpu.usage_percent !== undefined ? n1.cpu.usage_percent : 0) + '%';
                const cub = document.getElementById('node1-cpu-usage-bar'); if (cub) cub.style.width = Math.min(100, Math.max(0, n1.cpu.usage_percent || 0)) + '%';
                const ct = document.getElementById('node1-cpu-temp'); if (ct) ct.innerText = (n1.cpu.temperature_c ? n1.cpu.temperature_c + '°C' : '--');
                const cc = document.getElementById('node1-cpu-cores'); if (cc) cc.innerText = (n1.cpu.cores ? n1.cpu.cores + ' Threads' : '32 Threads');
              }

              if (n1.gpu && !n1.gpu.error) {
                const gb = document.getElementById('node1-gpu-busy'); if (gb) gb.innerText = (n1.gpu.busy_percent || 0) + '%';
                const gbb = document.getElementById('node1-gpu-busy-bar'); if (gbb) gbb.style.width = (n1.gpu.busy_percent || 0) + '%';
                const gt = document.getElementById('node1-gpu-temp'); if (gt) gt.innerText = (n1.gpu.temperature_c ? n1.gpu.temperature_c + '°C' : '--');
                const gp = document.getElementById('node1-gpu-power'); if (gp) gp.innerText = (n1.gpu.power_watts ? n1.gpu.power_watts + 'W' : '--');
                const gc = document.getElementById('node1-gpu-clock'); if (gc) gc.innerText = n1.gpu.sclk || '2900MHz';
                const gv = document.getElementById('node1-gpu-vram');
                const gvb = document.getElementById('node1-gpu-vram-bar');
                const gttU = Number(n1.gpu.gtt_used_gb || n1.gpu.vram_used_gb || 0);
                const gttT = Number(n1.gpu.gtt_total_gb || 120.0);
                const gttPct = Math.min(100, Math.max(0, (gttU / gttT) * 100));
                if (gv) gv.innerText = `${gttU.toFixed(1)} / ${gttT.toFixed(0)} GB (${gttPct.toFixed(1)}%)`;
                if (gvb) gvb.style.width = `${gttPct}%`;
              }

              if (n1.npu && n1.npu.available) {
                const nu = document.getElementById('node1-npu-busy'); if (nu) nu.innerText = (n1.npu.utilization_percent || 0) + '%';
                const nub = document.getElementById('node1-npu-busy-bar'); if (nub) nub.style.width = Math.min(100, Math.max(0, n1.npu.utilization_percent || 0)) + '%';
                const nt = document.getElementById('node1-npu-tasks'); if (nt) nt.innerText = `${n1.npu.tasks_active || 0} / ${n1.npu.tasks_max || 16} Tasks`;
                const ns = document.getElementById('node1-npu-status'); if (ns) ns.innerText = n1.npu.status || 'Active (D0)';
                const nc = document.getElementById('node1-npu-clock'); if (nc) nc.innerText = (n1.npu.h_clock_mhz || 1800) + 'MHz';
                const ntp = document.getElementById('node1-npu-tops'); if (ntp) ntp.innerText = (n1.npu.tops_curr || 58) + ' TOPS';
              }

              if (n1.wirespeeds && n1.wirespeeds.usb4stream_dma) {
                const dma = n1.wirespeeds.usb4stream_dma;
                let modeStr = dma.activity_mode ? ` [${dma.activity_mode}]` : '';
                if (dma.load_progress > 0 && dma.load_progress < 100) {
                    modeStr += ` (Loading: ${dma.load_progress.toFixed(1)}%)`;
                }
                const sp = document.getElementById('node1-usb4-speed'); if (sp) sp.innerText = `${dma.rate_mbs} MB/s (Peak: ${dma.peak_mbs} MB/s)${modeStr}`;
                const sb = document.getElementById('node1-usb4-bar'); if (sb) sb.style.width = Math.min(100, Math.max(dma.rate_mbs > 0 ? (dma.rate_mbs / 400.0 * 100) : 0, 1.0)) + '%';
              }

              if (n1.llm) {
                try { renderConsciousness(n1.llm); } catch (e) { console.error("renderConsciousness error:", e); }
              }
            }
          } catch (e1) {
            console.error("Node 1 render error:", e1);
          }
        }

        // Render Node 2 (bosgame2 - Compute Worker)
        if (n2) {
          try {
            const errBanner2 = document.getElementById('node2-error-banner');
            const n2Role = document.getElementById('node2-role');

            if (n2.error) {
              if (errBanner2) {
                errBanner2.innerText = `⚠️ bosgame2 Warning: ${n2.error}`;
                errBanner2.classList.remove('hidden');
              }
              if (n2Role) {
                n2Role.innerText = 'OFFLINE';
                n2Role.className = 'text-xs px-2.5 py-1 rounded-md bg-red-500/20 text-red-400 font-mono font-bold border border-red-500/30';
              }
            } else {
              if (errBanner2) errBanner2.classList.add('hidden');
              if (n2Role) {
                n2Role.innerText = (n2.llm?.role ? n2.llm.role.toUpperCase() : 'COMPUTE WORKER');
                n2Role.className = 'text-xs px-2.5 py-1 rounded-md bg-blue-500/20 text-blue-300 font-mono font-bold';
              }

              if (n2.system) {
                const h = document.getElementById('node2-host'); if (h) h.innerText = n2.system.hostname || 'bosgame2';
                const o = document.getElementById('node2-os'); if (o) o.innerText = n2.system.os || '--';
                const k = document.getElementById('node2-kernel'); if (k) k.innerText = n2.system.kernel || '--';
                const u = document.getElementById('node2-uptime'); if (u) u.innerText = n2.system.uptime || '--';
              }
              
              if (n2.network_ips) {
                const eth = document.getElementById('node2-ip-eth'); if (eth) eth.innerText = n2.network_ips.eno1 || '192.168.137.52';
                const wif = document.getElementById('node2-ip-wifi'); if (wif) wif.innerText = n2.network_ips.wlp195s0 || 'N/A';
              }

              if (n2.memory) {
                const uUsed = Number(n2.memory.unified_used_gb ?? n2.memory.used_gb ?? 0);
                const uTotal = Number(n2.memory.unified_total_gb ?? 128.0);
                const uPct = Number(n2.memory.unified_percent ?? n2.memory.usage_percent ?? 0);
                totalClusterMemUsed += uUsed;

                const ut = document.getElementById('node2-unified-text'); if (ut) ut.innerText = `${uUsed.toFixed(1)} / ${uTotal.toFixed(0)} GB`;
                const up = document.getElementById('node2-unified-pct'); if (up) up.innerText = `${uPct}%`;
                const ub = document.getElementById('node2-unified-bar'); if (ub) ub.style.width = `${Math.min(uPct, 100)}%`;

                const vr = document.getElementById('node2-sub-vram'); if (vr) vr.innerText = `${n2.memory.vram_used_gb ?? 0} GB`;
                const sy = document.getElementById('node2-sub-sys'); if (sy) sy.innerText = `${n2.memory.system_used_gb ?? 0} GB`;
                const fr = document.getElementById('node2-sub-free'); if (fr) fr.innerText = `${n2.memory.unified_free_gb ?? 0} GB`;
              }

              if (n2.cpu) {
                const cu = document.getElementById('node2-cpu-usage'); if (cu) cu.innerText = (n2.cpu.usage_percent !== undefined ? n2.cpu.usage_percent : 0) + '%';
                const cub = document.getElementById('node2-cpu-usage-bar'); if (cub) cub.style.width = Math.min(100, Math.max(0, n2.cpu.usage_percent || 0)) + '%';
                const ct = document.getElementById('node2-cpu-temp'); if (ct) ct.innerText = (n2.cpu.temperature_c ? n2.cpu.temperature_c + '°C' : '--');
                const cc = document.getElementById('node2-cpu-cores'); if (cc) cc.innerText = (n2.cpu.cores ? n2.cpu.cores + ' Threads' : '32 Threads');
              }

              if (n2.gpu && !n2.gpu.error) {
                const gb = document.getElementById('node2-gpu-busy'); if (gb) gb.innerText = (n2.gpu.busy_percent || 0) + '%';
                const gbb = document.getElementById('node2-gpu-busy-bar'); if (gbb) gbb.style.width = (n2.gpu.busy_percent || 0) + '%';
                const gt = document.getElementById('node2-gpu-temp'); if (gt) gt.innerText = (n2.gpu.temperature_c ? n2.gpu.temperature_c + '°C' : '--');
                const gp = document.getElementById('node2-gpu-power'); if (gp) gp.innerText = (n2.gpu.power_watts ? n2.gpu.power_watts + 'W' : '--');
                const gc = document.getElementById('node2-gpu-clock'); if (gc) gc.innerText = n2.gpu.sclk || '2900MHz';
                const gv = document.getElementById('node2-gpu-vram');
                const gvb = document.getElementById('node2-gpu-vram-bar');
                const gttU = Number(n2.gpu.gtt_used_gb || n2.gpu.vram_used_gb || 0);
                const gttT = Number(n2.gpu.gtt_total_gb || 120.0);
                const gttPct = Math.min(100, Math.max(0, (gttU / gttT) * 100));
                if (gv) gv.innerText = `${gttU.toFixed(1)} / ${gttT.toFixed(0)} GB (${gttPct.toFixed(1)}%)`;
                if (gvb) gvb.style.width = `${gttPct}%`;
              }

              if (n2.npu && n2.npu.available) {
                const nu = document.getElementById('node2-npu-busy'); if (nu) nu.innerText = (n2.npu.utilization_percent || 0) + '%';
                const nub = document.getElementById('node2-npu-busy-bar'); if (nub) nub.style.width = Math.min(100, Math.max(0, n2.npu.utilization_percent || 0)) + '%';
                const nt = document.getElementById('node2-npu-tasks'); if (nt) nt.innerText = `${n2.npu.tasks_active || 0} / ${n2.npu.tasks_max || 16} Tasks`;
                const ns = document.getElementById('node2-npu-status'); if (ns) ns.innerText = n2.npu.status || 'Ready (D0)';
                const nc = document.getElementById('node2-npu-clock'); if (nc) nc.innerText = (n2.npu.h_clock_mhz || 1800) + 'MHz';
                const ntp = document.getElementById('node2-npu-tops'); if (ntp) ntp.innerText = (n2.npu.tops_curr || 58) + ' TOPS';
              }

              if (n2.wirespeeds && n2.wirespeeds.usb4stream_dma) {
                const dma = n2.wirespeeds.usb4stream_dma;
                let modeStr = dma.activity_mode ? ` [${dma.activity_mode}]` : '';
                if (dma.load_progress > 0 && dma.load_progress < 100) {
                    modeStr += ` (Loading: ${dma.load_progress.toFixed(1)}%)`;
                }
                const sp = document.getElementById('node2-usb4-speed'); if (sp) sp.innerText = `${dma.rate_mbs} MB/s (Peak: ${dma.peak_mbs} MB/s)${modeStr}`;
                const sb = document.getElementById('node2-usb4-bar'); if (sb) sb.style.width = Math.min(100, Math.max(dma.rate_mbs > 0 ? (dma.rate_mbs / 400.0 * 100) : 0, 1.0)) + '%';
              }
            }
          } catch (e2) {
            console.error("Node 2 render error:", e2);
          }
        }

        const n1U = Number(n1?.npu?.utilization_percent || 0);
        const n2U = Number(n2?.npu?.utilization_percent || 0);
        const avgNpu = ((n1U + n2U) / 2.0).toFixed(1);
        const cnp = document.getElementById('cluster-npu-pct'); if (cnp) cnp.innerText = `${avgNpu}%`;
        const cnb = document.getElementById('cluster-npu-bar'); if (cnb) cnb.style.width = `${Math.min(avgNpu, 100)}%`;
        const cnt = document.getElementById('cluster-npu-tasks');
        if (cnt) {
          const totalTasks = (n1?.npu?.tasks_active || 0) + (n2?.npu?.tasks_active || 0);
          cnt.innerText = totalTasks > 0 ? `${totalTasks} Hardware Tasks Active` : 'Standby Ready';
        }

        const clusterMemPct = ((totalClusterMemUsed / 256.0) * 100).toFixed(1);
        const cmv = document.getElementById('cluster-mem-val'); if (cmv) cmv.innerText = `${totalClusterMemUsed.toFixed(1)} / 256 GB`;
        const cmp = document.getElementById('cluster-mem-pct'); if (cmp) cmp.innerText = `${clusterMemPct}%`;
        const cmb = document.getElementById('cluster-mem-bar'); if (cmb) cmb.style.width = `${Math.min(clusterMemPct, 100)}%`;

        const lu = document.getElementById('last-updated'); if (lu) lu.innerText = new Date().toLocaleTimeString();
      } catch (err) {
        console.error("Dashboard update failed:", err);
      }
    }

    document.addEventListener('DOMContentLoaded', () => {
      const pInput = document.getElementById('prompt-input');
      if (pInput) {
        pInput.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') sendTestPrompt();
        });
      }
    });

    // Start robust adaptive poll loop (no persistent connection exhaustion)
    scheduleNextPoll(10);
  </script>
</body>
</html>"""

_REMOTE_CACHE = None
_REMOTE_CACHE_TIME = 0.0
_REMOTE_LOCK = threading.Lock()

def get_remote_stats():
    global _REMOTE_CACHE, _REMOTE_CACHE_TIME
    now = time.time()
    with _REMOTE_LOCK:
        if _REMOTE_CACHE is not None and (now - _REMOTE_CACHE_TIME) < 0.9:
            return _REMOTE_CACHE

    peer_ip = "192.168.137.52" if socket.gethostname() == "bosgame1" else "192.168.137.51"
    try:
        req = urllib.request.Request(f"http://{peer_ip}:{DASHBOARD_PORT}/api/local_stats", headers={"User-Agent": "ClusterDash"})
        with urllib.request.urlopen(req, timeout=1.8) as resp:
            data = json.loads(resp.read().decode())
            with _REMOTE_LOCK:
                _REMOTE_CACHE = data
                _REMOTE_CACHE_TIME = time.time()
            return data
    except Exception as e:
        err = {"error": f"Peer unreachable ({peer_ip}): {str(e)}"}
        with _REMOTE_LOCK:
            _REMOTE_CACHE = err
            _REMOTE_CACHE_TIME = time.time()
        return err

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == "/api/consciousness":
                llm = get_llm_status()
                payload = json.dumps(llm).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            elif self.path == "/api/stream_consciousness":
                try:
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                try:
                    while True:
                        # Check if client disconnected via non-blocking select
                        rlist, _, _ = select.select([self.connection], [], [], 0.0)
                        if rlist:
                            peek = self.connection.recv(1, socket.MSG_PEEK)
                            if not peek:
                                break
                        llm = get_llm_status()
                        payload = f"data: {json.dumps(llm)}\n\n".encode("utf-8")
                        self.wfile.write(payload)
                        self.wfile.flush()
                        if llm.get("active_prompt"):
                            time.sleep(0.05)  # 20fps ultra-fluid stream when actively generating
                        else:
                            time.sleep(1.0)   # 1 Hz idle heartbeat
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception:
                    pass

            elif self.path == "/api/stats":
                local = get_all_local_stats()
                remote = get_remote_stats()

                payload = json.dumps({"local": local, "remote": remote}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            elif self.path == "/api/local_stats":
                local = get_all_local_stats()
                payload = json.dumps(local).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            else:
                payload = HTML_TEMPLATE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

    def log_message(self, format, *args):
        pass

class ReusableThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128
    timeout = 15.0

def run_proxy_server():
    try:
        proxy_server = ReusableThreadingTCPServer(("0.0.0.0", PROXY_PORT), StreamingAPIProxyHandler)
        print(f"Streaming API Proxy running on 0.0.0.0:{PROXY_PORT} -> backend {BACKEND_PORT}")
        proxy_server.serve_forever()
    except Exception as e:
        print(f"Proxy Server error: {e}")

def run_dashboard_server():
    try:
        dash_server = ReusableThreadingTCPServer(("0.0.0.0", DASHBOARD_PORT), DashboardHandler)
        print(f"Cluster Dashboard Web Server running on 0.0.0.0:{DASHBOARD_PORT}")
        dash_server.serve_forever()
    except Exception as e:
        print(f"Dashboard Server error: {e}")

if __name__ == "__main__":
    t_proxy = threading.Thread(target=run_proxy_server, daemon=True)
    t_proxy.start()
    run_dashboard_server()
