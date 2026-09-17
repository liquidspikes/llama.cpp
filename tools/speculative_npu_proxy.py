#!/usr/bin/env python3
"""
Heterogeneous NPU-GPU Speculative & Auto-Routing Proxy Server
Pairs FastFlowLM (AMD XDNA 2 NPU @ 96 tok/s) with
llama-server (Dual Strix Halo GPU Cluster 177B MoE @ 28 tok/s) and
Lemonade Orchestrator.

Listens on port 13306 (Lemonade API default) and 8000 (legacy proxy port).
Features:
- Dynamic Semantic Auto-Routing (detects math/code/reasoning vs fast queries)
- Transparent SSE Streaming and Non-Streaming Pass-Through
- Reverse Proxy to Lemonade daemon (on port 13307) for WebUI, static assets, and models
- Health monitoring and metrics injection
"""

import sys
import re
import time
import json
import socket
import threading
import requests
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

NPU_URL = "http://127.0.0.1:8999"
NODE2_NPU_URL = "http://192.168.137.52:8998"
LEMONADE_URL = "http://127.0.0.1:13307"

_GPU_URL_CACHE = {"url": "http://127.0.0.1:8002", "time": 0}
_HEALTH_CACHE = {"data": None, "code": 503, "time": 0}
_HEALTH_LOCK = threading.Lock()

def get_gpu_url():
    """
    Dynamically locates the active llama-server.real port (8001..8004)
    with 3-second cache to prevent multi-hop scan latency.
    """
    now = time.time()
    if now - _GPU_URL_CACHE["time"] < 3.0:
        return _GPU_URL_CACHE["url"]

    for port in (8001, 8002, 8003, 8004):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.2)
            if r.status_code in (200, 503):
                _GPU_URL_CACHE["url"] = f"http://127.0.0.1:{port}"
                _GPU_URL_CACHE["time"] = now
                return _GPU_URL_CACHE["url"]
        except Exception:
            pass
    return _GPU_URL_CACHE["url"]

PRIMARY_PORT = 13306
SECONDARY_PORT = 8000

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
}

class AutoRouter:
    """
    Evaluates incoming prompt complexity to dynamically steer requests:
    - Simple factual, short queries, translations, summaries -> AMD XDNA 2 NPU (95 tok/s, 0% GPU load)
    - Math, coding, logic riddles, deep reasoning -> Dual-GPU 177B MoE Cluster (28.4 tok/s)
    """
    COMPLEX_PATTERNS = [
        # Coding & Software Engineering
        r"\b(def |class |import |function |struct |template\s*<|void |async |await |nullptr|malloc|free|pointer|refactor|debug|traceback|regex|sql|query|algorithm)\b",
        r"```", # Markdown code blocks
        r"[{}();]{3,}", # Dense programming punctuation
        
        # Math, Physics & Formal Logic
        r"\b(prove|proof|calculate|solve|equation|formula|integral|derivative|matrix|vector|theorem|polynomial|eigen|factorial|probability|quantum|relativity)\b",
        r"\d+\s*[\*\/\^\+\-]\s*\d+", # Arithmetic like 19 * 23
        
        # Deep Reasoning, Logic Riddles & Multi-Step Analysis
        r"\b(step by step|step-by-step|reasoning|deduce|deduction|riddle|puzzle|all but \d+|sheep|paradox|counterexample)\b",
        r"\b(compare and contrast|in-depth|deep dive|comprehensive analysis|trade-offs|architectural design)\b",
        r"\b(explain why|explain how|justify|critique|evaluate the)\b",
    ]

    SIMPLE_PATTERNS = [
        r"^(hi|hello|hey|good morning|good afternoon|good evening|how are you|thanks|thank you)[\.!\?]?$",
        r"\b(what is the capital of|who is|who wrote|when was|translate|define|synonym|antonym|spelling)\b",
        r"\b(list \d+|name \d+|give me \d+ examples?)\b",
        r"\b(summarize in \w+ words?|in one (sentence|word))\b",
    ]

    @classmethod
    def route(cls, messages, model_param=""):
        # 1. Manual Overrides
        m_lower = (model_param or "").lower()
        if any(k in m_lower for k in ("npu", "0.6b", "flm")):
            return "npu", "manual override (requested NPU model)"
        if any(k in m_lower for k in ("spec", "speculative", "copilot", "hybrid")):
            return "speculative", "manual override (NPU speculative co-pilot mode)"
        if any(k in m_lower for k in ("gpu", "177b", "flash-next")):
            return "gpu", "manual override (requested GPU 177B model)"

        # 2. Extract full text from messages
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        words = len(text.split())

        # 3. High Context Length -> GPU
        if words > 200:
            return "gpu", f"high context length ({words} words)"

        # 4. Complex Patterns -> Speculative NPU Co-Pilot
        for pattern in cls.COMPLEX_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return "speculative", f"complex pattern matched ({pattern[:25]}...) -> NPU speculative co-pilot"

        # 5. Simple Patterns -> NPU
        for pattern in cls.SIMPLE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return "npu", "fast factual / simple query pattern matched"

        # 6. Default Fallback based on Length
        if words < 35:
            return "npu", f"short prompt ({words} words), routed to fast NPU"

        return "gpu", "standard reasoning threshold"

class ProxyHTTPHandler(BaseHTTPRequestHandler):
    session = requests.Session()

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self._handle_models()
            return

        if self.path in ("/health", "/healthz"):
            self._handle_health()
            return

        if self.path.startswith("/slots") or self.path.startswith("/props"):
            self._forward_to_gpu("GET")
            return

        # Reverse proxy any other GET request (e.g. Lemonade WebUI, static assets) to Lemonade core
        self._forward_to_lemonade("GET")

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat_completions()
            return

        if self.path in ("/v1/embeddings", "/embeddings"):
            self._handle_embeddings()
            return

        # Reverse proxy any other POST request (e.g. Lemonade API endpoints) to Lemonade core
        self._forward_to_lemonade("POST")

    def do_PUT(self):
        self._forward_to_lemonade("PUT")

    def do_DELETE(self):
        self._forward_to_lemonade("DELETE")

    def do_HEAD(self):
        self._forward_to_lemonade("HEAD")

    def _handle_embeddings(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        node2_url = f"{NODE2_NPU_URL}/v1/embeddings"
        try:
            req_data = json.loads(body.decode("utf-8")) if body else {}
            if "model" not in req_data or "gemma" not in req_data.get("model", "").lower():
                req_data["model"] = "embed-gemma:300m"
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            resp = self.session.post(node2_url, json=req_data, headers=headers, timeout=15)
            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(resp.content)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Node 2 NPU embeddings unreachable: {str(e)}"}).encode("utf-8"))

    def _handle_health(self):
        now = time.time()
        with _HEALTH_LOCK:
            if _HEALTH_CACHE["data"] and (now - _HEALTH_CACHE["time"] < 2.0):
                status_code = _HEALTH_CACHE["code"]
                resp_bytes = _HEALTH_CACHE["data"]
            else:
                npu_ok = False
                gpu_ok = False
                lem_ok = False
                node2_npu_ok = False
                try:
                    r = self.session.get(f"{NPU_URL}/v1/models", timeout=0.4)
                    npu_ok = (r.status_code == 200)
                except Exception:
                    pass

                try:
                    gpu_url = get_gpu_url()
                    r = self.session.get(f"{gpu_url}/health", timeout=0.4)
                    gpu_ok = (r.status_code == 200)
                except Exception:
                    pass

                try:
                    r = self.session.get(f"{LEMONADE_URL}/v1/models", timeout=0.4)
                    lem_ok = (r.status_code == 200)
                except Exception:
                    pass

                try:
                    r = self.session.get(f"{NODE2_NPU_URL}/v1/models", timeout=0.4)
                    node2_npu_ok = (r.status_code == 200)
                except Exception:
                    pass

                healthy = bool(gpu_ok)
                status_code = 200 if healthy else 503
                resp_data = {
                    "status": "ok" if healthy else "degraded",
                    "npu_coprocessor_online": npu_ok,
                    "gpu_cluster_online": gpu_ok,
                    "lemonade_core_online": lem_ok,
                    "node2_npu_embeddings_online": node2_npu_ok,
                    "auto_router": "enabled (semantic complexity heuristic)",
                    "primary_port": PRIMARY_PORT,
                    "secondary_port": SECONDARY_PORT
                }
                resp_bytes = json.dumps(resp_data, indent=2).encode("utf-8")
                _HEALTH_CACHE["data"] = resp_bytes
                _HEALTH_CACHE["code"] = status_code
                _HEALTH_CACHE["time"] = now

        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(resp_bytes)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def _handle_models(self):
        models_data = [
            {
                "id": "auto",
                "object": "model",
                "owned_by": "strix-halo-heterogeneous-cluster",
                "description": "Dynamic Semantic Auto-Routing (routes to NPU or GPU based on query complexity)"
            },
            {
                "id": "qwen3.8-flash-next-gpu",
                "object": "model",
                "owned_by": "dual-gpu-cluster",
                "description": "177B MoE Deep Reasoning on Dual Strix Halo (28.4 tok/s)"
            },
            {
                "id": "qwen3.8-flash-next-speculative",
                "object": "model",
                "owned_by": "strix-halo-heterogeneous-cluster",
                "description": "177B MoE Deep Reasoning accelerated by AMD XDNA 2 NPU Speculative Drafting (Node 1 NPU Draft -> Dual-GPU Verify)"
            },
            {
                "id": "qwen3:0.6b-npu",
                "object": "model",
                "owned_by": "amd-xdna2",
                "description": "Ultra-fast Coprocessor on /dev/accel/accel0 (95.4 tok/s, 0% GPU load)"
            },
            {
                "id": "embed-gemma:300m-npu",
                "object": "model",
                "owned_by": "bosgame2-xdna2",
                "description": "Zero-VRAM Text Embeddings on Node 2 NPU (dim=768, 0% GPU load)"
            }
        ]

        # Fetch extra models from Lemonade if available
        try:
            r = self.session.get(f"{LEMONADE_URL}/v1/models", timeout=2)
            if r.status_code == 200:
                lem_models = r.json().get("data", [])
                existing_ids = {m["id"] for m in models_data}
                for lm in lem_models:
                    if lm.get("id") not in existing_ids:
                        models_data.append(lm)
        except Exception:
            pass

        resp_bytes = json.dumps({"object": "list", "data": models_data}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(resp_bytes)

    def _handle_chat_completions(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            req_data = json.loads(body.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid JSON body: {str(e)}"}).encode("utf-8"))
            return

        messages = req_data.get("messages", [])
        max_tokens = req_data.get("max_tokens", 256)
        temperature = req_data.get("temperature", 1.0)
        stream = req_data.get("stream", False)
        model_req = req_data.get("model", "auto")

        decision, reason = AutoRouter.route(messages, model_req)

        if decision == "speculative":
            self._handle_speculative_chat(messages, max_tokens, temperature, stream)
            return

        if decision == "npu":
            backend_url = f"{NPU_URL}/v1/chat/completions"
            engine_label = "AMD XDNA 2 NPU (0.6B FastFlowLM @ 95 tok/s)"
            payload = {
                "model": "qwen3:0.6b",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
        else:
            gpu_url = get_gpu_url()
            backend_url = f"{gpu_url}/v1/chat/completions"
            engine_label = "Dual Strix Halo GPU Cluster (177B MoE @ 28 tok/s)"
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }

        # Attempt primary routed engine with automatic fallback
        try:
            self._dispatch_backend(backend_url, payload, engine_label, reason, decision, stream)
        except Exception as primary_err:
            fallback_url = get_gpu_url() if decision == "npu" else NPU_URL
            fallback_decision = "gpu" if decision == "npu" else "npu"
            fallback_label = "Dual Strix Halo GPU Cluster (Fallback)" if decision == "npu" else "AMD XDNA 2 NPU (Fallback)"
            print(f"[AUTO-ROUTER] Primary engine ({engine_label}) failed: {primary_err}. Falling back to {fallback_label}.")
            if fallback_decision == "npu":
                payload["model"] = "qwen3:0.6b"
            elif "model" in payload:
                del payload["model"]

            try:
                self._dispatch_backend(f"{fallback_url}/v1/chat/completions", payload, fallback_label, f"fallback after primary error: {primary_err}", fallback_decision, stream)
            except Exception as secondary_err:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": f"Both inference backends failed: primary={primary_err}, secondary={secondary_err}"
                }).encode("utf-8"))

    def _handle_speculative_chat(self, messages, max_tokens, temperature, stream):
        """
        Heterogeneous Speculative Decoding Pipeline:
        1. Node 1 XDNA 2 NPU (FastFlowLM @ 85 tok/s) drafts a high-speed reasoning plan (0 MB GPU VRAM).
        2. Dual-Node GPU Cluster (177B MoE @ 1M ctx) verifies the plan and executes with full precision.
        """
        user_prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_prompt = m.get("content", "")
                break

        draft_content = ""
        npu_ms = 0.0
        t0 = time.time()
        try:
            npu_payload = {
                "model": "qwen3:0.6b",
                "messages": [{"role": "user", "content": f"Provide a concise plan, outline, or key strategy for:\n{user_prompt[:500]}"}],
                "max_tokens": 48,
                "temperature": 0.0
            }
            r_npu = self.session.post(f"{NPU_URL}/v1/chat/completions", json=npu_payload, timeout=2.5)
            if r_npu.status_code == 200:
                draft_content = r_npu.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            npu_ms = (time.time() - t0) * 1000.0
        except Exception as e:
            npu_ms = (time.time() - t0) * 1000.0
            print(f"[SPECULATIVE] NPU draft step skipped: {e}")

        gpu_messages = list(messages)
        if draft_content:
            gpu_messages.append({
                "role": "assistant",
                "content": f"<think>\n[NPU Speculative Co-Pilot Plan ({npu_ms:.1f}ms)]:\n{draft_content}\n</think>\n"
            })
            engine_label = f"Heterogeneous Speculative Engine (NPU Draft {npu_ms:.0f}ms -> Dual-GPU 177B Verify)"
            reason = f"NPU pre-drafted reasoning outline ({npu_ms:.1f}ms, 0% GPU load)"
        else:
            engine_label = "Dual Strix Halo GPU Cluster (Direct MoE)"
            reason = "Direct GPU execution (NPU draft skipped)"

        gpu_url = get_gpu_url()
        backend_url = f"{gpu_url}/v1/chat/completions"
        payload = {
            "messages": gpu_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }

        try:
            self._dispatch_backend(backend_url, payload, engine_label, reason, "speculative", stream)
        except Exception as err:
            print(f"[SPECULATIVE] GPU dispatch failed: {err}. Falling back to NPU.")
            fallback_payload = {
                "model": "qwen3:0.6b",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
            self._dispatch_backend(f"{NPU_URL}/v1/chat/completions", fallback_payload, "AMD XDNA 2 NPU (Fallback)", f"fallback: {err}", "npu", stream)

    def _dispatch_backend(self, url, payload, engine_label, reason, decision, stream):
        if stream:
            resp = self.session.post(url, json=payload, stream=True, timeout=600)
            self.send_response(resp.status_code)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("X-Routed-Engine", engine_label)
            self.send_header("X-Routing-Reason", reason)
            self.send_header("X-Routing-Decision", decision)
            self._send_cors_headers()
            self.end_headers()

            try:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        if b"data: [DONE]" in chunk or b"[DONE]" in chunk:
                            break
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                resp.close()
            self.close_connection = True
        else:
            resp = self.session.post(url, json=payload, timeout=600)
            data = resp.json()
            data["routed_engine"] = engine_label
            data["routing_reason"] = reason
            data["routing_decision"] = decision

            resp_bytes = json.dumps(data).encode("utf-8")
            self.send_response(resp.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.send_header("X-Routed-Engine", engine_label)
            self.send_header("X-Routing-Reason", reason)
            self.send_header("X-Routing-Decision", decision)
            self._send_cors_headers()
            self.end_headers()
            try:
                self.wfile.write(resp_bytes)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True

    def _forward_to_lemonade(self, method):
        """
        Transparently reverse proxies unhandled HTTP requests to Lemonade core (port 13307).
        Enables seamless access to Lemonade WebUI, static assets, and control APIs.
        """
        target_url = f"{LEMONADE_URL}{self.path}"
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}

        body = None
        if method in ("POST", "PUT", "PATCH"):
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                body = self.rfile.read(content_length)

        try:
            resp = self.session.request(
                method=method,
                url=target_url,
                headers=headers,
                data=body,
                stream=True,
                timeout=120
            )

            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self._send_cors_headers()
            self.end_headers()

            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Lemonade core unreachable on {LEMONADE_URL}: {str(e)}"}).encode("utf-8"))

    def _forward_to_gpu(self, method):
        """
        Transparently forwards requests to active GPU backend (llama-server.real).
        Enables /slots, /props, and other llama.cpp endpoints across all proxy ports.
        """
        gpu_url = get_gpu_url()
        target_url = f"{gpu_url}{self.path}"
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}

        body = None
        if method in ("POST", "PUT", "PATCH"):
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                body = self.rfile.read(content_length)

        try:
            resp = self.session.request(
                method=method,
                url=target_url,
                headers=headers,
                data=body,
                timeout=10
            )
            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(resp.content)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"GPU backend unreachable on {gpu_url}: {str(e)}"}).encode("utf-8"))
            except Exception:
                pass
        self.close_connection = True

def start_listener(port):
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), ProxyHTTPHandler)
        print(f"[AUTO-ROUTING-PROXY] Listening on http://0.0.0.0:{port}")
        server.serve_forever()
    except OSError as e:
        print(f"[AUTO-ROUTING-PROXY] Port {port} not available or already bound: {e}")

def run_servers():
    print("=" * 70)
    print(" HETEROGENEOUS NPU-GPU SPECULATIVE & AUTO-ROUTING PROXY")
    print("=" * 70)
    gpu_url = get_gpu_url()
    print(f" • Fast NPU Coprocessor:  {NPU_URL} (AMD XDNA 2, 95 tok/s)")
    print(f" • Deep GPU Cluster:      {gpu_url} (177B MoE Dual Strix Halo, 28 tok/s)")
    print(f" • Lemonade Backend:      {LEMONADE_URL} (WebUI, Management)")
    print(f" • Primary Port:          {PRIMARY_PORT} (Lemonade API standard)")
    print(f" • Secondary Port:        {SECONDARY_PORT} (Speculative proxy legacy)")
    print("=" * 70)

    # Start optional backward-compatibility listeners (8000, 8001)
    for extra_port in (8000, 8001):
        if extra_port != PRIMARY_PORT:
            t = threading.Thread(target=start_listener, args=(extra_port,), daemon=True)
            t.start()

    start_listener(PRIMARY_PORT)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        PRIMARY_PORT = int(sys.argv[1])
    if len(sys.argv) > 2:
        SECONDARY_PORT = int(sys.argv[2])
    run_servers()
