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
LEMONADE_URL = "http://127.0.0.1:13307"

def get_gpu_url():
    """
    Dynamically locates the active llama-server.real port (8001..8004)
    or falls back to 8002.
    """
    for port in (8001, 8002, 8003, 8004):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.2)
            if r.status_code in (200, 503):
                return f"http://127.0.0.1:{port}"
        except Exception:
            pass
    return "http://127.0.0.1:8002"

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
        if any(k in m_lower for k in ("gpu", "177b", "flash-next")):
            return "gpu", "manual override (requested GPU 177B model)"

        # 2. Extract full text from messages
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        words = len(text.split())

        # 3. High Context Length -> GPU
        if words > 200:
            return "gpu", f"high context length ({words} words)"

        # 4. Complex Patterns -> GPU
        for pattern in cls.COMPLEX_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return "gpu", f"complex pattern matched ({pattern[:25]}...)"

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

        # Reverse proxy any other GET request (e.g. Lemonade WebUI, static assets) to Lemonade core
        self._forward_to_lemonade("GET")

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat_completions()
            return

        # Reverse proxy any other POST request (e.g. Lemonade API endpoints) to Lemonade core
        self._forward_to_lemonade("POST")

    def do_PUT(self):
        self._forward_to_lemonade("PUT")

    def do_DELETE(self):
        self._forward_to_lemonade("DELETE")

    def do_HEAD(self):
        self._forward_to_lemonade("HEAD")

    def _handle_health(self):
        npu_ok = False
        gpu_ok = False
        lem_ok = False
        try:
            r = self.session.get(f"{NPU_URL}/v1/models", timeout=2)
            npu_ok = (r.status_code == 200)
        except Exception:
            pass

        try:
            gpu_url = get_gpu_url()
            r = self.session.get(f"{gpu_url}/health", timeout=2)
            gpu_ok = (r.status_code == 200)
        except Exception:
            pass

        try:
            r = self.session.get(f"{LEMONADE_URL}/v1/models", timeout=2)
            lem_ok = (r.status_code == 200)
        except Exception:
            pass

        status_code = 200 if (npu_ok or gpu_ok or lem_ok) else 503
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        resp_data = {
            "status": "ok" if status_code == 200 else "degraded",
            "npu_coprocessor_online": npu_ok,
            "gpu_cluster_online": gpu_ok,
            "lemonade_core_online": lem_ok,
            "auto_router": "enabled (semantic complexity heuristic)",
            "primary_port": PRIMARY_PORT,
            "secondary_port": SECONDARY_PORT
        }
        self.wfile.write(json.dumps(resp_data, indent=2).encode("utf-8"))

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
                "id": "qwen3:0.6b-npu",
                "object": "model",
                "owned_by": "amd-xdna2",
                "description": "Ultra-fast Coprocessor on /dev/accel/accel0 (95.4 tok/s, 0% GPU load)"
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
            except (BrokenPipeError, ConnectionResetError):
                pass
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
            except (BrokenPipeError, ConnectionResetError):
                pass

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

def start_listener(port):
    server = ThreadingHTTPServer(("0.0.0.0", port), ProxyHTTPHandler)
    print(f"[AUTO-ROUTING-PROXY] Listening on http://0.0.0.0:{port}")
    server.serve_forever()

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

    t = threading.Thread(target=start_listener, args=(SECONDARY_PORT,), daemon=True)
    t.start()

    start_listener(PRIMARY_PORT)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        PRIMARY_PORT = int(sys.argv[1])
    if len(sys.argv) > 2:
        SECONDARY_PORT = int(sys.argv[2])
    run_servers()
