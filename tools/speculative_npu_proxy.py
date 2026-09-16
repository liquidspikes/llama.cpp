#!/usr/bin/env python3
"""
Heterogeneous NPU-GPU Speculative & Auto-Routing Proxy Server
Pairs FastFlowLM (AMD XDNA 2 NPU @ 96 tok/s) with
llama-server (Dual Strix Halo GPU Cluster 177B MoE @ 28 tok/s).

Features:
- Dynamic Semantic Auto-Routing (detects math/code/reasoning vs fast queries)
- Direct NPU and GPU manual routing overrides
- Health monitoring and metrics injection

Listens on port 8000, exposing an OpenAI-compatible /v1/chat/completions endpoint.
"""

import sys
import re
import time
import json
import socket
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

NPU_URL = "http://127.0.0.1:8999"
GPU_URL = "http://127.0.0.1:8001"
PROXY_PORT = 8000

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
        if "npu" in m_lower or "0.6b" in m_lower:
            return "npu", "manual override (requested NPU model)"
        if "gpu" in m_lower or "177b" in m_lower or "flash-next" in m_lower:
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

class SpeculativeEngine:
    def __init__(self, npu_url=NPU_URL, gpu_url=GPU_URL):
        self.npu_url = npu_url
        self.gpu_url = gpu_url

    def _post(self, url, endpoint, payload, timeout=120):
        req = urllib.request.Request(
            f"{url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def check_health(self):
        npu_ok = False
        gpu_ok = False
        try:
            r = urllib.request.urlopen(f"{self.npu_url}/v1/models", timeout=2)
            npu_ok = (r.status == 200)
        except Exception:
            pass

        try:
            r = urllib.request.urlopen(f"{self.gpu_url}/health", timeout=2)
            gpu_ok = (r.status == 200)
        except Exception:
            pass

        return npu_ok, gpu_ok

    def execute_routed(self, messages, max_tokens, temperature, model_param, stream=False):
        decision, reason = AutoRouter.route(messages, model_param)
        t0 = time.time()
        
        if decision == "npu":
            payload = {
                "model": "qwen3:0.6b",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
            res = self._post(self.npu_url, "/v1/chat/completions", payload, timeout=60)
            res["routed_engine"] = "AMD XDNA 2 NPU (0.6B FastFlowLM @ 95 tok/s)"
            res["routing_reason"] = reason
            res["routing_decision"] = "npu"
            return res
        else:
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
            res = self._post(self.gpu_url, "/v1/chat/completions", payload, timeout=120)
            res["routed_engine"] = "Dual Strix Halo GPU Cluster (177B MoE @ 28 tok/s)"
            res["routing_reason"] = reason
            res["routing_decision"] = "gpu"
            return res

class SpeculativeHTTPHandler(BaseHTTPRequestHandler):
    engine = SpeculativeEngine()

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            npu_ok, gpu_ok = self.engine.check_health()
            status_code = 200 if (npu_ok and gpu_ok) else 503
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok" if status_code == 200 else "degraded",
                "npu_coprocessor_online": npu_ok,
                "gpu_cluster_online": gpu_ok,
                "auto_router": "enabled (semantic complexity heuristic)",
                "npu_endpoint": NPU_URL,
                "gpu_endpoint": GPU_URL
            }).encode("utf-8"))
            return

        if self.path in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "object": "list",
                "data": [
                    {
                        "id": "auto",
                        "object": "model",
                        "owned_by": "strix-halo-heterogeneous-cluster",
                        "description": "Dynamic Semantic Auto-Routing (routes to NPU or GPU based on complexity)"
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
            }).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req_data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                return

            messages = req_data.get("messages", [])
            max_tokens = req_data.get("max_tokens", 64)
            temperature = req_data.get("temperature", 1.0)
            stream = req_data.get("stream", False)
            model_req = req_data.get("model", "auto")

            try:
                result = self.engine.execute_routed(messages, max_tokens, temperature, model_req, stream)
                resp_bytes = json.dumps(result).encode("utf-8")
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.send_header("X-Routed-Engine", result.get("routed_engine", ""))
                self.send_header("X-Routing-Reason", result.get("routing_reason", ""))
                self.end_headers()
                self.wfile.write(resp_bytes)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Inference engine failure: {str(e)}"}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

def run_server(port=PROXY_PORT):
    server = HTTPServer(("0.0.0.0", port), SpeculativeHTTPHandler)
    print(f"[AUTO-ROUTING-PROXY] Server listening on http://0.0.0.0:{port}")
    print(f"[AUTO-ROUTING-PROXY] Fast NPU Coprocessor: {NPU_URL}")
    print(f"[AUTO-ROUTING-PROXY] Deep GPU Cluster:     {GPU_URL}")
    print(f"[AUTO-ROUTING-PROXY] Auto-Routing:        ENABLED")
    server.serve_forever()

if __name__ == "__main__":
    port = PROXY_PORT
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
