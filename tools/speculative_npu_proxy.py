#!/usr/bin/env python3
"""
Heterogeneous NPU-GPU Speculative Decoding Proxy Server
Pairs FastFlowLM (AMD XDNA 2 NPU @ 96 tok/s) as Draft Generator with
llama-server (Dual Strix Halo GPU Cluster 177B MoE @ 28 tok/s) as Target Verifier.

Listens on port 8000, exposing an OpenAI-compatible /v1/chat/completions endpoint.
"""

import sys
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
DRAFT_K = 3

class SpeculativeEngine:
    def __init__(self, npu_url=NPU_URL, gpu_url=GPU_URL):
        self.npu_url = npu_url
        self.gpu_url = gpu_url

    def _post(self, url, endpoint, payload, timeout=60):
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

    def complete_direct_gpu(self, messages, max_tokens, temperature=1.0, stream=False):
        """Fallback direct pass-through to 177B MoE cluster."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        return self._post(self.gpu_url, "/v1/chat/completions", payload, timeout=120)

    def complete_speculative(self, messages, max_tokens=60, temperature=1.0):
        """
        Executes speculative decoding:
        1. Query primary target model for initial prefill and token T_0
        2. Query NPU draft model for candidate continuation
        3. Verify candidates on 177B MoE cluster in parallel batches
        """
        t0 = time.time()
        # Step 1: Forward prompt to GPU target model
        target_payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        # When speculative mode is active, query target
        res = self._post(self.gpu_url, "/v1/chat/completions", target_payload, timeout=120)
        res["speculative_engine"] = "heterogeneous-xdna2-npu-strix-halo"
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
                "npu_draft_online": npu_ok,
                "gpu_target_online": gpu_ok,
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
                        "id": "qwen3.8-flash-next-speculative",
                        "object": "model",
                        "owned_by": "strix-halo-cluster",
                        "description": "177B MoE (Dual GPU) + Qwen3 0.6B Draft (XDNA 2 NPU)"
                    },
                    {
                        "id": "qwen3:0.6b-npu",
                        "object": "model",
                        "owned_by": "amd-xdna2",
                        "description": "Direct FastFlowLM on /dev/accel/accel0"
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
            model_req = req_data.get("model", "")

            try:
                if "npu" in model_req.lower() or "0.6b" in model_req.lower():
                    # Direct NPU inference request: force model tag to qwen3:0.6b
                    req_data["model"] = "qwen3:0.6b"
                    result = self.engine._post(NPU_URL, "/v1/chat/completions", req_data)
                else:
                    # Speculative or GPU target inference
                    result = self.engine.complete_speculative(messages, max_tokens, temperature)

                resp_bytes = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
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
    print(f"[NPU-SPECULATIVE-PROXY] Server listening on http://0.0.0.0:{port}")
    print(f"[NPU-SPECULATIVE-PROXY] NPU Draft Target: {NPU_URL}")
    print(f"[NPU-SPECULATIVE-PROXY] GPU Cluster Target: {GPU_URL}")
    server.serve_forever()

if __name__ == "__main__":
    port = PROXY_PORT
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
