# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lightweight SandboxFusion-compatible HTTP server (/run_code) for ReTool multi-turn code execution."""

import argparse
import asyncio
import json
import os
import resource
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="SandboxFusion Local Server")

MAX_CONCURRENT_PROCESSES = int(os.environ.get("SANDBOX_MAX_CONCURRENCY", "64"))
_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_PROCESSES)
    return _SEMAPHORE


class RunCodeRequest(BaseModel):
    compile_timeout: float = 10.0
    run_timeout: float = 30.0
    code: str = ""
    stdin: Optional[str] = ""
    memory_limit_MB: int = 1024
    language: str = "python"
    files: dict[str, Any] = Field(default_factory=dict)
    fetch_files: list[str] = Field(default_factory=list)


def _make_preexec_fn(memory_limit_mb: int):
    def _preexec():
        os.setsid()
        if memory_limit_mb and memory_limit_mb > 0:
            # Allow virtual address space headroom for OpenBLAS/NumPy/SymPy shared libraries
            limit_bytes = max(memory_limit_mb, 4096) * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            except (ValueError, OSError):
                pass

    return _preexec


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run_code")
async def run_code(req: RunCodeRequest) -> dict[str, Any]:
    if req.language not in ("python", "python3"):
        return {
            "status": "SandboxError",
            "message": f"Unsupported language: {req.language}",
            "compile_result": None,
            "run_result": None,
        }

    timeout = max(1.0, min(float(req.run_timeout or 30.0), 120.0))
    stdin_bytes = (req.stdin or "").encode("utf-8")

    async with _get_semaphore():
        with tempfile.TemporaryDirectory(prefix="sandbox_exec_") as tmp_dir:
            script_path = os.path.join(tmp_dir, "solution.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(req.code)

            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": tmp_dir,
                "TMPDIR": tmp_dir,
                "PYTHONIOENCODING": "utf-8",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            if "PYTHONPATH" in os.environ:
                env["PYTHONPATH"] = os.environ["PYTHONPATH"]

            start_t = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                script_path,
                stdin=asyncio.subprocess.PIPE if stdin_bytes else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmp_dir,
                env=env,
                preexec_fn=_make_preexec_fn(req.memory_limit_MB),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(input=stdin_bytes if stdin_bytes else None),
                    timeout=timeout,
                )
                elapsed = time.monotonic() - start_t
                stdout_str = stdout_bytes.decode("utf-8", errors="replace")
                stderr_str = stderr_bytes.decode("utf-8", errors="replace")
                return_code = proc.returncode if proc.returncode is not None else 0
                return {
                    "status": "Success" if return_code == 0 else "Failed",
                    "message": "",
                    "compile_result": None,
                    "run_result": {
                        "status": "Finished",
                        "execution_time": elapsed,
                        "return_code": return_code,
                        "stdout": stdout_str,
                        "stderr": stderr_str,
                    },
                }
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - start_t
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    proc.kill()
                await proc.wait()
                return {
                    "status": "Failed",
                    "message": "TimeLimitExceeded",
                    "compile_result": None,
                    "run_result": {
                        "status": "TimeLimitExceeded",
                        "execution_time": elapsed,
                        "return_code": -9,
                        "stdout": "",
                        "stderr": f"TimeoutError: execution exceeded {timeout}s\n",
                    },
                }
            except (RuntimeError, OSError) as exc:
                elapsed = time.monotonic() - start_t
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                return {
                    "status": "Failed",
                    "message": str(exc),
                    "compile_result": None,
                    "run_result": {
                        "status": "Error",
                        "execution_time": elapsed,
                        "return_code": -1,
                        "stdout": "",
                        "stderr": f"{type(exc).__name__}: {exc}\n",
                    },
                }


def check_server(host: str = "127.0.0.1", port: int = 18080) -> bool:
    url = f"http://{host}:{port}/run_code"
    payload = json.dumps({"code": "print(6*7)", "run_timeout": 5}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            print(json.dumps(body))
            return resp.status == 200 and body.get("status") == "Success"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description="SandboxFusion-compatible local HTTP server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--check", action="store_true", help="Check if local server is healthy and exit")
    args = parser.parse_args()
    if args.check:
        target_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        sys.exit(0 if check_server(target_host, args.port) else 1)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers, log_level="warning")


if __name__ == "__main__":
    main()
