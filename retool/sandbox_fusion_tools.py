# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Google LLC
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
"""Ray-managed SandboxFusion code execution tool with auto-started loopback server on port 18080."""

import logging
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    """Execution pool concurrency mode."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray-global semaphore rate limiter for concurrent code executions."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self) -> None:
        """Acquire one execution slot."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self) -> None:
        """Release one execution slot."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self) -> int:
        """Return active concurrent slot count."""
        return self.current_count


class ExecutionWorker:
    """Worker actor that ensures a loopback 4-worker SandboxFusion server is up and dispatches requests."""

    def __init__(
        self,
        enable_global_rate_limit: bool = True,
        rate_limit: int = 10,
        local_port: Optional[int] = 18080,
    ):
        if local_port is not None:
            self._ensure_local_sandbox_server(port=local_port)
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    @staticmethod
    def _is_sandbox_healthy(port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    @classmethod
    def _ensure_local_sandbox_server(cls, port: int = 18080, workers: int = 4) -> None:
        if cls._is_sandbox_healthy(port):
            return
        server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_fusion_server.py")
        if not os.path.isfile(server_script):
            return
        with open("/tmp/sandbox_fusion_server_worker.log", "a", encoding="utf-8") as log_file:
            subprocess.Popen(
                [sys.executable, server_script, "--host", "127.0.0.1", "--port", str(port), "--workers", str(workers)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(75):
            if cls._is_sandbox_healthy(port):
                return
            time.sleep(0.2)

    def _init_rate_limit(self, rate_limit: int):
        for attempt in range(5):
            try:
                return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)
            except ValueError:
                if attempt == 4:
                    raise
                time.sleep(0.2)
                try:
                    return ray.get_actor("rate-limiter")
                except ValueError:
                    continue

    def ping(self) -> bool:
        """Health-check ping."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> Optional[T]:
        """Execute ``fn`` under the optional rate limiter."""
        with ExitStack() as stack:
            if self.rate_limit_worker is not None:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                logger.warning(f"Error when executing code: {e}")
                return None


class _LocalExecutionMethod:
    def __init__(self, worker: ExecutionWorker):
        self._worker = worker

    async def remote(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> Optional[T]:
        return self._worker.execute(fn, *fn_args, **fn_kwargs)


class _LocalExecutionPool:
    def __init__(self, local_port: Optional[int] = 18080):
        self.execute = _LocalExecutionMethod(ExecutionWorker(enable_global_rate_limit=False, local_port=local_port))


def init_execution_pool(
    num_workers: int,
    enable_global_rate_limit: bool = True,
    rate_limit: int = 10,
    mode: PoolMode = PoolMode.ThreadMode,
    local_port: Optional[int] = 18080,
    ray_actor_options: Optional[dict[str, Any]] = None,
):
    """Initialize a Ray-shared or in-process execution pool.

    ``ray_actor_options`` are forwarded to ``.options()`` of the shared execution actor, e.g.
    ``{"resources": {"node:__internal_head__": 0.001}}`` to place it (and its loopback server) on the head node.
    """
    if not ray.is_initialized():
        return _LocalExecutionPool(local_port=local_port)
    if mode == PoolMode.ThreadMode:
        options = {
            "name": "sandbox-execution-pool",
            "get_if_exists": True,
            "max_concurrency": num_workers,
            **(ray_actor_options or {}),
        }
        for attempt in range(5):
            try:
                return (
                    ray.remote(ExecutionWorker)
                    .options(**options)
                    .remote(
                        enable_global_rate_limit=enable_global_rate_limit,
                        rate_limit=rate_limit,
                        local_port=local_port,
                    )
                )
            except ValueError:
                if attempt == 4:
                    raise
                time.sleep(0.2)
                try:
                    return ray.get_actor("sandbox-execution-pool")
                except ValueError:
                    continue
    raise NotImplementedError("Process mode is not implemented yet")


class SandboxFusionTool(BaseTool):
    """Tool wrapper for executing Python snippets against a SandboxFusion-compatible HTTP endpoint."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self.num_workers = config.get("num_workers", 128)
        self.rate_limit = config.get("rate_limit", 128)
        self.default_timeout = config.get("default_timeout", 8)
        self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", False)
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
        self.memory_limit_mb = config.get("memory_limit_mb", 1024)
        if self.sandbox_fusion_url == "":
            raise ValueError("sandbox_fusion_url is not set")
        parsed_url = urllib.parse.urlparse(self.sandbox_fusion_url)
        env_port = os.environ.get("SANDBOX_FUSION_PORT")
        if env_port and parsed_url.hostname in ("127.0.0.1", "localhost"):
            parsed_url = parsed_url._replace(netloc=f"{parsed_url.hostname}:{int(env_port)}")
            self.sandbox_fusion_url = urllib.parse.urlunparse(parsed_url)
        self.local_port = parsed_url.port if parsed_url.hostname in ("127.0.0.1", "localhost") else None
        self.ray_actor_options = config.get("ray_actor_options")
        self._execution_pool = None
        logger.info(f"Init SandboxFusionTool with config: {config}")

    @property
    def execution_pool(self):
        """Execution pool, created on first use so that loading tool schemas (e.g. in the dataset) starts nothing."""
        if self._execution_pool is None:
            self._execution_pool = init_execution_pool(
                num_workers=self.num_workers,
                enable_global_rate_limit=self.enable_global_rate_limit,
                rate_limit=self.rate_limit,
                mode=PoolMode.ThreadMode,
                local_port=self.local_port,
                ray_actor_options=self.ray_actor_options,
            )
        return self._execution_pool

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the tool schema in OpenAI function-calling format."""
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """Create per-trajectory tool state."""
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, Any, Any]:
        """Execute a single ``code_interpreter`` tool call."""
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        if isinstance(result, ToolResponse):
            return result, None, None
        return ToolResponse(text=None if result is None else str(result)), None, None

    def execute_code(self, instance_id: str, code: str, timeout: int = 8, language: str = "python") -> ToolResponse:
        """Send ``code`` to the configured SandboxFusion HTTP endpoint and format stdout/stderr."""
        _result_status, metadata = _process_single_case(
            0, None, None, self.sandbox_fusion_url, code, timeout, self.memory_limit_mb, language
        )
        if metadata["run_status"] == "Finished":
            actual_output = metadata["stdout"] + metadata["stderr"]
            logger.debug(f"actual_output from sandbox fusion: {actual_output},{instance_id}")
            return ToolResponse(text=actual_output)
        return ToolResponse(text="no stdout here")

    async def calc_reward(self, instance_id: str, **kwargs) -> Any:
        """Return accumulated tool-level rewards."""
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release per-trajectory tool state."""
        del self._instance_dict[instance_id]
