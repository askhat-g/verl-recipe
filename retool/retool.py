# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""ReTool multi-turn code-interpreter tool, dataset loader, and reward function."""

import argparse
import importlib
import logging
import re
import sys
from typing import Any

import datasets
from recipe.retool.retool_dataset_utils import map_fn, map_fn2
from recipe.retool.sandbox_fusion_tools import SandboxFusionTool

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


class CustomSandboxFusionTool(SandboxFusionTool):
    """SandboxFusion code-interpreter tool that strips markdown fences and auto-prints final expressions."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, Any, Any]:
        """Execute a generated Python snippet in the local SandboxFusion server."""
        code = parameters["code"]
        if not isinstance(code, str):
            code = str(code)

        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # sandbox has no score or metrics, use Nones
        return result, None, None


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset(parquet_file)["train"]
            data_source = "/".join(parquet_file.split("/")[-2:])
            if data_source in ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025"]:
                dataframe = dataframe.map(
                    map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                )
            else:
                dataframe = dataframe.map(map_fn2, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, Any], **kwargs) -> dict:
    """Compute ReTool boxed-math score with auxiliary multi-turn tool-usage shaping on incorrect rollouts."""
    del data_source, kwargs
    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    num_turns = extra_info["num_turns"]
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result


def supports_raiden_unbind_weights() -> bool:
    """Return True when the installed tpu-sync-torch WeightSynchronizer exposes unbind_weights()."""
    try:
        mod = importlib.import_module("tpu_sync.api.torch.weight_synchronizer")
        return hasattr(getattr(mod, "WeightSynchronizer", None), "unbind_weights")
    except ImportError:
        return False


def _main() -> None:
    parser = argparse.ArgumentParser(description="ReTool helper utilities")
    parser.add_argument(
        "--check-raiden-unbind", action="store_true", help="Exit 0 if WeightSynchronizer.unbind_weights exists"
    )
    args = parser.parse_args()
    if args.check_raiden_unbind:
        sys.exit(0 if supports_raiden_unbind_weights() else 1)


if __name__ == "__main__":
    _main()
