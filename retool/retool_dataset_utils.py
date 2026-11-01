# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Dataset prompt formatting helpers for ReTool DAPO-Math-17k and AIME 2024/2025."""

from typing import Any, Optional

ANSWER_FORMAT = "\nThe answer format must be: \\boxed{'The final answer goes here.'}"
answer_format = ANSWER_FORMAT


def map_fn(row: dict[str, Any], *, data_source: Optional[str] = None) -> dict[str, Any]:
    """Normalize AIME 2024 and AIME 2025 evaluation rows into verl tool-agent format."""
    if data_source == "Maxwell-Jia/AIME_2024":
        problem, answer = row["Problem"], row["Answer"]
    elif data_source == "yentinglin/aime_2025":
        problem, answer = row["problem"], row["answer"]
    else:
        raise ValueError(f"Unsupported AIME data_source: {data_source!r}")
    prompt = problem + ANSWER_FORMAT
    return {
        "data_source": data_source.split("/")[1].lower(),
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "MATH",
        "reward_model": {"ground_truth": str(answer)},
        "agent_name": "tool_agent",
    }


def map_fn2(row: dict[str, Any]) -> dict[str, Any]:
    """Append boxed answer instructions and tool-agent routing to DAPO-Math-17k training rows."""
    content = row["prompt"][0]["content"]
    row["prompt"][0]["content"] = content + ANSWER_FORMAT
    row["agent_name"] = "tool_agent"
    return row
