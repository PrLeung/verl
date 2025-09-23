# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import re

from mathruler.grader import extract_boxed_content, grade_answer


def _extract_answer_tag_content(predict_str: str) -> str | None:
    pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(predict_str)
    return match.group(1) if match else None


def format_reward_think(predict_str: str) -> float:
    # Valid only if there is exactly one <think>...</think> and one <answer>...</answer>,
    # the whole string starts with <think> and ends with </answer>.
    if not predict_str:
        return 0.0

    # Anchor: start with <think>...</think> then <answer>...</answer> till the end
    pattern = re.compile(r"^\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$", re.IGNORECASE)
    anchored_match = pattern.search(predict_str) is not None

    if not anchored_match:
        return 0.0

    # Ensure only one occurrence for each tag pair
    open_think_count = len(re.findall(r"(?i)<think>", predict_str))
    close_think_count = len(re.findall(r"(?i)</think>", predict_str))
    open_answer_count = len(re.findall(r"(?i)<answer>", predict_str))
    close_answer_count = len(re.findall(r"(?i)</answer>", predict_str))

    only_once = (
        open_think_count == 1
        and close_think_count == 1
        and open_answer_count == 1
        and close_answer_count == 1
    )

    return 1.0 if only_once else 0.0

def format_reward_no_think(predict_str: str) -> float:
    # Valid only if <think>...</think> appears before <answer>...</answer>
    no_think_then_answer = (
        re.search(r"<no_think></no_think>.*?<answer>.*?</answer>", predict_str, flags=re.DOTALL | re.IGNORECASE)
        is not None
    )
    return 1.0 if no_think_then_answer else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    # Prefer <answer>...</answer> content; fall back to boxed if requested; otherwise raw string
    answer = _extract_answer_tag_content(predict_str)
    # print("--------------------------------")
    # print(f"answer: {answer}")
    # print(f"ground_truth: {ground_truth}")
    # print("score: ", grade_answer(answer, ground_truth))
    # print("--------------------------------")
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(data_source, solution_str, ground_truth, format_score: float = 0.1, extra_info=None) -> float:
    use_think = data_source == "llava_cot"
    format_fn = format_reward_think if use_think else format_reward_no_think
    solution_str=solution_str.lower()
    ground_truth=ground_truth.lower()
    return (1.0 - format_score) * acc_reward(solution_str, ground_truth) + format_score * format_fn(solution_str)
