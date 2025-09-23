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


def format_reward(predict_str: str) -> float:
    # Format is valid if it contains both <think>...</think> and <answer>...</answer>
    has_think = re.search(r"<think>.*?</think>", predict_str, flags=re.DOTALL | re.IGNORECASE) is not None
    has_answer = re.search(r"<answer>.*?</answer>", predict_str, flags=re.DOTALL | re.IGNORECASE) is not None
    return 1.0 if has_think and has_answer else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    # Prefer <answer>...</answer> content; fall back to boxed if requested; otherwise raw string
    answer = _extract_answer_tag_content(predict_str)
    if answer is None:
        answer = extract_boxed_content(predict_str) if use_boxed else predict_str
    # print("--------------------------------")
    # print(f"answer: {answer}")
    # print(f"ground_truth: {ground_truth}")
    # print("score: ", grade_answer(answer, ground_truth))
    # print("--------------------------------")
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )
