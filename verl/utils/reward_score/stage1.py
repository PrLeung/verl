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
    # 仅当 </|answer|> 恰好出现一次时提取其前面的内容
    if not predict_str:
        return None
    close_answer_count = len(re.findall(r"(?i)</\|answer\|>", predict_str))
    if close_answer_count != 1:
        return None
    pattern = re.compile(r"\s*(.*?)\s*</\|answer\|>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(predict_str)
    return match.group(1) if match else None


def format_reward(predict_str: str) -> float:
    # Valid only if the string ends with </|answer|> and does not start with <|answer|>
    if not predict_str:
        return 0.0

    s = predict_str.strip()
    s_lower = s.lower()
    ends_with_answer_close = s_lower.endswith("</|answer|>")
    starts_with_answer_open = s_lower.startswith("<|answer|>")
    close_answer_count = len(re.findall(r"(?i)</\|answer\|>", s))

    # 需要严格满足：以 </answer> 结尾、不开头为 <answer>、且 </answer> 只出现一次
    return 1.0 if (ends_with_answer_close and not starts_with_answer_open and close_answer_count == 1) else 0.0


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
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    return (1.0 - format_score) * acc_reward(solution_str, ground_truth) + format_score * format_reward(solution_str)
