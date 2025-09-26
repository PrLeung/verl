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

from mathruler.grader import grade_answer

def _extract_answer_tag_content(predict_str: str) -> str | None:
    if not predict_str:
        return None

    # 统计开闭标签数量（大小写不敏感）
    open_answer_count = len(re.findall(r"(?i)<\|answer\|>", predict_str))
    close_answer_count = len(re.findall(r"(?i)</\|answer\|>", predict_str))

    # 必须各出现一次
    if open_answer_count != 1 or close_answer_count != 1:
        return None

    # 必须以 </|answer|> 结尾
    if not re.search(r"(?i)</\|answer\|>\s*$", predict_str):
        return None

    # 提取中间内容
    pattern = re.compile(r"<\|answer\|>\s*(.*?)\s*</\|answer\|>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(predict_str)
    return match.group(1).strip() if match else None




def format_reward_think(predict_str: str) -> float:
    # Valid only if there is exactly one </think> and one <answer>...</answer>,
    # no opening <think> present, and the whole string ends with </answer>.
    if not predict_str:
        return 0.0

    # Anchor: must start with <think>...</think> then <answer>...</answer> till the end
    pattern = re.compile(r"^<\|think\|>[\s\S]*?</\|think\|>\s*<\|answer\|>[\s\S]*?</\|answer\|>\s*$", re.IGNORECASE)
    anchored_match = pattern.search(predict_str) is not None

    if not anchored_match:
        return 0.0

    # Ensure only one occurrence for each tag and no opening <think>
    open_think_count = len(re.findall(r"(?i)<\|think\|>", predict_str))
    close_think_count = len(re.findall(r"(?i)</\|think\|>", predict_str))
    open_answer_count = len(re.findall(r"(?i)<\|answer\|>", predict_str))
    close_answer_count = len(re.findall(r"(?i)</\|answer\|>", predict_str))
    # Do not allow any <no_think> tags in think mode
    any_no_think = re.search(r"(?i)</?\|think_no\|>", predict_str) is not None

    only_once = (
        open_think_count == 1
        and close_think_count == 1
        and open_answer_count == 1
        and close_answer_count == 1
        and not any_no_think
    )

    return 1.0 if only_once else 0.0

import re

def format_reward_no_think(predict_str: str) -> float:
    if not predict_str:
        return 0.0

    # Anchor: 必须匹配 <no_think>...</no_think> 然后 <answer>...</answer>，并覆盖整串
    pattern = re.compile(r"^<\|think_no\|>\s*</\|think_no\|>\s*<\|answer\|>[\s\S]*?</\|answer\|>\s*$", re.IGNORECASE)
    anchored_match = pattern.search(predict_str) is not None

    if not anchored_match:
        return 0.0

    # 确保只有一次标签
    open_no_think_count = len(re.findall(r"(?i)<\|think_no\|>", predict_str))
    close_no_think_count = len(re.findall(r"(?i)</\|think_no\|>", predict_str))
    open_answer_count = len(re.findall(r"(?i)<\|answer\|>", predict_str))
    close_answer_count = len(re.findall(r"(?i)</\|answer\|>", predict_str))
    any_think = re.search(r"(?i)</?\|think\|>", predict_str) is not None

    only_once = (
        open_no_think_count == 1
        and close_no_think_count == 1
        and open_answer_count == 1
        and close_answer_count == 1
        and not any_think
    )

    if not only_once:
        return 0.0

    return 1.0



def format_reward(predict_str: str, data_source: str = None) -> float:
    # 根据data_source决定使用哪种格式验证
    if data_source == "llava_cot":
        return format_reward_think(predict_str)
    else:
        return format_reward_no_think(predict_str)


def acc_reward(predict_str: str, ground_truth: str) -> float:
    # Prefer <answer>...</answer> content; fall back to boxed if requested; otherwise raw string
    answer = _extract_answer_tag_content(predict_str)
    # print("--------------------------------")
    # print(f"answer: {answer}")
    # print(f"ground_truth: {ground_truth}")
    # print("score: ", grade_answer(answer, ground_truth))
    # print("--------------------------------")
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(data_source, solution_str, ground_truth, format_score: float = 0.5, extra_info=None, stage=None) -> float:
    solution_str = solution_str.lower()
    ground_truth = ground_truth.lower()
    
    if stage == "stage1":
        format_score=0.1
        if data_source == "llava_cot":
            solution_str = '<|think|> </|think|><|answer|>'+solution_str
        else:
            solution_str = '<|think_no|></|think_no|><|answer|>'+solution_str
    elif stage == "stage2":
        if data_source == "llava_cot":
            solution_str = '<|think|>'+solution_str
        else:
            solution_str = '<|think_no|>'+solution_str
    elif stage == "stage3":
        solution_str='<|think'+solution_str
    else:
        solution_str=solution_str
    
    format_reward_score = format_reward(solution_str, data_source)
    acc_reward_score = acc_reward(solution_str, ground_truth)
    total_reward_score = (1.0 - format_score) * acc_reward_score + format_score * format_reward_score
    
    return total_reward_score, format_reward_score, acc_reward_score


# test_think = "<|think|> I believe the answer is 42. </|think|> <|answer|> 42 </|answer|>"
# test_no_think = "<|think_no|></|think_no|> <|answer|> 42 </|answer|>"
# print(format_reward_think(test_think))
# print(format_reward_no_think(test_no_think))
# print(_extract_answer_tag_content(test_think))

