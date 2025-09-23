import re
from verl.utils.reward_score.stage2 import (
    format_reward_think,
    format_reward_no_think,
)
# def format_reward(predict_str: str) -> float:
#     # Valid only if the string ends with </answer> and does not start with <answer>
#     if not predict_str:
#         return 0.0

#     s = predict_str.strip()
#     s_lower = s.lower()
#     ends_with_answer_close = s_lower.endswith("</answer>")
#     starts_with_answer_open = s_lower.startswith("<answer>")
#     close_answer_count = len(re.findall(r"(?i)</answer>", s))

#     # 需要严格满足：以 </answer> 结尾、不开头为 <answer>、且 </answer> 只出现一次
#     return 1.0 if (ends_with_answer_close and not starts_with_answer_open and close_answer_count == 1) else 0.0



def format_reward(predict_str: str, data_source: str = None) -> float:
    return {
            "think_reward": format_reward_think(predict_str),
            "no_think_reward": format_reward_no_think(predict_str)
            }
