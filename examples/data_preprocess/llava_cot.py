# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import argparse
import os
import json
import re

import datasets
from datasets import Features, Value

from verl.utils.hdfs_io import copy, makedirs


def main():
    parser = argparse.ArgumentParser()
    # Output paths
    parser.add_argument("--local_dir", default="/vlm/peirouliang/verl/data/llava_cot_40k")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--output_filename", default="train.parquet")

    # LLaVA-like raw jsonl
    parser.add_argument("--llava_json_path", default="/vlm/peirouliang/data/llava_cot_rl_46k_sample_40k.json")
    parser.add_argument("--images_root", default="/mnt/vlmdata/data/train_images/llava-cot-100k", help="Optional root to prefix llava image path")

    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    # ---------------------------
    # 1) 读取 LLaVA JSON/JSONL -> Dataset（不抽样）
    # ---------------------------
    _llava_records = []
    with open(args.llava_json_path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[" or args.llava_json_path.endswith(".json"):
            loaded = json.load(f)
            if isinstance(loaded, dict) and "data" in loaded and isinstance(loaded["data"], list):
                _llava_records = loaded["data"]
            elif isinstance(loaded, list):
                _llava_records = loaded
            else:
                print("[ERROR] JSON 格式不符合预期，应为列表或包含 data 列表的字典")
                _llava_records = []
        else:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                record = json.loads(line)
                _llava_records.append(record)

    def _replace_think_tags(text):
        """把 <think> 替换成 <|think|>，</think> 替换成 <|/think|>"""
        if not isinstance(text, str):
            return text
        text = re.sub(r"<think>", "<|think|>", text, flags=re.IGNORECASE)
        text = re.sub(r"</think>", "</|think|>", text, flags=re.IGNORECASE)
        text = re.sub(r"<answer>", "<|answer|>", text, flags=re.IGNORECASE)
        text = re.sub(r"</answer>", "</|answer|>", text, flags=re.IGNORECASE)
        return text
    
    def normalize_record_types(record):
        # 统一 id 与对话字段类型
        if "id" in record:
            record["id"] = str(record["id"])
        if "conversations" in record and isinstance(record["conversations"], list):
            for conv in record["conversations"]:
                if isinstance(conv, dict):
                    if "from" in conv:
                        conv["from"] = str(conv["from"])
                    if "value" in conv and not isinstance(conv["value"], str):
                        conv["value"] = str(conv["value"])
        return record

    _llava_records = [normalize_record_types(r) for r in _llava_records]
    print(f"[INFO] 原始LLaVA数据量: {len(_llava_records)}")
    
    # 不进行筛选与抽样，直接转换为 Dataset
    llava_sampled = datasets.Dataset.from_list(_llava_records)

    def _strip_think_tags(text):
        if not isinstance(text, str):
            return text
        return re.sub(r"</?(think|no_think)>", "", text, flags=re.IGNORECASE)

    def _strip_the_answer_is(text):
        # If text starts with "The answer is ...", keep only the trailing content
        if not isinstance(text, str):
            return text
        stripped = text.strip()
        stripped = re.sub(r"^\s*(?:the\s+answer\s+is)\s*[:：\-]?\s*", "", stripped, flags=re.IGNORECASE)
        # Remove surrounding quotes if present
        if len(stripped) >= 2 and ((stripped[0] == '"' and stripped[-1] == '"') or (stripped[0] == "'" and stripped[-1] == "'")):
            stripped = stripped[1:-1].strip()
        # If the remaining content is a single letter followed by a period (e.g., "C."), drop the trailing period
        match_single_letter_with_period = re.match(r"^\s*([A-Za-z])\.\s*$", stripped)
        if match_single_letter_with_period:
            stripped = match_single_letter_with_period.group(1)
        return stripped

    def _extract_think_span(text):
        """Return content inside <think>...</think>; if not found, return empty string."""
        if not isinstance(text, str):
            return ""
        match = re.search(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_answer_span(text):
        """Return content inside <\|answer\|>...</\|answer\|>; if not found, return original text."""
        if not isinstance(text, str):
            return text
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, flags=re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # 去掉末尾一个英文句号（若存在）
            if content.endswith('.'):
                content = content[:-1]
            return content
        return text

    def make_llava_map_fn(split):
        data_source = "think"

        def process_fn(example, idx):
            sample_id = example.get("id")
            sample_id = str(int(sample_id)) if isinstance(sample_id, (int, float)) else str(sample_id)

            image_path = example.get("image")
            # 统一生成绝对路径字符串；无图时给空列表
            if isinstance(image_path, (str, os.PathLike)):
                image_full = (
                    image_path
                    if os.path.isabs(image_path)
                    else os.path.join(args.images_root, str(image_path))
                )
                images = [str(image_full)]
            else:
                images = []

            conversations = example.get("conversations") or []
            human = ""
            gpt = ""
            if isinstance(conversations, list) and len(conversations) >= 2:
                a, b = conversations[0], conversations[1]
                if isinstance(a, dict):
                    human = a.get("value") or ""
                if isinstance(b, dict):
                    gpt = b.get("value") or ""

            # 提取think内容
            # think_content = _extract_think_span(gpt)
            
            # 拼接problem: human + <think>内容 + <\|answer\|>标签（保留前缀）
            # problem_parts = []
            # if human:
                # problem_parts.append(human)
            # if think_content:
            #     problem_parts.append(f"<think>{think_content}</think>")
            # 保留<\|answer\|>标签前缀
            # problem_parts.append("<\|answer\|>")
            # problem = "\n".join(problem_parts)
            problem = human or ""
            answer = gpt or ""
            # 去掉 answer 中的 <think>/<no_think> 标签
            # answer = _strip_think_tags(answer)
            # 仅保留 <\|answer\|>...</\|answer\|> 包裹的部分作为 ground_truth
            ground_truth = _extract_answer_span(answer)
            # 如果是 "The answer is xx" 的格式，只保留 xx
            ground_truth = _strip_the_answer_is(ground_truth)
            answer = _replace_think_tags(answer)
            choices = None

            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": problem}],
                "images": images,  # 统一 List[str]
                "ability": "think",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": sample_id,
                    "answer": answer,
                    "question": problem,
                    "choices": choices,
                },
            }

        return process_fn

    llava_mapped = llava_sampled.map(function=make_llava_map_fn("train"), with_indices=True, num_proc=8)
    print(f"[INFO] 数据映射后LLaVA数据量: {len(llava_mapped)}")

    # ------------------------------------------
    # 3) 只保留公共列 + 强制统一 Features
    # ------------------------------------------
    COMMON_COLS = ["data_source", "prompt", "images", "ability", "reward_model", "extra_info"]

    def _select_common(ds):
        keep = [c for c in COMMON_COLS if c in ds.column_names]
        return ds.select_columns(keep)

    llava_mapped = _select_common(llava_mapped)

    unified_features = Features({
        "data_source": Value("string"),
        "prompt": [{
            "content": Value("string"),
            "role": Value("string"),
        }],

        "images": [Value("string")],  # 关键：List[str]
        "ability": Value("string"),
        "reward_model": {
            "ground_truth": Value("string"),
            "style": Value("string"),
        },
        "extra_info": {
            "answer": Value("string"),
            "choices": Value("null"),          # 允许为 None
            "id": Value("string"),             # 统一为 string
            "index": Value("int64"),
            "question": Value("string"),
            "split": Value("string"),
        },
    })

    llava_mapped = llava_mapped.cast(unified_features)

    # ------------------------------------------
    # 4) 写 parquet（仅 LLaVA）
    # ------------------------------------------
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    output_path = os.path.join(local_dir, args.output_filename)

    llava_mapped.to_parquet(output_path)
    for i in range(2):
        print(llava_mapped[i])
    # 额外导出用于快速验证的 2.2parquet（随机抽样 10 条）
    sample_size = min(10, len(llava_mapped))
    if sample_size > 0:
        test_output_path = os.path.join(local_dir, "test.parquet")
        llava_mapped.shuffle(seed=42).select(range(sample_size)).to_parquet(test_output_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print(f"[OK] Wrote parquet to: {output_path}")
    if hdfs_dir:
        print(f"[OK] Copied to HDFS dir: {hdfs_dir}")


if __name__ == "__main__":
    main()
