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
    parser.add_argument("--local_dir", default="/vlm/peirouliang/verl/data/llava_next_20k")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--output_filename", default="train.parquet")

    # LLaVA-like raw jsonl
    parser.add_argument("--llava_json_path", default="/vlm/peirouliang/data/llava_next_rl_75k_sample_20k.json")
    parser.add_argument("--images_root", default="/mnt/vlmdata/data/train_images/llava_next_raw_format", help="Optional root to prefix llava image path")
    parser.add_argument("--data_source", default="think_no", help="Data source identifier")
    parser.add_argument("--ability", default="think_no", help="Ability identifier")

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
        text = re.sub(r"<no_think>", "<|think_no|>", text, flags=re.IGNORECASE)
        text = re.sub(r"</no_think>", "</|think_no|>", text, flags=re.IGNORECASE)
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

    # def _strip_think_tags(text):
    #     if not isinstance(text, str):
    #         return text
    #     return re.sub(r"</?(think|no_think)>", "", text, flags=re.IGNORECASE)

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
        data_source = args.data_source

        def process_fn(example, idx):
            sample_id = example.get("id")
            sample_id = str(int(sample_id)) if isinstance(sample_id, (int, float)) else str(sample_id)

            # 统一生成绝对路径字符串；兼容 image: str 与 images: List[str]
            images = []
            image_path = example.get("image")
            images_list = example.get("images")
            if isinstance(image_path, (str, os.PathLike)):
                image_full = (
                    image_path
                    if os.path.isabs(image_path)
                    else os.path.join(args.images_root, str(image_path))
                )
                images = [str(image_full)]
            elif isinstance(images_list, list):
                norm_images = []
                for it in images_list:
                    if not isinstance(it, (str, os.PathLike)):
                        continue
                    it = str(it)
                    image_full = it if os.path.isabs(it) else os.path.join(args.images_root, it)
                    norm_images.append(str(image_full))
                images = norm_images

            # 解析文本，兼容 LLaVA 的 conversations 与 messages 结构
            def _strip_image_placeholder(text):
                if not isinstance(text, str):
                    return text
                return text.replace("<image>", "").strip()

            conversations = example.get("conversations") or []
            messages = example.get("messages") or []
            # assert 1==4, f'conversations:{conversations}, messages:{messages}'
            human = ""
            gpt = ""
            if isinstance(conversations, list) and len(conversations) >= 2:
                a, b = conversations[0], conversations[1]
                if isinstance(a, dict):
                    human = a.get("value") or ""
                if isinstance(b, dict):
                    gpt = b.get("value") or ""
            elif isinstance(messages, list) and len(messages) >= 1:
                # 取第一条 user/human 作为 human，最后一条 assistant/gpt 作为 gpt（若存在）
                for m in messages:
                    if isinstance(m, dict) and str(m.get("role")).lower() in ["user", "human"]:
                        if m.get("content"):
                            human = m.get("content")
                            break
                for m in reversed(messages):
                    if isinstance(m, dict) and str(m.get("role")).lower() in ["gpt", "assistant"]:
                        if m.get("content"):
                            gpt = m.get("content")
                            break
                if not gpt:
                    for m in reversed(messages):
                        if isinstance(m, dict) and str(m.get("role")).lower() not in ["user", "human"]:
                            if m.get("content"):
                                gpt = m.get("content")
                                break

            # human = _strip_image_placeholder(human)
            gpt = _strip_image_placeholder(gpt)

            problem = human or ""
            answer = gpt or ""
            # 去掉 answer 中的 <think>/<no_think> 标签，并仅保留 <\|answer\|>...</\|answer\|> 的内容
            # answer = _strip_think_tags(answer)
            ground_truth = _extract_answer_span(answer)
            answer = _replace_think_tags(answer)
            choices = None

            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": problem}],
                "images": images,  # 统一 List[str]
                "ability": args.ability,
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
    for i in range(10):
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
