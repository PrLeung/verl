# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import argparse
import os
import json
import re
import random

import datasets
from datasets import Features, Value

from verl.utils.hdfs_io import copy, makedirs


def main():
    parser = argparse.ArgumentParser()
    # Output paths
    parser.add_argument("--local_dir", default="/vlm/peirouliang/verl/data/mix_100k")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--output_filename", default="train.parquet")
    parser.add_argument("--val_filename", default="test.parquet")
    parser.add_argument("--val_size", type=int, default=10, help="验证集大小，默认10条")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子，用于可重复的数据分割")

    # LLaVA-like raw jsonl
    parser.add_argument("--llava_json_path", default="/vlm/data/grpo_lpr/mix_100k.json")

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
    

    # def _replace_think_tags(text):
    #     """把 <think> 替换成 <|think|>，</think> 替换成 <|/think|>"""
    #     if not isinstance(text, str):
    #         return text
    #     text = re.sub(r"<no_think>", "<|think_no|>", text, flags=re.IGNORECASE)
    #     text = re.sub(r"</no_think>", "</|think_no|>", text, flags=re.IGNORECASE)
    #     text = re.sub(r"<answer>", "<|answer|>", text, flags=re.IGNORECASE)
    #     text = re.sub(r"</answer>", "</|answer|>", text, flags=re.IGNORECASE)
    #     return text

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
        match = re.search(r"<\|answer\|>([\s\S]*?)</\|answer\|>", text, flags=re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # 去掉末尾一个英文句号（若存在）
            if content.endswith('.'):
                content = content[:-1]
            return content
        return text

    def make_llava_map_fn(split):
        def process_fn(example, idx):
            sample_id = example.get("id")
            sample_id = str(int(sample_id)) if isinstance(sample_id, (int, float)) else str(sample_id)

            # 从样本中读取 data_source
            data_source = example.get("data_source", "unknown")

            # 直接从样本中读取图片路径；兼容 image: str 与 images: List[str]
            images = []
            image_path = example.get("image")
            images_list = example.get("images")
            if isinstance(image_path, (str, os.PathLike)):
                images = [str(image_path)]
            elif isinstance(images_list, list):
                norm_images = []
                for it in images_list:
                    if isinstance(it, (str, os.PathLike)):
                        norm_images.append(str(it))
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
            # answer = _replace_think_tags(answer)
            choices = None

            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": problem}],
                "images": images,  # 统一 List[str]
                "ability": data_source,
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

    # 先进行数据映射
    llava_mapped = llava_sampled.map(function=make_llava_map_fn("train"), with_indices=True, num_proc=8)
    print(f"[INFO] 数据映射后LLaVA数据量: {len(llava_mapped)}")
    
    # 分割训练集和验证集
    if args.val_size > 0 and len(llava_mapped) > args.val_size:
        # 设置随机种子确保可重复性
        random.seed(args.random_seed)
        
        # 生成随机索引
        indices = list(range(len(llava_mapped)))
        random.shuffle(indices)
        
        # 固定验证集大小为10条
        val_indices = indices[:args.val_size]
        train_indices = indices[args.val_size:]
        
        train_dataset = llava_mapped.select(train_indices)
        val_dataset = llava_mapped.select(val_indices)
        
        # 更新验证集数据中的split字段
        def update_split_to_val(example):
            example["extra_info"]["split"] = "val"
            return example
        
        val_dataset = val_dataset.map(update_split_to_val)
        
        print(f"[INFO] 训练集数据量: {len(train_dataset)}")
        print(f"[INFO] 验证集数据量: {len(val_dataset)}")
    else:
        train_dataset = llava_mapped
        val_dataset = None
        print(f"[INFO] 数据量不足或验证集大小为0，全部作为训练集: {len(train_dataset)}")

    # ------------------------------------------
    # 3) 只保留公共列 + 强制统一 Features
    # ------------------------------------------
    COMMON_COLS = ["data_source", "prompt", "images", "ability", "reward_model", "extra_info"]

    def _select_common(ds):
        keep = [c for c in COMMON_COLS if c in ds.column_names]
        return ds.select_columns(keep)

    train_dataset = _select_common(train_dataset)
    if val_dataset is not None:
        val_dataset = _select_common(val_dataset)

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

    train_dataset = train_dataset.cast(unified_features)
    if val_dataset is not None:
        val_dataset = val_dataset.cast(unified_features)

    # ------------------------------------------
    # 4) 写 parquet（训练集和验证集）
    # ------------------------------------------
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # 输出训练集
    train_output_path = os.path.join(local_dir, args.output_filename)
    train_dataset.to_parquet(train_output_path)
    print(f"[INFO] 训练集样本预览:")
    for i in range(min(3, len(train_dataset))):
        print(f"  样本 {i}: {train_dataset[i]}")
    
    # 输出验证集（如果存在）
    if val_dataset is not None:
        val_output_path = os.path.join(local_dir, args.val_filename)
        val_dataset.to_parquet(val_output_path)
        print(f"[INFO] 验证集样本预览:")
        for i in range(min(3, len(val_dataset))):
            print(f"  样本 {i}: {val_dataset[i]}")
    
    # 额外导出用于快速验证的 test.parquet（随机抽样 10 条）
    sample_size = min(10, len(train_dataset))
    if sample_size > 0:
        test_output_path = os.path.join(local_dir, "test.parquet")
        train_dataset.shuffle(seed=42).select(range(sample_size)).to_parquet(test_output_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print(f"[OK] 训练集已写入: {train_output_path}")
    if val_dataset is not None:
        print(f"[OK] 验证集已写入: {val_output_path}")
    if hdfs_dir:
        print(f"[OK] 已复制到 HDFS 目录: {hdfs_dir}")


if __name__ == "__main__":
    main()
