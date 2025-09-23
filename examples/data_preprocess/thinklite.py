# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

import argparse
import os
import io

import datasets
from datasets import Features, Value
from PIL import Image

from verl.utils.hdfs_io import copy, makedirs


def main():
    parser = argparse.ArgumentParser()
    # Output paths
    parser.add_argument("--local_dir", default="/vlm/peirouliang/verl/data/thinklite_70k")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--output_filename", default="train.parquet")

    # ThinkLite-70k parquet
    parser.add_argument("--thinklite_parquet", default="/vlm/data/russwang/ThinkLite-VL-70k/ThinkLite-VL-70k.parquet")
    parser.add_argument(
        "--thinklite_image_out_dir",
        default="/vlm/data/ThinkLiteVL_70k",
        help="Directory to save ThinkLiteVL byte-stream images",
    )

    # 保留 ThinkLite 处理，不包含 LLaVA Next

    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    os.makedirs(args.thinklite_image_out_dir, exist_ok=True)

    thinklite_ds = datasets.load_dataset("parquet", data_files=args.thinklite_parquet)["train"]
    # 将 bytes image 写成 jpg 文件路径；注意返回新列 image(str)，并移除原始二进制列
    def _thinklite_image_to_path(example):
        image = example.get("image")
        sample_id = example.get("id")
        sample_id = str(int(sample_id)) if isinstance(sample_id, (int, float)) else str(sample_id)

        # bytes/bytearray -> 写盘为 JPG；已是 str/path 则原样返回
        if isinstance(image, (bytes, bytearray)):
            try:
                with io.BytesIO(image) as bio:
                    img = Image.open(bio).convert("RGB")
                    # 检查图片尺寸，如果高度或宽度小于28则跳过
                    width, height = img.size
                    if height < 28 or width < 28:
                        print(f"[SKIP] 图片 {sample_id} 尺寸过小: {width}x{height}, 跳过处理")
                        return {"image": None}
                    
                    save_path = os.path.join(args.thinklite_image_out_dir, f"{sample_id}.jpg")
                    if not os.path.exists(save_path):
                        img.save(save_path, format="JPEG")
                    return {"image": save_path}
            except Exception as e:
                print(f"[ERROR] 处理图片 {sample_id} 时出错: {e}")
                return {"image": None}
        else:
            return {"image": None}

    # 先把 image 二进制转成路径（并覆盖为字符串列）
    thinklite_ds = thinklite_ds.map(
        _thinklite_image_to_path,
        load_from_cache_file=False,
        num_proc=8,
    )
    
    # 过滤掉image为None的样本（尺寸过小的图片）
    def has_valid_image(example):
        return example.get("image") is not None
    
    thinklite_ds = thinklite_ds.filter(has_valid_image, num_proc=8)

    def make_thinklite_map_fn(split):
        data_source = "russwang/ThinkLite-VL-70k"

        def process_fn(example, idx):
            problem = example.get("problem") or ""
            answer = example.get("answer") or ""
            ground_truth = example.get("ground_truth", answer) or ""
            image = example.get("image")
            choices = example.get("choices")
            sample_id = example.get("id")
            sample_id = str(int(sample_id)) if isinstance(sample_id, (int, float)) else str(sample_id)
            
            if isinstance(image, bytes):
                images = [image.decode("utf-8")]
            elif isinstance(image, str):
                images = [image]
            elif isinstance(image, os.PathLike):
                images = [str(image)]
            else:
                images = []

            if len(images) == 0:
                print(f"image: {image}")
                print(f"sample_id: {sample_id}, problem: {problem}, answer: {answer}, ground_truth: {ground_truth}, choices: {choices}")
            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": problem}],
                "images": images,  # 统一 List[str]
                "ability": "mix",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": sample_id,  # 统一字符串
                    "answer": answer,
                    "question": problem,
                    "choices": choices,  # 允许为 None
                },
            }

        return process_fn

    thinklite_mapped = thinklite_ds.map(function=make_thinklite_map_fn("train"), with_indices=True, num_proc=8)

    # ------------------------------------------
    # 仅保留公共列 + 强制统一 Features（仅 ThinkLite）
    # ------------------------------------------
    COMMON_COLS = ["data_source", "prompt", "images", "ability", "reward_model", "extra_info"]

    def _select_common(ds):
        keep = [c for c in COMMON_COLS if c in ds.column_names]
        return ds.select_columns(keep)

    thinklite_mapped = _select_common(thinklite_mapped)

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

    thinklite_mapped = thinklite_mapped.cast(unified_features)

    # ------------------------------------------
    # 4) 写 parquet（仅 ThinkLite）
    # ------------------------------------------
    mixed = thinklite_mapped

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    output_path = os.path.join(local_dir, args.output_filename)

    mixed.to_parquet(output_path)

    # 额外导出用于快速验证的 2.2parquet（随机抽样 10 条）
    try:
        sample_size = min(10, len(mixed))
        if sample_size > 0:
            test_output_path = os.path.join(local_dir, "test.parquet")
            mixed.shuffle(seed=42).select(range(sample_size)).to_parquet(test_output_path)
    except Exception as e:
        print(f"[WARN] 生成 test.parquet 失败: {e}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print(f"[OK] Wrote parquet to: {output_path}")
    if hdfs_dir:
        print(f"[OK] Copied to HDFS dir: {hdfs_dir}")


if __name__ == "__main__":
    main()
