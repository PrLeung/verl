import argparse
import json
from pathlib import Path
from typing import Any, List

import pandas as pd
import numpy as np


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert objects (e.g., numpy types/arrays) to JSON-safe Python types."""
    # None, bool, int, float, str are already safe
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # Numpy arrays or objects with tolist()
    if hasattr(obj, "tolist"):
        try:
            return _make_json_safe(obj.tolist())
        except Exception:
            pass

    # Dict
    if isinstance(obj, dict):
        return {str(_make_json_safe(k)): _make_json_safe(v) for k, v in obj.items()}

    # List/Tuple/Set
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]

    # Fallback to string representation
    return str(obj)


def _extract_prompt(value: Any) -> str:
    """Normalize prompt field to a readable string.

    - If value is a list/array of dicts with 'content', return first content
    - If value is a dict with 'content', return it
    - If value is str, return as-is
    - Otherwise stringify
    """
    if isinstance(value, str):
        return value
    # Numpy array -> list
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
        return str(first)
    if isinstance(value, dict) and "content" in value:
        return str(value["content"])
    return str(value)


def _extract_images(value: Any) -> List[str]:
    """Normalize image paths to a list of strings.

    Accepts:
    - str: returns [str]
    - list/tuple/set/numpy array: converts each element to str
    - dict: tries keys like 'path' or 'image' -> str
    - others: stringifies and return as single-element list
    """
    if value is None:
        return []
    # numpy array -> list
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for k in ["path", "image", "file", "filepath", "file_path"]:
            if k in value and isinstance(value[k], str):
                return [value[k]]
        # Fallback: stringify dict
        return [str(value)]
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for v in value:
            if v is None:
                continue
            if hasattr(v, "tolist"):
                try:
                    v = v.tolist()
                except Exception:
                    pass
            if isinstance(v, str):
                result.append(v)
            elif isinstance(v, dict):
                found = None
                for k in ["path", "image", "file", "filepath", "file_path"]:
                    if k in v and isinstance(v[k], str):
                        found = v[k]
                        break
                result.append(found if found is not None else str(v))
            else:
                result.append(str(v))
        return result
    return [str(value)]


def sample_parquet(parquet_path: Path, num_samples: int, seed: int) -> pd.DataFrame:
    """Read a Parquet file and return a random sample of rows.

    Falls back to returning all rows if the file has fewer than requested.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    # 读取完成后打印第一个样本（若为空则提示空表）
    if len(df) > 0:
        record = df.iloc[0].to_dict()
        safe_record = _make_json_safe(record)
        print("读取 {} 的第一个样本: {}".format(parquet_path, json.dumps(safe_record, ensure_ascii=False)))
    else:
        print(f"读取 {parquet_path} 的第一个样本: 空表，无数据")
    if len(df) == 0:
        return df

    if num_samples >= len(df):
        # 如果数据少于需要的数量，直接返回全部（并重置索引）
        return df.sample(n=len(df), random_state=seed).reset_index(drop=True)

    return df.sample(n=num_samples, random_state=seed).reset_index(drop=True)


def save_as_json(records: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从两个 Parquet 各抽样指定数量的数据，合并并保存为 JSON 文件"
        )
    )
    parser.add_argument(
        "--cot-path",
        type=Path,
        default=Path("/vlm/peirouliang/verl/data/llava_cot_90k/train.parquet"),
        help="llava_cot_90k 的 train.parquet 路径",
    )
    parser.add_argument(
        "--next-path",
        type=Path,
        default=Path("/vlm/peirouliang/verl/data/llava_next_45k/train.parquet"),
        help="llava_next_45k 的 train.parquet 路径",
    )
    parser.add_argument(
        "--n1",
        type=int,
        default=100,
        help="从第一个数据集中抽样的数量（默认100）",
    )
    parser.add_argument(
        "--n2",
        type=int,
        default=100,
        help="从第二个数据集中抽样的数量（默认100）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认42）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/vlm/peirouliang/verl/data/sample_200.json"),
        help="输出 JSON 文件路径（默认 /vlm/peirouliang/verl/data/mixed_sample.json）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df1 = sample_parquet(args.cot_path, args.n1, args.seed)
    df1["data_source"] = "llava_cot_90k"

    df2 = sample_parquet(args.next_path, args.n2, args.seed)
    df2["data_source"] = "llava_next_45k"

    # 列不同也可合并；缺失列以 NaN 填充
    combined = pd.concat([df1, df2], ignore_index=True, sort=False)

    # 归一化字段并只保留需要的字段：prompt、answer、images、data_source
    # 回填 answer: 若缺失则尝试从 extra_info.answer 获取
    if "answer" not in combined.columns and "extra_info" in combined.columns:
        try:
            combined["answer"] = combined["extra_info"].apply(
                lambda x: _make_json_safe(x).get("answer") if isinstance(_make_json_safe(x), dict) else None
            )
        except Exception:
            pass

    # 将数组型/结构化 prompt 提取为字符串
    if "prompt" in combined.columns:
        combined["prompt"] = combined["prompt"].apply(_extract_prompt)
    else:
        for alt in ["input", "instruction", "question"]:
            if alt in combined.columns:
                combined["prompt"] = combined[alt].apply(_extract_prompt)
                break

    # 归一化 images 路径，支持别名
    if "images" in combined.columns:
        combined["images"] = combined["images"].apply(_extract_images)
    else:
        for alt in ["image", "image_path", "image_paths", "file_path", "filepath", "path"]:
            if alt in combined.columns:
                combined["images"] = combined[alt].apply(_extract_images)
                break

    wanted_cols = [c for c in ["prompt", "answer", "images", "data_source"] if c in combined.columns]
    combined = combined[wanted_cols] if wanted_cols else combined

    # 转为字典列表
    # 转为字典列表并 JSON 安全化
    records = [_make_json_safe(r) for r in combined.to_dict(orient="records")]

    save_as_json(records, args.output)
    print(
        f"已保存 {len(records)} 条样本到: {args.output}"
    )


if __name__ == "__main__":
    main()


