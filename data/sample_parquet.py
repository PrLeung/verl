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


def sample_parquet(parquet_path: Path, num_samples: int, seed: int) -> pd.DataFrame:
    """Read a Parquet file and return a random sample of rows.

    Falls back to returning all rows if the file has fewer than requested.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"读取 {parquet_path}: 总共 {len(df)} 条数据")
    
    if len(df) == 0:
        return df

    if num_samples >= len(df):
        # 如果数据少于需要的数量，直接返回全部（并重置索引）
        print(f"数据量不足，返回全部 {len(df)} 条数据")
        return df.sample(n=len(df), random_state=seed).reset_index(drop=True)

    print(f"从 {len(df)} 条数据中抽样 {num_samples} 条")
    return df.sample(n=num_samples, random_state=seed).reset_index(drop=True)


def create_test_samples(df: pd.DataFrame, num_test: int, seed: int) -> pd.DataFrame:
    """从训练数据中抽取测试样本"""
    if len(df) <= num_test:
        print(f"数据量不足，返回全部 {len(df)} 条作为测试数据")
        return df.copy()
    
    print(f"从 {len(df)} 条数据中抽取 {num_test} 条作为测试数据")
    return df.sample(n=num_test, random_state=seed).reset_index(drop=True)


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """保存DataFrame为parquet文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"已保存 {len(df)} 条数据到: {output_path}")


def main() -> None:
    # 设置随机种子
    seed = 42
    np.random.seed(seed)
    
    # 输入文件路径
    vqa83k_path = Path("/vlm/peirouliang/verl/data/vqa83k_think/train.parquet")
    vqa100k_path = Path("/vlm/peirouliang/verl/data/vqa100k_think_no/train.parquet")
    
    # 输出目录
    output_dir_40k = Path("/vlm/peirouliang/verl/data/vqa40k_think")
    output_dir_20k = Path("/vlm/peirouliang/verl/data/vqa20k_think_no")
    
    print("开始数据抽样...")
    
    # 从vqa83k_think抽样40k数据
    print("\n=== 处理 vqa83k_think 数据 ===")
    df_40k = sample_parquet(vqa83k_path, 40000, seed)
    
    # 创建测试数据（5个样本）
    df_40k_test = create_test_samples(df_40k, 5, seed + 1)
    
    # 保存40k数据
    save_parquet(df_40k, output_dir_40k / "train.parquet")
    save_parquet(df_40k_test, output_dir_40k / "test.parquet")
    
    # 从vqa100k_think_no抽样20k数据
    print("\n=== 处理 vqa100k_think_no 数据 ===")
    df_20k = sample_parquet(vqa100k_path, 20000, seed)
    
    # 创建测试数据（5个样本）
    df_20k_test = create_test_samples(df_20k, 5, seed + 2)
    
    # 保存20k数据
    save_parquet(df_20k, output_dir_20k / "train.parquet")
    save_parquet(df_20k_test, output_dir_20k / "test.parquet")
    
    print("\n=== 抽样完成 ===")
    print(f"vqa40k_think: 训练数据 {len(df_40k)} 条，测试数据 {len(df_40k_test)} 条")
    print(f"vqa20k_think_no: 训练数据 {len(df_20k)} 条，测试数据 {len(df_20k_test)} 条")


if __name__ == "__main__":
    main()
