import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


def _to_builtin(obj):
    """将 numpy/pandas 对象递归转换为原生 Python 类型。"""
    if obj is None:
        return None
    # numpy 标量
    if isinstance(obj, np.generic):
        return obj.item()
    # numpy 数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # pandas 时间戳/时间间隔
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    # pandas NA
    if obj is pd.NA:
        return None
    # 字典
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    # 列表/元组
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def print_random_sample(parquet_path: str, seed: int | None = None) -> None:
    """读取 Parquet 文件并随机输出一个样本（JSON 格式）。"""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {parquet_path}")

    df = pd.read_parquet(path)
    if len(df) == 0:
        print("文件中没有数据行。")
        return

    sample_record = df.sample(n=1, random_state=seed).to_dict(orient="records")[0]
    sample_record = _to_builtin(sample_record)
    print(json.dumps(sample_record, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 Parquet 并输出一个样本")
    parser.add_argument(
        "--path",
        type=str,
        default="/vlm/peirouliang/verl/data/mix_llava_cot_40k_llava_next_20k/train.parquet",
        help="Parquet 文件路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（可选，保证可复现）",
    )
    args = parser.parse_args()

    print_random_sample(args.path, args.seed)


if __name__ == "__main__":
    main()


