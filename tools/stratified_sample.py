#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # try common container key
        for key in ("data", "items", "samples"):
            if key in data and isinstance(data[key], list):
                return data[key]  # type: ignore[return-value]
    raise ValueError("输入JSON格式不支持：需要list，或dict中包含[data|items|samples]列表")


def stratified_sample(
    records: List[Dict[str, Any]],
    label_key: str,
    fraction: float,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        label = rec.get(label_key)
        if label is None:
            label = "__MISSING__"
        buckets[str(label)].append(rec)

    sampled: List[Dict[str, Any]] = []
    for label, bucket in buckets.items():
        n = len(bucket)
        k = int(n * fraction)
        k = max(0, min(n, k))
        idxs = list(range(n))
        rng.shuffle(idxs)
        take = set(idxs[:k])
        sampled.extend(bucket[i] for i in take)
    return sampled


def compute_counts(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for rec in records:
        value = rec.get(key)
        if value is None:
            value = "__MISSING__"
        counts[str(value)] += 1
    # 转为普通dict
    return dict(counts)


def compute_nested_counts(
    records: List[Dict[str, Any]], parent_key: str, child_key: str
) -> Dict[str, Dict[str, int]]:
    nested: Dict[str, Dict[str, int]] = {}
    for rec in records:
        parent = rec.get(parent_key)
        child = rec.get(child_key)
        parent = "__MISSING__" if parent is None else str(parent)
        child = "__MISSING__" if child is None else str(child)
        if parent not in nested:
            nested[parent] = {}
        nested[parent][child] = nested[parent].get(child, 0) + 1
    return nested


def main() -> None:
    parser = argparse.ArgumentParser(description="按条件过滤后，基于分层键进行抽样")
    parser.add_argument("--input", default="/vlm/data/mix_think_no_50k_think_41k.json", help="输入JSON路径")
    parser.add_argument("--output", required=False, help="输出JSON路径，默认自动生成")
    parser.add_argument("--label-key", default="dataset", help="分层标签字段（对其进行分层抽样），默认=dataset")
    parser.add_argument("--fraction", type=float, default=0.5, help="抽样比例，默认0.5")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认42")
    parser.add_argument("--sample-if-key", default="data_source", help="仅当该键的值等于sample-if-value时才抽样，默认=data_source")
    parser.add_argument("--sample-if-value", default="think_no", help="触发抽样的键值，默认=think_no")
    args = parser.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"找不到输入文件: {in_path}")

    out_path = args.output
    if not out_path:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}.sampled{ext or '.json'}"

    records = read_json(in_path)
    # 将数据分为：需要抽样的子集（满足过滤条件），与无需抽样的子集
    to_sample: List[Dict[str, Any]] = []
    to_keep: List[Dict[str, Any]] = []
    for rec in records:
        if str(rec.get(args.sample_if_key)) == str(args.sample_if_value):
            to_sample.append(rec)
        else:
            to_keep.append(rec)

    sampled_part = stratified_sample(to_sample, args.label_key, args.fraction, args.seed)
    sampled = sampled_part + to_keep

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    # 统计分布：按 data_source 和 dataset
    stats = {
        "total_input": len(records),
        "total_sampled": len(sampled),
        # 顶层：各 data_source 数量
        "by_data_source": compute_counts(sampled, "data_source"),
        # 各 data_source 下的 dataset 分布
        "by_data_source_then_dataset": compute_nested_counts(sampled, "data_source", "dataset"),
        # 直接汇总：各 dataset 数量（便于快速查看）
        "by_dataset": compute_counts(sampled, "dataset"),
    }

    stats_path = os.path.splitext(out_path)[0] + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"输入样本数: {len(records)}")
    print(f"输出样本数: {len(sampled)}")
    print(f"已写出: {out_path}")
    print(f"分布统计已写出: {stats_path}")


if __name__ == "__main__":
    main()


