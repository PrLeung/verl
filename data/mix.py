#!/usr/bin/env python3
import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two Parquet files and split into train/test."
    )
    parser.add_argument(
        "--in1",
        type=str,
        default="/vlm/peirouliang/verl/data/llava_next_20k/train.parquet",
        help="Path to the first input Parquet file.",
    )
    parser.add_argument(
        "--in2",
        type=str,
        default="/vlm/peirouliang/verl/data/llava_cot_40k/train.parquet",
        help="Path to the second input Parquet file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/vlm/peirouliang/verl/data/mix_llava_cot_40k_llava_next_20k_new_format",
        help="Output directory to write merged parquet files.",
    )
    parser.add_argument(
        "--train_file_name",
        type=str,
        default="train.parquet",
        help="Output file name for train parquet (default: train.parquet).",
    )
    parser.add_argument(
        "--test_file_name",
        type=str,
        default="test.parquet",
        help="Output file name for test parquet (default: test.parquet).",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=5,
        help="Number of samples to draw from each dataset for test set (default: 5).",
    )
    return parser.parse_args()


def read_parquet(path: str):
    try:
        import pandas as pd
        return pd.read_parquet(path), "pandas"
    except Exception as pandas_error:
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(path)
            df = table.to_pandas()
            return df, "pyarrow"
        except Exception as pyarrow_error:
            raise RuntimeError(
                f"Failed to read parquet with pandas ({pandas_error}) and pyarrow ({pyarrow_error})."
            ) from pyarrow_error


def write_parquet(df, output_path: str) -> None:
    try:
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df.to_parquet(output_path, index=False)
        return
    except Exception:
        pass

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if hasattr(df, "to_dict"):
            df = pa.Table.from_pandas(df)
        elif not isinstance(df, pa.Table):
            df = pa.table(df)
        pq.write_table(df, output_path)
    except Exception as error:
        raise RuntimeError(f"Failed to write parquet to {output_path}: {error}") from error


def main() -> None:
    args = parse_args()

    in1 = args.in1
    in2 = args.in2
    out_dir = args.out_dir
    train_file_name = args.train_file_name
    test_file_name = args.test_file_name
    n_samples = args.test_samples

    if not os.path.isfile(in1):
        print(f"Input file not found: {in1}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(in2):
        print(f"Input file not found: {in2}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    df1, reader1 = read_parquet(in1)
    df2, reader2 = read_parquet(in2)

    import pandas as pd

    # 随机采样各自5条数据作为 test
    test1 = df1.sample(n=min(n_samples, len(df1)), random_state=42)
    test2 = df2.sample(n=min(n_samples, len(df2)), random_state=42)
    test_df = pd.concat([test1, test2], ignore_index=True)

    # 剩下的作为 train
    train1 = df1.drop(test1.index)
    train2 = df2.drop(test2.index)
    train_df = pd.concat([train1, train2], ignore_index=True)

    # 写出 parquet
    train_output_path = os.path.join(out_dir, train_file_name)
    test_output_path = os.path.join(out_dir, test_file_name)
    write_parquet(train_df, train_output_path)
    write_parquet(test_df, test_output_path)

    print(
        "Split successfully",
        f"in1_reader={reader1}",
        f"in2_reader={reader2}",
        f"train_rows={len(train_df)}",
        f"test_rows={len(test_df)}",
        f"train_output={train_output_path}",
        f"test_output={test_output_path}",
    )


if __name__ == "__main__":
    main()
