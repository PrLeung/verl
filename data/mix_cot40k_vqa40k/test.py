import pandas as pd
import os

def sample_data_by_source():
    """
    从train.parquet文件中读取数据，从每个data_source中抽取5条样本，
    并保存为test.parquet文件
    """
    # 读取parquet文件
    input_file = "/vlm/peirouliang/verl/data/mix_cot40k_vqa40k/train.parquet"
    output_file = "./test.parquet"
    
    print(f"正在读取文件: {input_file}")
    df = pd.read_parquet(input_file)
    
    print(f"原始数据总行数: {len(df)}")
    print(f"数据列名: {list(df.columns)}")
    
    # 检查是否有data_source列
    if 'data_source' not in df.columns:
        print("错误: 数据中没有找到'data_source'列")
        return
    
    # 查看不同的data_source
    unique_sources = df['data_source'].unique()
    print(f"发现的数据源: {unique_sources}")
    print(f"每个数据源的样本数量:")
    for source in unique_sources:
        count = len(df[df['data_source'] == source])
        print(f"  {source}: {count} 条样本")
    
    # 从每个data_source中抽取5条样本
    sampled_data = []
    for source in unique_sources:
        source_data = df[df['data_source'] == source]
        # 如果该数据源的样本数少于5条，则全部抽取
        sample_size = min(5, len(source_data))
        sampled = source_data.sample(n=sample_size, random_state=42)
        sampled_data.append(sampled)
        print(f"从 {source} 中抽取了 {sample_size} 条样本")
    
    # 合并所有抽取的样本
    final_df = pd.concat(sampled_data, ignore_index=True)
    
    print(f"总共抽取了 {len(final_df)} 条样本")
    
    # 保存为parquet文件
    final_df.to_parquet(output_file, index=False)
    print(f"数据已保存到: {output_file}")
    
    # 验证保存的文件
    verify_df = pd.read_parquet(output_file)
    print(f"验证: 保存的文件包含 {len(verify_df)} 条样本")
    print("各数据源的样本分布:")
    for source in verify_df['data_source'].unique():
        count = len(verify_df[verify_df['data_source'] == source])
        print(f"  {source}: {count} 条样本")

if __name__ == "__main__":
    sample_data_by_source()
