import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_logits_curves(csv_file_path):
    """
    读取CSV文件，计算相同step的logit平均值，并绘制曲线图
    
    Args:
        csv_file_path: CSV文件路径
    """
    # 读取CSV文件
    print("正在读取CSV文件...")
    df = pd.read_csv(csv_file_path)
    
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"Step范围: {df['step'].min()} - {df['step'].max()}")
    
    # 按step分组，计算每个step的平均logit值
    print("正在计算每个step的平均logit值...")
    grouped = df.groupby('step').agg({
        '6576logit': 'mean',
        '91logit': 'mean'
    }).reset_index()
    
    print(f"分组后的数据形状: {grouped.shape}")
    print(f"前5行数据:")
    print(grouped.head())
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制两条曲线
    plt.plot(grouped['step'], grouped['6576logit'], 
             label='6576logit (Average)', linewidth=2, marker='o', markersize=3)
    plt.plot(grouped['step'], grouped['91logit'], 
             label='91logit (Average)', linewidth=2, marker='s', markersize=3)
    
    # 设置图形属性
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Logit Value', fontsize=12)
    plt.title('Logit Values vs Step', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴
    plt.xlim(grouped['step'].min(), grouped['step'].max())
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = '/vlm/peirouliang/verl/logits_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {output_path}")
    
    # 显示图片
    plt.show()
    
    # 打印一些统计信息
    print("\n统计信息:")
    print(f"6576logit - 最小值: {grouped['6576logit'].min():.4f}, 最大值: {grouped['6576logit'].max():.4f}")
    print(f"91logit - 最小值: {grouped['91logit'].min():.4f}, 最大值: {grouped['91logit'].max():.4f}")
    
    return grouped

if __name__ == "__main__":
    # CSV文件路径
    csv_file = "/vlm/peirouliang/verl/logits_origin.csv"
    
    # 执行绘图
    result_df = plot_logits_curves(csv_file)
    
    print("\n处理完成！")
