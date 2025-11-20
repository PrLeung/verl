import pandas as pd
import os
from PIL import Image

# 路径配置
parquet_path = "/vlm/peirouliang/verl_new/data/mix_100k/test.parquet"
output_dir = "/vlm/peirouliang/output_images"
os.makedirs(output_dir, exist_ok=True)

# 读取 parquet 文件
df = pd.read_parquet(parquet_path)

# 遍历样本，根据 images 字段保存图片
for idx, row in df.iterrows():
    try:
        img_path = row['images'][0]  # 直接读取图片绝对路径
        print(img_path)
        img = Image.open(img_path)
        img.save(os.path.join(output_dir, f"img_{idx}.png"))
    except Exception as e:
        print(f"第{idx}行图片保存失败：{e}")