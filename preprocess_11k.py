import pandas as pd
import os
import shutil

# 读取CSV文件
df = pd.read_csv("/Users/wentibaobao/Desktop/HandInfo.csv")

# 查看aspectOfHand列的唯一值，了解可能的值
print("手部视角的唯一值:")
print(df["aspectOfHand"].unique())

# 筛选出aspectOfHand包含'dorsal'的行
dorsal_hands = df[df["aspectOfHand"].str.contains("dorsal", case=False)]

# 显示筛选后数据的数量
print(f"包含dorsal的图片数量: {len(dorsal_hands)}")

# 假设原始图像文件夹和目标文件夹
original_images_folder = "/Users/wentibaobao/Desktop/Hands"  # 替换为你的原始图像文件夹路径
target_folder = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/dorsal_hand_images"  # 目标文件夹

# 创建目标文件夹(如果不存在)
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 复制dorsal图像到目标文件夹
copied_count = 0
not_found_count = 0
for index, row in dorsal_hands.iterrows():
    image_name = row["imageName"]
    source_path = os.path.join(original_images_folder, image_name)
    target_path = os.path.join(target_folder, image_name)
    
    # 复制文件(如果存在)
    if os.path.exists(source_path):
        shutil.copy2(source_path, target_path)
        copied_count += 1
    else:
        not_found_count += 1
        print(f"未找到图像: {image_name}")
    
    # 每复制100张图片显示一次进度
    if copied_count % 100 == 0 and copied_count > 0:
        print(f"已复制 {copied_count} 张图片...")

# 生成一个新的CSV文件，只包含dorsal图像的信息
dorsal_csv_path = "dorsal_hands_info.csv"
dorsal_hands.to_csv(dorsal_csv_path, index=False)

print(f"\n处理完成!")
print(f"已复制 {copied_count} 张dorsal手部图像到 {target_folder} 文件夹")
print(f"未找到 {not_found_count} 张图像")
print(f"dorsal手部图像信息已保存到 {dorsal_csv_path}")

