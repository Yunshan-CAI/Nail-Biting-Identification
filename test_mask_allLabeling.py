import json
import numpy as np
import cv2
import glob
import os

# 读取 labelme 标注 JSON，并转换为对应的掩码
def json_to_masks(json_path, output_folder, base_filename):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    image_shape = (data["imageHeight"], data["imageWidth"])
    
    # 为每种标注类型创建单独的掩码
    masks = {
        "nail_contour": np.zeros(image_shape, dtype=np.uint8),
        "nail_fold": np.zeros(image_shape, dtype=np.uint8),
        "distal_border": np.zeros(image_shape, dtype=np.uint8)
    }
    
    # 标记是否找到每种类型
    found_labels = {label: False for label in masks.keys()}
    
    # 处理每种形状
    for shape in data["shapes"]:
        label = shape["label"]
        if label in masks:
            points = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(masks[label], [points], 255)  # 填充区域为白色 (255)
            found_labels[label] = True
    
    # 保存每种掩码到对应文件
    for label, mask in masks.items():
        if found_labels[label]:  # 只保存找到的标签
            # 创建标签对应的子文件夹
            label_folder = os.path.join(output_folder, label)
            os.makedirs(label_folder, exist_ok=True)
            
            # 保存掩码
            output_path = os.path.join(label_folder, base_filename)
            cv2.imwrite(output_path, mask)

# 主处理函数
def process_all_json_files():
    # JSON文件路径
    json_files = glob.glob("/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/11k_annoted_100/*.json")
    
    # 基本输出文件夹
    base_output_folder = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/masks_multi"
    os.makedirs(base_output_folder, exist_ok=True)
    
    # 处理每个JSON文件
    processed_counts = {
        "nail_contour": 0,
        "nail_fold": 0,
        "distal_border": 0
    }
    
    for json_file in json_files:
        base_filename = os.path.splitext(os.path.basename(json_file))[0] + ".png"
        json_to_masks(json_file, base_output_folder, base_filename)
    
    # 计算统计信息
    for label in processed_counts.keys():
        label_folder = os.path.join(base_output_folder, label)
        if os.path.exists(label_folder):
            processed_counts[label] = len(os.listdir(label_folder))
    
    # 打印处理结果
    print("掩码生成完成！")
    print(f"生成nail_contour掩码: {processed_counts['nail_contour']}个")
    print(f"生成nail_fold掩码: {processed_counts['nail_fold']}个")
    print(f"生成distal_border掩码: {processed_counts['distal_border']}个")

# 执行处理
if __name__ == "__main__":
    process_all_json_files()