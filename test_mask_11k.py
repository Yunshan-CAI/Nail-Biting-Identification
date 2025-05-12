import json
import numpy as np
import cv2
import glob
import os

# 读取 labelme 标注 JSON，并转换为黑白 Mask
def json_to_mask(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    image_shape = (data["imageHeight"], data["imageWidth"])
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for shape in data["shapes"]:
        # 只处理nail_contour标签
        if shape["label"] == "nail_contour":
            points = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)  # 填充指甲区域为白色 (255)
    
    cv2.imwrite(output_path, mask)

# 处理所有 JSON 标注文件
json_files = glob.glob("/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/11k_annoted_100/*.json")
output_folder = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/masks_11k_100"

os.makedirs(output_folder, exist_ok=True)

for json_file in json_files:
    filename = os.path.splitext(os.path.basename(json_file))[0] + ".png"
    output_path = os.path.join(output_folder, filename)
    json_to_mask(json_file, output_path)

print("Mask 生成完成！")