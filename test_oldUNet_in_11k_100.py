# 这个文件是检验用网络照片训练的旧模型在新标注的100 11k上的表现

# 旧模型在新数据上的表现(按手ID分组):
# Accuracy: 0.9892
# F1: 0.0000
# Precision: 0.0000
# Recall: 0.0000
# IoU: 0.0000
# Dice: 0.0000
# 这种零指标的情况表明模型完全无法在新数据上泛化，这是典型的域迁移问题。后面再看吧。

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 直接定义compute_all_metrics函数
def compute_all_metrics(pred, target, threshold=0.5):
    
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = target.astype(np.uint8)
    
    # 只在调用sklearn函数时添加zero_division参数
    f1 = f1_score(target_bin.flatten(), pred_bin.flatten(), zero_division=0)
    precision = precision_score(target_bin.flatten(), pred_bin.flatten(), zero_division=0)
    recall = recall_score(target_bin.flatten(), pred_bin.flatten(), zero_division=0)
    
    acc = accuracy_score(target_bin.flatten(), pred_bin.flatten())
    f1 = f1_score(target_bin.flatten(), pred_bin.flatten())
    precision = precision_score(target_bin.flatten(), pred_bin.flatten())
    recall = recall_score(target_bin.flatten(), pred_bin.flatten())
    
    # IoU
    intersection = np.sum(pred_bin * target_bin)
    union = np.sum(pred_bin) + np.sum(target_bin) - intersection
    iou = intersection / union if union != 0 else 0

    # Dice coefficient
    dice = 2 * intersection / (np.sum(pred_bin) + np.sum(target_bin)) if (np.sum(pred_bin) + np.sum(target_bin)) != 0 else 0

    return {
        "Accuracy": acc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "IoU": iou,
        "Dice": dice
    }
    
# 直接定义UNet类
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 编码器部分（Contracting Path）
        self.enc1 = self.contracting_block(3, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 解码器部分（Expansive Path）
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.expansive_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.expansive_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.expansive_block(128, 64)
        
        # 最后输出层
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
    
    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder
        dec4 = self.upconv4(enc4)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)
        
        out = self.final_conv(dec2)
        out = torch.sigmoid(out)
        return out

# 按手ID分组数据
def group_data_by_hand_id(csv_path, image_dir, mask_dir):
    # 先获取有掩码的图片列表
    mask_files = os.listdir(mask_dir)
    mask_basenames = [os.path.splitext(f)[0] for f in mask_files if f.endswith('.png')]
    valid_images = [f"{name}.jpg" for name in mask_basenames 
                   if os.path.exists(os.path.join(image_dir, f"{name}.jpg"))]
    
    print(f"找到同时有图片和掩码的图片数量: {len(valid_images)}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 只保留dorsal视图的图片
    dorsal_hands = df[df["aspectOfHand"].str.contains("dorsal", case=False)]
    
    # 获取已标注图片列表
    # annotated_files = [f for f in os.listdir(annotated_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # 筛选出已标注的图片信息
    annotated_df = dorsal_hands[dorsal_hands["imageName"].isin(valid_images)].copy()
    
    # 创建手ID
    annotated_df['hand_id'] = annotated_df.apply(
        lambda row: f"{row['id']}_{row['age']}_{row['gender']}_{row['skinColor']}", 
        axis=1
    )
    
    # 按hand_id分组并划分数据集
    unique_hands = annotated_df['hand_id'].unique()
    hand_train, hand_test = train_test_split(unique_hands, test_size=0.2, random_state=42)
    
    # 获取测试集图片
    test_images = annotated_df[annotated_df['hand_id'].isin(hand_test)]['imageName'].tolist()
    
    return test_images

# 评估旧模型在新数据上的表现(考虑手ID)
def evaluate_old_model_on_new_data_by_id(model_path, csv_path, image_dir, mask_dir):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 按手ID获取测试图片
    test_images = group_data_by_hand_id(csv_path, image_dir, mask_dir)
    print(f"按手ID分组后的测试图片数量: {len(test_images)}")
    
    # 评估指标
    all_metrics = {
        "Accuracy": [], "F1": [], "Precision": [],
        "Recall": [], "IoU": [], "Dice": []
    }
    
    # 评估模型
    with torch.no_grad():
        for img_name in test_images:
            # 构建图像和掩码路径
            img_path = os.path.join(image_dir, img_name)
            mask_name = os.path.splitext(img_name)[0] + '.png'
            mask_path = os.path.join(mask_dir, mask_name)
            
            # 检查文件是否存在
            if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                print(f"跳过: {img_name} (文件不存在)")
                continue
            
            # 读取和预处理图像
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"跳过: {img_name} (无法读取图像或掩码)")
                continue
                
            # 调整尺寸
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            
            # 归一化
            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0
            
            # 转换格式
            image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.tensor(image).unsqueeze(0).to(device)
            
            # 模型预测
            pred = model(image_tensor).squeeze().cpu().numpy()
            
            # 计算指标
            metrics = compute_all_metrics(pred, mask)
            
            # 累积指标
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    # 计算并输出平均指标
    print("\n旧模型在新数据上的表现(按手ID分组):")
    for key in all_metrics:
        avg_value = np.mean(all_metrics[key])
        print(f"{key}: {avg_value:.4f}")
    
    return all_metrics

# 使用示例
if __name__ == "__main__":
    model_path = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/nail_segmentation_unet.pth"
    csv_path = "/Users/wentibaobao/Desktop/HandInfo.csv"
    image_dir = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/dorsal_hand_images"
    mask_dir = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/masks_11k_100"
    
    metrics = evaluate_old_model_on_new_data_by_id(model_path, csv_path, image_dir, mask_dir)