import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

############################################
# Step 1: 定义评价指标函数
############################################
def compute_all_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = target.astype(np.uint8)
    
    acc = accuracy_score(target_bin.flatten(), pred_bin.flatten())
    f1 = f1_score(target_bin.flatten(), pred_bin.flatten(), zero_division=0)
    precision = precision_score(target_bin.flatten(), pred_bin.flatten(), zero_division=0)
    recall = recall_score(target_bin.flatten(), pred_bin.flatten(), zero_division=0)
    
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

############################################
# Step 2: 定义 U-Net 模型（与旧模型相同）
############################################
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

############################################
# Step 3: 定义数据集拆分函数（按手ID分组）
############################################
def split_data_by_hand_id(csv_path, image_dir, mask_dir, test_size=0.2, val_size=0.1):
    """按照手ID分组，确保相同手ID的图片都在同一个集合中"""
    # 先获取所有有掩码的图片文件
    mask_files = os.listdir(mask_dir)
    mask_basenames = [os.path.splitext(f)[0] for f in mask_files if f.endswith('.png')]
    valid_images = [f"{name}.jpg" for name in mask_basenames 
                   if os.path.exists(os.path.join(image_dir, f"{name}.jpg"))]
    
    print(f"找到同时有图像和掩码的图片数量: {len(valid_images)}")
    
    # 读取CSV文件，获取图片的手ID信息
    df = pd.read_csv(csv_path)
    dorsal_hands = df[df["aspectOfHand"].str.contains("dorsal", case=False)]
    annotated_df = dorsal_hands[dorsal_hands["imageName"].isin(valid_images)].copy()
    
    # 如果有valid_images中的图片在CSV中没找到，输出警告
    if len(annotated_df) < len(valid_images):
        missing_count = len(valid_images) - len(annotated_df)
        print(f"警告：有{missing_count}张图片在CSV中没有找到对应信息")
    
    # 创建手ID字段
    annotated_df['hand_id'] = annotated_df.apply(
        lambda row: f"{row['id']}_{row['age']}_{row['gender']}_{row['skinColor']}", 
        axis=1
    )
    
    # 按hand_id分组
    unique_hands = annotated_df['hand_id'].unique()
    print(f"数据集中独特的手ID数量: {len(unique_hands)}")
    
    # 按手ID拆分训练、验证和测试集
    train_val_hands, test_hands = train_test_split(unique_hands, test_size=test_size, random_state=42)
    train_hands, val_hands = train_test_split(train_val_hands, test_size=val_size/(1-test_size), random_state=42)
    
    # 获取对应的图片列表
    train_images = annotated_df[annotated_df['hand_id'].isin(train_hands)]['imageName'].tolist()
    val_images = annotated_df[annotated_df['hand_id'].isin(val_hands)]['imageName'].tolist()
    test_images = annotated_df[annotated_df['hand_id'].isin(test_hands)]['imageName'].tolist()
    
    print(f"训练集: {len(train_images)}张图片, {len(train_hands)}个手ID")
    print(f"验证集: {len(val_images)}张图片, {len(val_hands)}个手ID")
    print(f"测试集: {len(test_images)}张图片, {len(test_hands)}个手ID")
    
    return train_images, val_images, test_images

############################################
# Step 4: 定义数据集类
############################################
class NailDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list=None):
        """
        参数:
            image_dir: 图片目录
            mask_dir: 掩码目录
            image_list: 如果提供，则只使用列表中的图片
        """
        if image_list:
            self.image_paths = [os.path.join(image_dir, img_name) for img_name in image_list]
            self.mask_paths = [os.path.join(mask_dir, os.path.splitext(img_name)[0] + '.png') 
                              for img_name in image_list]
        else:
            # 获取所有图片和掩码
            self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
            self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # 调整为固定尺寸
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # 转换格式
        image = np.transpose(image, (2, 0, 1))  # (3, 256, 256)
        mask = np.expand_dims(mask, axis=0)      # (1, 256, 256)
        
        return torch.tensor(image), torch.tensor(mask)

############################################
# Step 5: 训练和评估函数
############################################
def train_and_evaluate_model(image_dir, mask_dir, csv_path, num_epochs=15, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 按手ID分割数据
    train_images, val_images, test_images = split_data_by_hand_id(csv_path, image_dir, mask_dir)
    
    # 创建数据集和数据加载器
    train_dataset = NailDataset(image_dir, mask_dir, train_images)
    val_dataset = NailDataset(image_dir, mask_dir, val_images)
    test_dataset = NailDataset(image_dir, mask_dir, test_images)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # 初始化模型、损失函数和优化器
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss = train_loss / len(train_dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                
        val_loss = val_loss / len(val_dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "nail_segmentation_unet_new.pth")
            print(f"模型已保存 (验证损失: {val_loss:.4f})")
    
    print("训练完成!")
    
    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()
    
    # 加载最佳模型进行评估
    model.load_state_dict(torch.load("nail_segmentation_unet_new.pth"))
    model.eval()
    
    # 计算测试集指标
    all_metrics = {
        "Accuracy": [],
        "F1": [],
        "Precision": [],
        "Recall": [],
        "IoU": [],
        "Dice": []
    }
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks_np = masks.squeeze().cpu().numpy()
            
            outputs = model(images)
            pred_np = outputs.squeeze().cpu().numpy()
            
            metrics = compute_all_metrics(pred_np, masks_np)
            
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    # 计算和输出平均指标
    print("\n测试集评估指标:")
    for key in all_metrics:
        avg_value = np.mean(all_metrics[key])
        print(f"{key}: {avg_value:.4f}")
    
    return model, all_metrics

############################################
# Step 6: 主程序
############################################
if __name__ == "__main__":
    # 设置路径
    image_dir = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/dorsal_hand_images"
    mask_dir = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/masks_11k_100"
    csv_path = "/Users/wentibaobao/Desktop/HandInfo.csv"
    
    # 训练和评估模型
    model, metrics = train_and_evaluate_model(image_dir, mask_dir, csv_path, num_epochs=15, batch_size=4)
    
    # 保存评估结果
    with open("evaluation_results.txt", "w") as f:
        f.write("测试集评估指标:\n")
        for key, values in metrics.items():
            avg_value = np.mean(values)
            f.write(f"{key}: {avg_value:.4f}\n")
            
# 测试集评估指标:
# Accuracy: 0.9927
# F1: 0.5609
# Precision: 0.8869
# Recall: 0.4769
# IoU: 0.4298
# Dice: 0.5609