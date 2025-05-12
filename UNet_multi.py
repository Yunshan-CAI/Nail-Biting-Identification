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
# Step 1: 定义多类别评价指标函数
############################################
def compute_multi_metrics(preds, targets, threshold=0.5):
    """
    计算多类别分割的评估指标
    """
    # 类别名称列表
    class_names = ["nail_contour", "nail_fold", "distal_border"]
    metrics = {name: {} for name in class_names}
    
    # 为每个类别分别计算指标
    for i, class_name in enumerate(class_names):
        pred = preds[i]
        target = targets[i]
        
        pred_bin = (pred > threshold).astype(np.uint8)
        target_bin = (target > threshold).astype(np.uint8)
        
        # 计算基本指标
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
        
        metrics[class_name] = {
            "Accuracy": acc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            "IoU": iou,
            "Dice": dice
        }
    
    # 计算平均指标
    metrics["average"] = {
        metric: np.mean([metrics[class_name][metric] for class_name in class_names])
        for metric in ["Accuracy", "F1", "Precision", "Recall", "IoU", "Dice"]
    }
    
    return metrics

############################################
# Step 2: 修改 U-Net 模型以支持多类别输出
############################################
class MultiClassUNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiClassUNet, self).__init__()
        # 编码器部分（Contracting Path）- 与原始代码相同
        self.enc1 = self.contracting_block(3, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 解码器部分（Expansive Path）- 与原始代码相同
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.expansive_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.expansive_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.expansive_block(128, 64)
        
        # 最后输出层 - 修改为输出num_classes个通道
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def contracting_block(self, in_channels, out_channels):
        # 与原始代码相同
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
        # 与原始代码相同
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
        # Encoder - 与原始代码相同
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder - 与原始代码相同
        dec4 = self.upconv4(enc4)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2(dec2)
        
        # 输出多个通道，每个通道一个类别
        out = self.final_conv(dec2)
        out = torch.sigmoid(out)  # 每个通道独立的二元分割
        return out

############################################
# Step 3: 定义多类别数据集类
############################################
class MultiNailDataset(Dataset):
    def __init__(self, image_dir, mask_base_dir, image_list=None):
        """
        加载多类别的指甲分割数据
        
        参数:
            image_dir: 图片目录
            mask_base_dir: 掩码基本目录，下面应有nail_contour, nail_fold, distal_border子目录
            image_list: 如果提供，则只使用列表中的图片
        """
        if image_list:
            self.image_paths = [os.path.join(image_dir, img_name) for img_name in image_list]
            
            # 为三种不同的掩码创建路径
            self.contour_mask_paths = [os.path.join(mask_base_dir, "nail_contour", os.path.splitext(img_name)[0] + '.png') 
                                      for img_name in image_list]
            self.fold_mask_paths = [os.path.join(mask_base_dir, "nail_fold", os.path.splitext(img_name)[0] + '.png') 
                                   for img_name in image_list]
            self.border_mask_paths = [os.path.join(mask_base_dir, "distal_border", os.path.splitext(img_name)[0] + '.png') 
                                     for img_name in image_list]
        else:
            # 如果未提供图像列表，则获取所有图像
            self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
            
            # 为三种不同的掩码创建路径
            base_names = [os.path.splitext(os.path.basename(p))[0] for p in self.image_paths]
            self.contour_mask_paths = [os.path.join(mask_base_dir, "nail_contour", name + '.png') for name in base_names]
            self.fold_mask_paths = [os.path.join(mask_base_dir, "nail_fold", name + '.png') for name in base_names]
            self.border_mask_paths = [os.path.join(mask_base_dir, "distal_border", name + '.png') for name in base_names]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        if image is None:
            print(f"无法读取图像: {self.image_paths[idx]}")
            # 提供一个空图像作为替代
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # 读取三种掩码，如果文件不存在则使用空掩码
        contour_mask = self._read_mask_safe(self.contour_mask_paths[idx], image.shape[:2])
        fold_mask = self._read_mask_safe(self.fold_mask_paths[idx], image.shape[:2])
        border_mask = self._read_mask_safe(self.border_mask_paths[idx], image.shape[:2])
        
        # 调整为固定尺寸
        image = cv2.resize(image, (256, 256))
        contour_mask = cv2.resize(contour_mask, (256, 256))
        fold_mask = cv2.resize(fold_mask, (256, 256))
        border_mask = cv2.resize(border_mask, (256, 256))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        contour_mask = contour_mask.astype(np.float32) / 255.0
        fold_mask = fold_mask.astype(np.float32) / 255.0
        border_mask = border_mask.astype(np.float32) / 255.0
        
        # 合并掩码为一个3通道tensor
        masks = np.stack([contour_mask, fold_mask, border_mask], axis=0)  # (3, 256, 256)
        
        # 转换图像格式
        image = np.transpose(image, (2, 0, 1))  # (3, 256, 256)
        
        return torch.tensor(image), torch.tensor(masks)
    
    def _read_mask_safe(self, mask_path, shape):
        """安全地读取掩码，如果不存在则返回空掩码"""
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return np.zeros(shape, dtype=np.uint8)
            return mask
        else:
            return np.zeros(shape, dtype=np.uint8)

############################################
# Step 4: 训练和评估函数
############################################
def train_and_evaluate_model(image_dir, mask_base_dir, csv_path, num_epochs=15, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 按手ID分割数据
    train_images, val_images, test_images = split_data_by_hand_id(csv_path, image_dir, mask_base_dir)
    
    # 创建数据集和数据加载器
    train_dataset = MultiNailDataset(image_dir, mask_base_dir, train_images)
    val_dataset = MultiNailDataset(image_dir, mask_base_dir, val_images)
    test_dataset = MultiNailDataset(image_dir, mask_base_dir, test_images)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # 初始化模型、损失函数和优化器
    model = MultiClassUNet(num_classes=3).to(device)
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
            torch.save(model.state_dict(), "multi_nail_segmentation_unet.pth")
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
    plt.savefig('multi_loss_curves.png')
    plt.close()
    
    # 加载最佳模型进行评估
    model.load_state_dict(torch.load("multi_nail_segmentation_unet.pth"))
    model.eval()
    
    # 计算测试集指标
    class_metrics = {
        "nail_contour": {metric: [] for metric in ["Accuracy", "F1", "Precision", "Recall", "IoU", "Dice"]},
        "nail_fold": {metric: [] for metric in ["Accuracy", "F1", "Precision", "Recall", "IoU", "Dice"]},
        "distal_border": {metric: [] for metric in ["Accuracy", "F1", "Precision", "Recall", "IoU", "Dice"]},
        "average": {metric: [] for metric in ["Accuracy", "F1", "Precision", "Recall", "IoU", "Dice"]}
    }
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks_np = masks.squeeze().cpu().numpy()
            
            outputs = model(images)
            preds_np = outputs.squeeze().cpu().numpy()
            
            # 计算每个类别的指标
            metrics = compute_multi_metrics(preds_np, masks_np)
            
            # 累积各类别的指标
            for class_name in class_metrics.keys():
                for metric in class_metrics[class_name].keys():
                    class_metrics[class_name][metric].append(metrics[class_name][metric])
    
    # 计算和输出平均指标
    print("\n测试集评估指标:")
    for class_name, metrics_dict in class_metrics.items():
        print(f"\n{class_name}类别:")
        for metric, values in metrics_dict.items():
            avg_value = np.mean(values)
            print(f"{metric}: {avg_value:.4f}")
    
    return model, class_metrics

############################################
# Step 5: 辅助函数 - 按手ID分割数据
############################################
def split_data_by_hand_id(csv_path, image_dir, mask_base_dir, test_size=0.2, val_size=0.1):
    """按照手ID分组，确保相同手ID的图片都在同一个集合中"""
    # 先获取所有有掩码的图片文件 (检查至少有一种掩码存在)
    valid_images = []
    
    # 检查dorsal_hand_images目录中的所有jpg文件
    for img_name in os.listdir(image_dir):
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        base_name = os.path.splitext(img_name)[0]
        
        # 检查是否至少有一种掩码存在
        contour_path = os.path.join(mask_base_dir, "nail_contour", base_name + '.png')
        fold_path = os.path.join(mask_base_dir, "nail_fold", base_name + '.png')
        border_path = os.path.join(mask_base_dir, "distal_border", base_name + '.png')
        
        if os.path.exists(contour_path) or os.path.exists(fold_path) or os.path.exists(border_path):
            valid_images.append(img_name)
    
    print(f"找到同时有图像和至少一种掩码的图片数量: {len(valid_images)}")
    
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
# Step 6: 主程序
############################################
if __name__ == "__main__":
    # 设置路径
    image_dir = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/dorsal_hand_images"
    mask_base_dir = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/masks_multi"
    csv_path = "/Users/wentibaobao/Desktop/HandInfo.csv"
    
    # 训练和评估模型
    model, metrics = train_and_evaluate_model(image_dir, mask_base_dir, csv_path, num_epochs=15, batch_size=4)
    
    # 保存评估结果
    with open("multi_evaluation_results.txt", "w") as f:
        f.write("测试集评估指标:\n")
        for class_name, metrics_dict in metrics.items():
            f.write(f"\n{class_name}类别:\n")
            for metric, values in metrics_dict.items():
                avg_value = np.mean(values)
                f.write(f"{metric}: {avg_value:.4f}\n")
                
# 测试集评估指标:

# nail_contour类别:
# Accuracy: 0.9930
# F1: 0.3665
# Precision: 0.9339
# Recall: 0.2589
# IoU: 0.2448
# Dice: 0.3665

# nail_fold类别:
# Accuracy: 0.9977
# F1: 0.0000
# Precision: 0.0000
# Recall: 0.0000
# IoU: 0.0000
# Dice: 0.0000

# distal_border类别:
# Accuracy: 0.9972
# F1: 0.0000
# Precision: 0.0000
# Recall: 0.0000
# IoU: 0.0000
# Dice: 0.0000

# average类别:
# Accuracy: 0.9960
# F1: 0.1222
# Precision: 0.3113
# Recall: 0.0863
# IoU: 0.0816
# Dice: 0.1222