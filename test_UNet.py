import os
import glob
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_all_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = target.astype(np.uint8)
    
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



############################################
# Step 1: 定义一个简单的 U-Net 模型
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
        # 拼接对应 encoder 层输出 enc3
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
# Step 2: 定义数据集类（加载图片和对应的 Mask）
############################################
class NailDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        # 获取图片和 mask 的文件路径，注意确保排序一致
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):

        # 读取图像，保持彩色
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # 调整为固定尺寸，例如256x256
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        
        # 归一化到 [0,1]
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        image = np.transpose(image, (2, 0, 1))  # 转换为 (3, 256, 256)
        mask = np.expand_dims(mask, axis=0)  # 变为 (1, 256, 256)
        
        
        # 转换为 Tensor
        image = torch.tensor(image)
        mask = torch.tensor(mask)
        
        return image, mask

############################################
# Step 3: 数据加载与 DataLoader
############################################
# 请替换以下路径为你的图片和 mask 存放路径
image_dir = r"/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/pics"
mask_dir = r"/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/path_to_output_masks"

dataset = NailDataset(image_dir, mask_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

############################################
# Step 4: 模型、损失函数与优化器设置
############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCELoss()  # 二值交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

############################################
# Step 5: 训练模型
############################################
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")
torch.save(model.state_dict(), "nail_segmentation_unet.pth")
print("Model saved as 'nail_segmentation_unet.pth'")


############################################
# Step 6: 全体样本评估（Accuracy, F1, Precision, Recall, IoU, Dice）
############################################
model.eval()
all_metrics = {
    "Accuracy": [],
    "F1": [],
    "Precision": [],
    "Recall": [],
    "IoU": [],
    "Dice": []
}

with torch.no_grad():
    for idx in range(len(dataset)):
        image, mask = dataset[idx]
        image = image.unsqueeze(0).to(device)
        mask_np = mask.squeeze().cpu().numpy()

        pred_mask = model(image).squeeze().cpu().numpy()
        metrics = compute_all_metrics(pred_mask, mask_np)

        # 累积各项指标
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

# 计算平均值
print("Average Metrics on Full Dataset:")
for key in all_metrics:
    avg_value = np.mean(all_metrics[key])
    print(f"{key}: {avg_value:.4f}")



        
        
# 今天TO DO:
# 1.计算平均的f1,accuracy and iou
# Average Metrics on Full Dataset:
# Accuracy: 0.9225
# F1: 0.3593
# Precision: 0.4998
# Recall: 0.3668
# IoU: 0.2444
# Dice: 0.3593

# 2.看看怎么从网络上下载更多的照片
# 11k dataset, 5680 pics
# 先标注100张（contour, nail fold, distal border, free edge）
# 标注后可以做的事情：
# a) 把新标注的100张跟原来的一百张放一起用unet，看看边缘能不能更好地被预测
# b) 新标注的一百张可以根据nail bed长度（nail bed的长度感觉是一个更重要的标准），free edge是否rough来判断是否啃指甲，然后把这个结果跟标注对比
# c) 新标注的一百张可以预测nail fold, distal border, free edge （是否必要？）

# 重要：判断是否啃指甲的形态学差距（这个还可以再好好研究一下形态学的差距）
# nail contour: 不啃指甲的人nail contour更接近一个椭圆
# nail bed: 啃指甲的人nail bed显著更短
# distal order: 啃指甲的人指甲边缘更rough