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
from sklearn.metrics import accuracy_score, f1_score

# 计算 Intersection over Union (IoU)
def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    iou = intersection / union if union != 0 else 0
    return iou

# 计算准确率和F1得分
def compute_metrics(pred, target):
    pred = (pred > 0.5).astype(np.uint8)  # 预测为1的地方
    target = target.astype(np.uint8)
    
    acc = accuracy_score(target.flatten(), pred.flatten())  # 计算准确率
    f1 = f1_score(target.flatten(), pred.flatten())  # 计算F1得分
    return acc, f1


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
image_dir = r"C:\Users\Yunshan.Cai\Desktop\Nail-Biting-Identification\pics"
mask_dir = r"C:\Users\Yunshan.Cai\Desktop\Nail-Biting-Identification\path_to_output_masks"

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

############################################
# Step 6: 测试与结果可视化
############################################
model.eval()
with torch.no_grad():
    # 选择多个样本进行测试
    num_samples = 5  # 显示5张测试样本
    for idx in range(num_samples):
        sample_image, sample_mask = dataset[idx]
        sample_image = sample_image.unsqueeze(0).to(device)  # (1, 3, 256, 256)
        pred_mask = model(sample_image).squeeze().cpu().numpy()  # 预测 mask
        
        # 原图与 Ground Truth mask
        sample_image_np = np.transpose(sample_image.squeeze().cpu().numpy(), (1, 2, 0))  # 变为 (256, 256, 3)
        sample_mask_np = sample_mask.squeeze().cpu().numpy()  # 变为 (256, 256)
        
        # 计算评估指标
        acc, f1 = compute_metrics(pred_mask, sample_mask_np)
        iou = compute_iou(pred_mask, sample_mask_np)

        # 显示结果
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(sample_image_np)
        plt.title(f"Original Image {idx+1}")
        
        plt.subplot(1, 3, 2)
        plt.imshow(sample_mask_np, cmap="gray")
        plt.title("Ground Truth Mask")
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Predicted Mask")
        
        plt.show()
        
        # 打印评估结果
        print(f"Sample {idx+1}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"IoU: {iou:.4f}")
        print("-" * 50)