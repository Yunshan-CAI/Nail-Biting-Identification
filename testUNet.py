import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
from PIL import Image

def check_masks(mask_base_dir):
    """检查各类掩码的有效性"""
    categories = ["nail_contour", "nail_fold", "distal_border"]
    
    for category in categories:
        folder = os.path.join(mask_base_dir, category)
        if not os.path.exists(folder):
            print(f"警告：{folder} 文件夹不存在")
            continue
            
        mask_files = [f for f in os.listdir(folder) if f.endswith('.png')]
        non_empty_count = 0
        total_pixels = 0
        avg_size = 0
        
        for mask_file in mask_files:
            mask_path = os.path.join(folder, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"错误：无法读取 {mask_path}")
                continue
                
            # 计算非零像素
            non_zero = np.count_nonzero(mask)
            if non_zero > 0:
                non_empty_count += 1
                total_pixels += non_zero
                
                # 获取掩码尺寸信息
                avg_size += mask.shape[0] * mask.shape[1]
        
        if non_empty_count > 0:
            print(f"{category}: 总文件 {len(mask_files)}, 非空文件 {non_empty_count}, " 
                  f"平均非零像素 {total_pixels/non_empty_count:.1f}, "
                  f"平均占比 {(100*total_pixels)/(avg_size):.2f}%")
        else:
            print(f"{category}: 总文件 {len(mask_files)}, 非空文件 0")
            
        # 可视化前5个掩码
        visualize_sample_masks(folder, category, 5)

def visualize_sample_masks(folder, category, num_samples=5):
    """可视化指定类别的样本掩码"""
    mask_files = [f for f in os.listdir(folder) if f.endswith('.png')]
    if not mask_files:
        print(f"文件夹 {folder} 中没有掩码文件")
        return
        
    # 只显示指定数量的样本
    samples = mask_files[:min(num_samples, len(mask_files))]
    
    plt.figure(figsize=(15, 3))
    for i, mask_file in enumerate(samples):
        mask_path = os.path.join(folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        plt.subplot(1, len(samples), i+1)
        plt.imshow(mask, cmap='gray')
        plt.title(f"{category}\n{os.path.basename(mask_file)}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{category}_samples.png")
    plt.close()
    print(f"保存了 {category} 的样本可视化到 {category}_samples.png")

def visualize_predictions(model_path, image_dir, mask_base_dir, num_samples=5, output_dir="prediction_viz"):
    """可视化模型在样本上的预测结果"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型 - 简化的方法
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # 尝试直接加载整个模型
        model = torch.load(model_path, map_location=device)
        model.eval()
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        # 如果失败，给出建议
        print(f"无法加载模型: {e}")
        print("请确保模型保存时使用了torch.save(model, path)而不是torch.save(model.state_dict(), path)")
        return
    
    # 获取图像文件列表
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
    
    if not image_files:
        print(f"警告：在 {image_dir} 中没有找到图像文件")
        return
        
    # 只处理指定数量的样本
    samples = image_files[:min(num_samples, len(image_files))]
    
    for idx, img_path in enumerate(samples):
        try:
            # 加载图像
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"无法读取图像: {img_path}")
                continue
                
            image_display = image.copy()
            
            # 预处理
            image = cv2.resize(image, (256, 256))
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image_tensor = torch.tensor(image).unsqueeze(0).to(device)
            
            # 预测
            with torch.no_grad():
                outputs = model(image_tensor)
                preds = outputs.squeeze().cpu().numpy()
            
            # 确保preds是3D数组（多类别情况）
            if len(preds.shape) == 2:  # 如果只有一个通道
                preds = np.expand_dims(preds, axis=0)
            
            # 加载真实掩码
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            categories = ["nail_contour", "nail_fold", "distal_border"]
            masks = []
            
            for category in categories:
                mask_path = os.path.join(mask_base_dir, category, base_name + '.png')
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask, (256, 256))
                else:
                    mask = np.zeros((256, 256), dtype=np.uint8)
                masks.append(mask)
            
            # 创建对比图
            plt.figure(figsize=(15, 10))
            
            # 显示原图
            plt.subplot(3, 3, 1)
            plt.imshow(cv2.cvtColor(cv2.resize(image_display, (256, 256)), cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')
            
            # 显示每个类别的真实掩码和预测
            for i, category in enumerate(categories):
                if i < len(preds):
                    # 真实掩码
                    plt.subplot(3, 3, i*3 + 2)
                    plt.imshow(masks[i], cmap='gray')
                    plt.title(f"True {category}")
                    plt.axis('off')
                    
                    # 预测掩码
                    plt.subplot(3, 3, i*3 + 3)
                    pred_mask = (preds[i] > 0.5).astype(np.uint8) * 255
                    plt.imshow(pred_mask, cmap='gray')
                    plt.title(f"Pred {category}\nMax: {preds[i].max():.2f}")
                    plt.axis('off')
                    
                    # 打印预测统计
                    print(f"{base_name} - {category}: 最大值={preds[i].max():.4f}, 最小值={preds[i].min():.4f}, 平均值={preds[i].mean():.4f}")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pred_{idx}_{base_name}.png"))
            plt.close()
            
            print(f"已处理 {idx+1}/{len(samples)}: {base_name}")
            
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")
    
    print(f"可视化结果已保存到 {output_dir}")

if __name__ == "__main__":
    # 设置路径
    mask_base_dir = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/masks_multi"
    image_dir = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/dorsal_hand_images"
    model_path = "/Users/wentibaobao/Desktop/graduation thesis/Nail-Biting-Identification/multi_nail_segmentation_unet.pth"
    
    # 1. 检查掩码内容
    print("开始检查掩码内容...")
    check_masks(mask_base_dir)
    
    # 2. 可视化预测结果
    print("\n开始可视化预测结果...")
    visualize_predictions(model_path, image_dir, mask_base_dir, num_samples=5)
    
    print("\n检查和可视化完成!")