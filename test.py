import cv2
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing the data
def preprocess_image(image_path):
    # 读取图像并调整大小
    img = cv2.imread(image_path)
    img = cv2.resize(img, (500, 500))
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 二值化
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    return img, thresh

#转成HSV通过更细微的颜色差别提取指甲范围
def hsv_segmentation(image_path):
    # 读取图像并调整大小
    img = cv2.imread(image_path)
    img = cv2.resize(img, (500, 500))
    
    # 将图像从 BGR 转换到 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义指甲可能的颜色范围（这里需要根据你图像实际情况调节）
    # 下面的范围是一个初步的估计，可能需要调节 lower_bound 和 upper_bound
    lower_bound = np.array([0, 30, 80])
    upper_bound = np.array([30, 200, 255])
    
    # 使用 inRange() 生成掩码：符合范围的像素设为 255，否则设为 0
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # 显示原图、HSV 图像以及掩码结果
    cv2.imshow("Original Image", img)
    cv2.imshow("HSV Image", hsv)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#尝试：转成灰度图，看指甲与其他部分亮度是否比较不一致
def step1_display_gray(image_path):
    # 1. 读取图像，并调整为统一大小（500x500）
    img = cv2.imread(image_path)
    img = cv2.resize(img, (500, 500))
    
    # 2. 将彩色图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 显示灰度图，观察指甲区域亮度
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 4. 显示灰度图的直方图，帮助分析亮度分布情况
    plt.hist(gray.ravel(), bins=256, range=(0, 256))
    plt.title("Histogram of Grayscale Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
    
    return img, gray

#查找所有轮廓
def show_all_contours(thresh, img):
    # 查找所有轮廓
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 拷贝原图用于显示轮廓
    img_contours = img.copy() 
    
    # 将所有轮廓绘制在图像上
    for cnt in contours:
        cv2.drawContours(img_contours, [cnt], -1, (0, 255, 0), 2)
    
    # 显示绘制轮廓后的图像
    cv2.imshow("All Contours", img_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Detect nail peripheral
def extract_nail_contour(thresh):
    # 查找轮廓
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # 选取最大的轮廓，假设指甲区域面积最大
    nail_contour = max(contours, key=cv2.contourArea)
    return nail_contour

def analyze_nail(nail_contour, img):
    # 绘制轮廓供观察
    cv2.drawContours(img, [nail_contour], -1, (0, 255, 0), 2)
    
    # 计算轮廓的凸包与凸缺陷来判断边缘参差情况
    hull = cv2.convexHull(nail_contour, returnPoints=False)
    if hull is not None and len(hull) > 3:
        defects = cv2.convexityDefects(nail_contour, hull)
        if defects is not None:
            # 计算所有凸缺陷的平均深度
            defect_depths = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                defect_depths.append(d)
            avg_defect_depth = np.mean(defect_depths)
            print("Average defect depth:", avg_defect_depth)
    else:
        print("No convexity defects found")
    
    # 这里可以增加更多特征，如甲床长度的测量
    return img

# 主程序
image_path = "C:\\Users\\Yunshan.Cai\\Desktop\\Nail-Biting-Identification\\pics\\7.png"  

hsv_segmentation(image_path)
'''
img, gray = step1_display_gray(image_path)
img, thresh = preprocess_image(image_path)
show_all_contours(thresh, img)
nail_contour = extract_nail_contour(thresh)
if nail_contour is not None:
    result_img = analyze_nail(nail_contour, img)
    while True:
        cv2.imshow("Result", result_img)
        if cv2.waitKey(1) & 0xFF == ord('5'):  # 按 'q' 退出
            break
    cv2.destroyAllWindows()
else:
    print("No nail contour found.")
'''

'''
今天尝试了的操作：
目标是找指甲的轮廓
1. 找了所有的轮廓（用了最初的image processing function），发现找到的轮廓都是深色的区域
2. 都转成灰度图，发现指甲的亮度跟其他部分的亮度差别不大
3. 转成HSV

To Do List:
在网上找数据并且进行标注（可能需要标注指甲边缘以及指甲区域）
'''
