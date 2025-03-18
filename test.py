import cv2
import numpy as np

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
path_base = "/Users/wentibaobao/Desktop/graduation thesis/nail biting code/pics/";
image_path = path_base+"2.png"  # 替换为你的指甲图像路径
img, thresh = preprocess_image(image_path)
nail_contour = extract_nail_contour(thresh)
if nail_contour is not None:
    result_img = analyze_nail(nail_contour, img)
    while True:
        cv2.imshow("Result", result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
            break
    cv2.destroyAllWindows()
else:
    print("No nail contour found.")