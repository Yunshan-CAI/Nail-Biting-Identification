import os
# 指定文件夹路径
folder_path = "C:\\Users\\Yunshan.Cai\\Desktop\\Nail-Biting-Identification\\pics_annotated"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.startswith("屏幕截图"):
        new_filename = filename.replace("屏幕截图 ", "", 1).strip()
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_filename}")

print("重命名完成！")
