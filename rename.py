import os

# 指定目录路径
directory = "/home/jinxulin/focal_calibration-main/model/resnet50-cifar10-bsce/epoch"

# 遍历指定目录
for filename in os.listdir(directory):
    if filename.startswith("resnet50_focal_loss_") :
        # 构建完整的旧文件路径
        old_file = os.path.join(directory, filename)
        
        # 生成新文件名
        new_filename = filename.replace("focal_loss", "bsce")
        
        # 构建完整的新文件路径
        new_file = os.path.join(directory, new_filename)
        
        # 重命名文件
        os.rename(old_file, new_file)
        
        # 打印信息确认文件已被重命名
        print(f"Renamed '{old_file}' to '{new_file}'")