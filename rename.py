# import os

# # 指定目录路径
# directory = "/home/jinxulin/focal_calibration-main/model/resnet50-cifar10-bsce/epoch"

# # 遍历指定目录
# for filename in os.listdir(directory):
#     if filename.startswith("resnet50_focal_loss_") :
#         # 构建完整的旧文件路径
#         old_file = os.path.join(directory, filename)
        
#         # 生成新文件名
#         new_filename = filename.replace("focal_loss", "bsce")
        
#         # 构建完整的新文件路径
#         new_file = os.path.join(directory, new_filename)
        
#         # 重命名文件
#         os.rename(old_file, new_file)
        
#         # 打印信息确认文件已被重命名
#         print(f"Renamed '{old_file}' to '{new_file}'")

import os

def rename_folders(base_path):
    # 遍历指定目录中的所有项
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        # 检查这个项是否是目录
        if os.path.isdir(item_path):
            # 分割旧的文件夹名，假设文件夹名称符合'model-dataset-loss'的格式
            parts = item.split('-')
            if len(parts) == 3:
                model, dataset, loss = parts
                # 构建新的文件夹名为'dataset-model-loss'
                new_folder_name = f"{dataset}-{model}-{loss}"
                new_folder_path = os.path.join(base_path, new_folder_name)
                # 重命名文件夹
                os.rename(item_path, new_folder_path)
                print(f"Renamed '{item}' to '{new_folder_name}'")
            else:
                print(f"Skipped '{item}', does not fit the expected pattern.")
        else:
            print(f"Skipped '{item}', it is not a directory.")

# 设置你的基本路径
base_path = "/home/jinxulin/UQ/model"
rename_folders(base_path)