import os
import re

# 指定目录路径
directory = '/home/jinxulin/UQ/model/cifar10/resnet50/cifar10-resnet50-temperature_focal_loss_gra/1/epoch'

# 遍历目录中的文件
for filename in os.listdir(directory):
    # 检查文件名是否符合模式
    if re.match(r'.*_gra_4.0_.*\.model$', filename):
        # 构造新的文件名
        new_filename = re.sub(r'_gra_', '_gra_gamma_', filename)
        # 获取完整路径
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # 重命名文件
        os.rename(old_file, new_file)