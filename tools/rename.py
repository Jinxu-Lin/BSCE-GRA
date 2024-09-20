import os
import re

# 指定目录路径
directory = '/home/jinxulin/UQ/model/cifar10-resnet50-bsce_gra/1/epoch'

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 检查文件是否以.model结尾
    if filename.endswith('.model'):
        # 使用正则表达式匹配文件名中的模式
        match = re.search(r'(.*_norm_)(15__1__1)(_.*.model)', filename)
        
        if match:
            # 构造新的文件名
            new_filename = match.group(1) + '1' + match.group(3)
            
            # 构造完整的文件路径
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            
            # 重命名文件
            os.rename(old_path, new_path)
            print(f'已重命名: {filename} -> {new_filename}')

print('重命名过程完成')