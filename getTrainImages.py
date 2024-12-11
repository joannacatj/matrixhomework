import os
import shutil
import pandas as pd

# 获取训练数据集
base_path = r"/home/t2f/data/dataset/image/image_256/image/"
save_path = r"/home/t2f/data/dataset/test/test_image/"
os.makedirs(save_path, exist_ok = True)

train_txt=r"/home/t2f/data/dataset/test/filenames.pickle"
data=pd.read_pickle(train_txt)
count=0
for item in data:
    count+=1
    image_source_path = os.path.join(base_path, '{}.png'.format(item))
    image_save_path = os.path.join(save_path, '{}.png'.format(item))
    shutil.copy(image_source_path,image_save_path)
print(count)


