import os
import shutil
import random

def move_half_images_to_validation(test_folder, validation_folder):
    """
    将测试集中一半的图像随机移动到验证集文件夹中。

    参数:
    - test_folder: 测试集文件夹的路径
    - validation_folder: 验证集文件夹的路径

    注意：此方法假设测试集和验证集文件夹已存在。
    """

    # 获取测试集中的所有图像文件
    test_images = [f for f in os.listdir(test_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # 计算要移动的图像数量 32
    num_images_to_move = 2000-32

    # 随机选择要移动的图像
    images_to_move = random.sample(test_images, num_images_to_move)

    # 移动图像到验证集文件夹
    for image in images_to_move:
        image_path = os.path.join(test_folder, image)
        destination_path = os.path.join(validation_folder, image)
        shutil.move(image_path, destination_path)

def count_files_in_folder(folder_path):

    # 使用 os.listdir 获取文件夹中所有文件和子文件夹的列表
    files_and_folders = os.listdir(folder_path)

    # 使用列表推导式过滤出文件的列表
    files = [f for f in files_and_folders if os.path.isfile(os.path.join(folder_path, f))]

    # 返回文件数量
    return len(files)


if __name__ == "__main__":
    # 设置测试集和验证集文件夹路径
    test_folder_path = r"/home/t2f/data/dataset/test/test_image_2000/"
    validation_folder_path = r"/home/t2f/data/dataset/test/val_image_2000/"
    os.makedirs(validation_folder_path, exist_ok=True)

    # 调用方法，将测试集中的一半图像随机移动到验证集
    move_half_images_to_validation(test_folder_path, validation_folder_path)

    files_count = count_files_in_folder(test_folder_path)

    if files_count is not None:
        print(f"文件夹 '{test_folder_path}' 中的文件数量为: {files_count}")
    files_count = count_files_in_folder(validation_folder_path)

    if files_count is not None:
        print(f"文件夹 '{validation_folder_path}' 中的文件数量为: {files_count}")

