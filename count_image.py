import os

def count_files_in_folder(folder_path):
    files_and_folders = os.listdir(folder_path)

    files = [f for f in files_and_folders if os.path.isfile(os.path.join(folder_path, f))]

    return len(files)

if __name__ == "__main__":

    folder_path = r"/home/t2f/StableSR/results/pretrained_faceres/samples/"

    files_count = count_files_in_folder(folder_path)

    if files_count is not None:
        print(f"'{folder_path}' : {files_count}")