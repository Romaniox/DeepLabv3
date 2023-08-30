import os
import glob
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def tif2png(src_path, dst_path):
    for tif_file in glob.glob(os.path.join(src_path, '*.tif')):
        png_file = os.path.join(dst_path, os.path.basename(tif_file).replace('.tif', '.png'))
        img = cv2.imread(tif_file, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        cv2.imwrite(png_file, img)


def train_val_split(src_path, dst_path):
    all_files = glob.glob(os.path.join(src_path, 'images', '*.png'))
    train, val = train_test_split(all_files, test_size=0.1)
    for file in val:
        shutil.move(file, os.path.join(dst_path, 'val', 'images'))
        shutil.move(file.replace('images', 'masks'), os.path.join(dst_path, 'val', 'masks'))


def cut_mask(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img[0:5000, 0:5000]
    cv2.imwrite(img_path, img)


def correct_mask(imgs_path):
    for img_path in glob.glob(os.path.join(imgs_path, '*.jpg')):
        img_path_new = img_path.replace('.jpg', '.png')
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(img_path_new, img)
        os.remove(img_path)


def change_size(imgs_path, size=(640, 640)):
    # walk through all files in the directory using os.walk
    for root, dirs, files in os.walk(imgs_path):
        for file in files:
            # read the image using cv2
            img = cv2.imread(os.path.join(root, file))
            # resize the image to desired size
            img = cv2.resize(img, size)
            # save the image with the same name, but different extension
            cv2.imwrite(os.path.join(root, file), img)
            print(os.path.join(root, file))


def main():
    # src_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\fullsize\train'
    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\fullsize'
    # train_val_split(src_path, dst_path)

    # src_path = r'D:\SKZ\GEO_AI\datasets\aerials\init_'
    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\fullsize\test'
    # tif2png(src_path, dst_path)

    # img_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\fullsize\val\masks\Sample AI_transparent_mosaic_group1_2_2.jpg'
    # cut_mask(img_path)

    # src_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\fullsize\val\masks'
    # correct_mask(src_path)

    src_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\halfsize'
    change_size(src_path)


if __name__ == '__main__':
    main()
