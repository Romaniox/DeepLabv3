import os
import glob
import shutil
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split


def add_padding(image: np.ndarray, tile_size: int, overlap: int = 0) -> np.ndarray:
    h, w = image.shape[:2]

    h_pad = 0
    w_pad = 0
    if h % (tile_size - overlap) != 0:
        h_pad = (tile_size - overlap) - (h % (tile_size - overlap))

    if w % (tile_size - overlap) != 0:
        w_pad = (tile_size - overlap) - (w % (tile_size - overlap))

    transform = A.Compose([
        A.PadIfNeeded(min_height=h + h_pad, min_width=w + w_pad, border_mode=cv2.BORDER_CONSTANT, value=0)
    ])

    image = transform(image=image)['image']

    return image


def tif2png(src_path, dst_path):
    for tif_file in glob.glob(os.path.join(src_path, '*.tif')):
        png_file = os.path.join(dst_path, os.path.basename(tif_file).replace('.tif', '.png'))
        img = cv2.imread(tif_file, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        cv2.imwrite(png_file, img)


# def train_val_split(src_path, dst_path):
#     all_files = glob.glob(os.path.join(src_path, 'images', '*.png'))
#     train, val = train_test_split(all_files, test_size=0.1)
#     for file in val:
#         shutil.move(file, os.path.join(dst_path, 'val', 'images'))
#         shutil.move(file.replace('images', 'masks'), os.path.join(dst_path, 'val', 'masks'))

def train_val_split(src_path, dst_path):
    all_files = glob.glob(os.path.join(src_path, 'images', '*'))
    train, val = train_test_split(all_files, test_size=0.1)
    train_save_path = os.path.join(dst_path, 'train')
    val_save_path = os.path.join(dst_path, 'val')
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(val_save_path, exist_ok=True)
    os.makedirs(os.path.join(train_save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_save_path, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(val_save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_save_path, 'masks'), exist_ok=True)

    for file in train:
        shutil.copy(file, os.path.join(train_save_path, 'images'))
        shutil.copy(file.replace('images', 'masks'), os.path.join(train_save_path, 'masks'))
    for file in val:
        shutil.copy(file, os.path.join(val_save_path, 'images'))
        shutil.copy(file.replace('images', 'masks'), os.path.join(val_save_path, 'masks'))


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


def create_crops(src_path, dst_path, size=(640, 640)):
    for root, dirs, files in os.walk(src_path):
        for file in files:
            subdir = os.path.relpath(root, src_path)

            save_path = os.path.join(dst_path, subdir)
            os.makedirs(save_path, exist_ok=True)

            img = cv2.imread(os.path.join(root, file))
            h, w, _ = img.shape
            rows, cols = h // size[0], w // size[1]
            overlap_h = (size[0] * (rows + 1) - h) // rows
            overlap_w = (size[1] * (cols + 1) - w) // cols

            x, y = 0, 0
            for i in range(rows + 1):
                for j in range(cols + 1):
                    crop = img[y:y + size[0], x:x + size[1]]
                    crop = add_padding(crop, size[0])
                    x += size[0] - overlap_w
                    cv2.imwrite(os.path.join(save_path, file.replace('.png', f'_{i}_{j}.png')), crop)
                x = 0
                y += size[1] - overlap_h
            print(os.path.join(root, file))


def main():
    src_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\crops\all'
    dst_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\crops'
    train_val_split(src_path, dst_path)

    # src_path = r'D:\SKZ\GEO_AI\datasets\aerials\init_'
    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\fullsize\test'
    # tif2png(src_path, dst_path)

    # img_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\fullsize\val\masks\Sample AI_transparent_mosaic_group1_2_2.jpg'
    # cut_mask(img_path)

    # src_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\fullsize\val\masks'
    # correct_mask(src_path)

    # src_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\halfsize'
    # change_size(src_path)

    # src_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\fullsize'
    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\crops'
    # create_crops(src_path, dst_path, size=(1280, 1280))


if __name__ == '__main__':
    main()
