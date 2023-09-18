import os
import glob
import shutil
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split
import cv_tools as cvt
from natsort import natsorted


def tif2png(src_path, dst_path):
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    cv2.imwrite(dst_path, img)


def tif2png_dir(src_path, dst_path):
    for tif_file in glob.glob(os.path.join(src_path, '*.tif')):
        png_file = os.path.join(dst_path, os.path.basename(tif_file).replace('.tif', '.png'))
        img = cv2.imread(tif_file, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        cv2.imwrite(png_file, img)


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


def merge_crops(src_path, dst_path, size=(10000, 10000)):
    dir_name = os.path.basename(src_path)

    res_img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    crop_list = os.listdir(src_path)
    crop = cv2.imread(os.path.join(src_path, crop_list[0]))
    h, w, _ = crop.shape
    rows, cols = size[0] // h, size[1] // w
    overlap_h = (h * (rows + 1) - size[0]) // rows
    overlap_w = (w * (cols + 1) - size[1]) // cols

    x, y = 0, 0
    count = 0
    for crop_name in crop_list:
        crop = cv2.imread(os.path.join(src_path, crop_name))
        h, w, _ = crop.shape
        if x + w > size[1]:
            crop = crop[:, 0:size[1] - x]
            h, w, _ = crop.shape

        if y + h > size[0]:
            crop = crop[0:size[0] - y, :, :]
            h, w, _ = crop.shape

        res_img[y:y + h, x:x + w, :] = crop
        x += w - overlap_w
        print(count)
        count += 1
        if count % (cols + 1) == 0:
            x = 0
            y += h - overlap_h

    cv2.imwrite(os.path.join(dst_path, f'{dir_name}.png'), res_img)


def get_tiles(src_path, dst_path, size=(640, 640)):
    img = cv2.imread(src_path)
    tiles = cvt.tiles_creation.get_tiles(img, size)

    for i, tile in enumerate(tiles):
        cv2.imwrite(os.path.join(dst_path, f'{i}.png'), tile)


# def img2squares(src_path, dst_path, size=(640, 640)):


def main():
    # src_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\crops\all'
    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\crops'
    # train_val_split(src_path, dst_path)

    # src_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\new\train\images'
    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\new\train\images'
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

    # src_path = r'D:\SKZ\GEO_AI\deeplabv3\outputs\runs\r50_31082023\inference_results\masks\3859'
    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\outputs\runs\r50_31082023\inference_results\masks'
    # merge_crops(src_path, dst_path, size=(10000, 10000))

    # src_path = r'D:\SKZ\GEO_AI\deeplabv3\road_classif\rgb_trans.jpg'
    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\road_classif\tiles'
    # get_tiles(src_path, dst_path, size=(6500, 6500))

    # masks_path = r'D:\SKZ\GEO_AI\deeplabv3\road_classif\masks'
    # rgb_path = r'D:\SKZ\GEO_AI\deeplabv3\road_classif\rgb_trans_mask.jpg'
    #
    # # rgb = cv2.imread(r'D:\SKZ\GEO_AI\deeplabv3\road_classif\rgb_trans.jpg')
    # # size = rgb.shape[:2]
    # size = (13000, 52000)
    # tiles = []
    # masks_list = natsorted(os.listdir(masks_path))
    # for mask_name in masks_list:
    #     mask = cv2.imread(os.path.join(masks_path, mask_name))
    #     mask = cv2.resize(mask, (6500, 6500))
    #     tiles.append(mask)
    #
    # rgb_trans_mask = cvt.tiles_creation.merge_tiles(tiles, size)
    #
    # cv2.imwrite(rgb_path, rgb_trans_mask)

    # img = cv2.imread(r'D:\SKZ\GEO_AI\deeplabv3\road_classif\rgb_trans.jpg')
    # size = img.shape[:2]
    #
    # mask = cv2.imread(r'D:\SKZ\GEO_AI\deeplabv3\road_classif\rgb_trans_mask.jpg')
    #
    # mask_unpad = cvt.image_transform.remove_padding(mask, size)
    # cv2.imwrite(r'D:\SKZ\GEO_AI\deeplabv3\road_classif\rgb_trans_mask_.jpg', mask_unpad)

    # mask = cv2.imread(r'D:\SKZ\GEO_AI\deeplabv3\road_classif\rgb_trans_mask.jpg')
    # mask_ = mask[:, :-1]
    # cv2.imwrite(r'D:\SKZ\GEO_AI\deeplabv3\road_classif\rgb_trans_mask.jpg', mask_)

    # img = cv2.imread(r'D:\SKZ\GEO_AI\deeplabv3\road_classif\rgb_trans.jpg')
    # mask = cv2.imread(r'D:\SKZ\GEO_AI\deeplabv3\road_classif\rgb_trans_mask.jpg')
    #
    # road = cvt.draw.cut_mask(img, mask)
    # cv2.imwrite(r'D:\SKZ\GEO_AI\deeplabv3\road_classif\road.jpg', road)

    # mask = cvt.coco_functions.get_masks(r'D:\SKZ\GEO_AI\datasets\aerials\annot\instances_default_True.json', ['trees_field', 'roads'], img_name='True_1259.tif')
    # cv2.imwrite(r'D:\SKZ\GEO_AI\deeplabv3\dataset_roads_aerial\all\masks_roads_treesfields\True_1259.png', mask)

    # imgs_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset_roads_aerial\all\masks_roads_treesfields'

    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset_roads_aerial\other\images\3860.png'
    # img_path = r'D:\SKZ\GEO_AI\datasets\aerials\init\3860.tif'
    # tif2png(img_path, dst_path)

    # imgs_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset_roads_aerial\other\images_ful'
    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset_roads_aerial\other\images'

    # for img_name in os.listdir(imgs_path):
    #     img_path = os.path.join(imgs_path, img_name)
    #     img = cv2.imread(img_path)
    #     tiles = cvt.tiles_creation.get_tiles(img, (1280, 1280), 0)
    #     for i, tile in enumerate(tiles):
    #         cv2.imwrite(os.path.join(dst_path, img_name[:-4]+f'_{i}.png'), tile)

    # imgs_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset_roads_aerial\other\images'
    # masks_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset_roads_aerial\other\masks'
    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset_roads_aerial\other\cuts'
    #
    # for img_name in os.listdir(imgs_path):
    #     img = cv2.imread(os.path.join(imgs_path, img_name))
    #     mask = cv2.imread(os.path.join(masks_path, img_name))
    #     res = cvt.draw.cut_mask(img, mask)
    #
    #     cv2.imwrite(os.path.join(dst_path, img_name), res)

    # img = cv2.imread(r'D:\SKZ\GEO_AI\deeplabv3\dataset\new\train\mask\3859.png')
    # dst_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset\new\crops\mask'
    # tiles = cvt.tiles_creation.get_tiles(img, (1280, 1280), 256)
    #
    # for i, tile in enumerate(tiles):
    #     cv2.imwrite(os.path.join(dst_path, f'{i}.png'), tile)

    # create masks from COCO
    imgs_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset_roads_aerial\all\images'
    annots_path = r'D:\SKZ\GEO_AI\datasets\aerials\annotation_all'
    masks_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset_roads_aerial\all\masks_roads_tracks'
    save_path = r'D:\SKZ\GEO_AI\deeplabv3\dataset_roads_aerial\180923_roads&track'

    os.makedirs(masks_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    for img_name in os.listdir(imgs_path):
        print(img_name)
        # if img_name == '3859.png':
        #     continue

        annot_path = os.path.join(annots_path, img_name[:-4] + '.json')

        mask = cvt.coco_functions.get_masks(annot_path, ['tracks', 'roads'], img_name=img_name)
        cv2.imwrite(os.path.join(masks_path, img_name[:-4] + '.png'), mask)

    tiles_imgs = []
    tiles_masks = []

    for img_name in os.listdir(imgs_path):
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path)
        tiles = cvt.tiles_creation.get_tiles(img, (3000, 3000), 1000)
        tiles_imgs.extend(tiles)

        mask_path = os.path.join(masks_path, img_name[:-4] + '.png')
        mask = cv2.imread(mask_path)
        tiles = cvt.tiles_creation.get_tiles(mask, (3000, 3000), 1000)
        tiles_masks.extend(tiles)

    indexes = np.arange(len(tiles_imgs))

    train_idx, test_idx = train_test_split(indexes, test_size=0.1, random_state=42)

    train_imgs = [tiles_imgs[i] for i in train_idx]
    train_masks = [tiles_masks[i] for i in train_idx]

    test_imgs = [tiles_imgs[i] for i in test_idx]
    test_masks = [tiles_masks[i] for i in test_idx]

    train_save_path = os.path.join(save_path, 'train')
    test_save_path = os.path.join(save_path, 'val')

    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)

    images_save_path = os.path.join(train_save_path, 'images')
    masks_save_path = os.path.join(train_save_path, 'masks')

    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(masks_save_path, exist_ok=True)

    for i, (tile_img, tile_mask) in enumerate(zip(train_imgs, train_masks)):
        cv2.imwrite(os.path.join(images_save_path, f'{i}.png'), tile_img)
        cv2.imwrite(os.path.join(masks_save_path, f'{i}.png'), tile_mask)

    images_save_path = os.path.join(test_save_path, 'images')
    masks_save_path = os.path.join(test_save_path, 'masks')

    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(masks_save_path, exist_ok=True)

    for i, (tile_img, tile_mask) in enumerate(zip(test_imgs, test_masks)):
        cv2.imwrite(os.path.join(images_save_path, f'{i}.png'), tile_img)
        cv2.imwrite(os.path.join(masks_save_path, f'{i}.png'), tile_mask)


if __name__ == '__main__':
    main()
