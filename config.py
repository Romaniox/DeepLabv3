ALL_CLASSES = ['background', 'road']

LABEL_COLORS_LIST = [
    (0, 0, 0),  # Background.
    (255, 255, 255),  # Road.
]

dataset_name = 'dataset_roads_aerial/dataset_270923_220343'
MODEL = 'deeplabv3_resnet50'
IMG_SIZE = 640
EPOCHS = 100
LR = 0.00005
BATCH = 2

MULTI_GPU_MODE = True
