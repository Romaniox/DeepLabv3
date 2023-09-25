ALL_CLASSES = ['background', 'building']

LABEL_COLORS_LIST = [
    (0, 0, 0),  # Background.
    (255, 255, 255),  # building.
]

dataset_name = 'dataset_360/building/dataset_250923_212027'

MODEL = 'deeplabv3_resnet50'
IMG_SIZE = 640
EPOCHS = 100
LR = 0.00005
BATCH = 2

MULTI_GPU_MODE = True
