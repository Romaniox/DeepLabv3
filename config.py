ALL_CLASSES = ['background', 'trees_field', 'road']

LABEL_COLORS_LIST = [
    (0, 0, 0),  # Background.
    (128, 128, 128),  # Trees.
    (255, 255, 255),  # Road.
]

dataset_name = '040923_roads&trees'

IMG_SIZE = 640
EPOCHS = 100
LR = 0.00005
BATCH = 4

MULTI_GPU_MODE = True
