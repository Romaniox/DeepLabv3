ALL_CLASSES = ['background', 'truck', 'road']

LABEL_COLORS_LIST = [
    (0, 0, 0),  # Background.
    (127, 127, 127),  # Track.
    (255, 255, 255),  # Road.
]

dataset_name = '180923_roads&track'

IMG_SIZE = 640
EPOCHS = 100
LR = 0.00005
BATCH = 2

MULTI_GPU_MODE = True
