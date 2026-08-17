"""Static facts about PASTIS that both the server and the browser need.

Nothing here is learned or computed -- it is the fixed vocabulary of the dataset:
what the 20 class ids mean, what colour each one is drawn in, which of the 10
Sentinel-2 bands is which, and how big a single pixel is on the ground.
"""

# Class id -> crop name. Index in this list IS the integer the model predicts.
# Id 19 ("Void") marks pixels the dataset authors could not label; it is excluded
# from every score, which is what --ignore_index does during training.
CLASS_NAMES = [
    "Background",
    "Meadow",
    "Soft winter wheat",
    "Corn",
    "Winter barley",
    "Winter rapeseed",
    "Spring barley",
    "Sunflower",
    "Grapevine",
    "Beet",
    "Winter triticale",
    "Winter durum wheat",
    "Fruits, vegetables, flowers",
    "Potatoes",
    "Leguminous fodder",
    "Soybeans",
    "Orchard",
    "Mixed cereal",
    "Sorghum",
    "Void",
]

# Same order as CLASS_NAMES. Kept identical to the palette the old report used so
# that anything already screenshotted stays comparable.
CLASS_COLORS = [
    "#f0efec",
    "#1baf7a",
    "#2a78d6",
    "#eda100",
    "#9085e9",
    "#e87ba4",
    "#86b6ef",
    "#eb6834",
    "#4a3aa7",
    "#d55181",
    "#008300",
    "#0d366b",
    "#e34948",
    "#c98500",
    "#199e70",
    "#3987e5",
    "#184f95",
    "#d95926",
    "#6da7ec",
    "#ffffff",
]

VOID_CLASS = 19
NUM_CLASSES = 20

# The 10 Sentinel-2 bands PASTIS keeps, in the order they sit in the tensor.
# The satellite records 13; the three atmospheric ones (B1 aerosol, B9 water
# vapour, B10 cirrus) are dropped because they describe the air, not the ground.
BANDS = [
    {"idx": 0, "name": "B2", "label": "Blue", "nm": 490},
    {"idx": 1, "name": "B3", "label": "Green", "nm": 560},
    {"idx": 2, "name": "B4", "label": "Red", "nm": 665},
    {"idx": 3, "name": "B5", "label": "Red edge 1", "nm": 705},
    {"idx": 4, "name": "B6", "label": "Red edge 2", "nm": 740},
    {"idx": 5, "name": "B7", "label": "Red edge 3", "nm": 783},
    {"idx": 6, "name": "B8", "label": "NIR", "nm": 842},
    {"idx": 7, "name": "B8A", "label": "Narrow NIR", "nm": 865},
    {"idx": 8, "name": "B11", "label": "SWIR 1", "nm": 1610},
    {"idx": 9, "name": "B12", "label": "SWIR 2", "nm": 2190},
]

RED_BAND = 2  # B4
GREEN_BAND = 1  # B3
BLUE_BAND = 0  # B2
NIR_BAND = 6  # B8

# A patch is 128 x 128 pixels at 10 m ground resolution.
PIXEL_SIZE_M = 10.0
PATCH_SIZE_PX = 128
PIXEL_AREA_HA = (PIXEL_SIZE_M * PIXEL_SIZE_M) / 10_000.0  # 0.01 ha per pixel
PATCH_AREA_HA = PIXEL_AREA_HA * PATCH_SIZE_PX * PATCH_SIZE_PX  # 163.84 ha
