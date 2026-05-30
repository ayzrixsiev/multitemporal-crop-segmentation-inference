# Pastis

Dataset contains 2,433 unique patches over French metropolitan territory. This dataset split into 5 folds/groups, for each of this fold we have pre-calculated Mean and Std for future scalling. Images are in the form of four dimensional spatio-temporal tensor, example: (43, 10, 128, 128), where 43 is time series counter, in other words - versions of the same territory throughout the year. In one of the sample images we observed over 100 crop fields. Each field in dataset is uniqely identified with 7 digit numbers, but for each fields within image we have dedicated sequential IDs for each field (e.g 0-119). The dataset contains both semantic (pixel level object detection) and instance annotations (object detection and separation from each other).

**DATA_S2** (2,468 files), this folder contains images.     

**ANNOTATIONS** (4,866 files, exactly 2 per image):     
- TARGET_*.npy: The actual Crop Types (Corn, Wheat, Meadow, etc. overall we have 0-19 crop types). This is our training target.
- ParcelIDs_*.npy: The permanent Government Land Registry IDs (7-digit numbers). Which are needed to label each field uniquely.

**INSTANCE_ANNOTATIONS** (7,299 files, exactly 3 per image):        
- INSTANCES_*.npy. Local field counter map (looks like multicolored block). It assigns each pixel in the same field one integer value.       
- HEATMAP_*.npy. A centerness map, pixels in the exact center of a fiels are bright 1.0, and closer they are to the border pixels fade down to dark 0.0. This is necessary for teaching a model to find "heart" of the fields
- ZONES_*.npy. Border tracking map, it categorizes pixels into three zones, 1 for inside the field, 2 exact border/edhe of the field and 0 is outside.

**NORM_S2_patch.json** contains pre-calculated channel statistics (Mean and Standard Deviation) to normilize images properly.