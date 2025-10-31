# MLT_CAC classification
Standalone codebase for CAC classification from DICOM CXRs through multi-task learning

# Requirements
All libraries required for the code is listed in the requirements.txt

Check that these files are in the appropriate structure. The model weights can be downloaded here: [link](https://drive.google.com/drive/folders/1dxvBqHSiqLH14lSNH4akby8wthmVud0A?usp=sharing)
```
├── MACE_classification
│   ├── base
│   ├── options
│   ├── bone_supp
│   │   ├── network_intermediate_4.tar                # trained bonesuppression model
│   ├── clahe_clf
│   │   ├── best_model_config.pth.tar                 # trained CLAHE model
│   ├── joint_clf
│   │   ├── joint_clf.pth                             # trained joint model
│   ├── view_clf
│   │   ├── best.pth.tar                              # trained view model
│   ├── cac_mace_clf
│   │   ├── CAC_MACE_clf.pth                          # trained MTL CAC_MACE model
│   │   ├── normalized_hist_array_manual.pkl          # normalized array for cropped image
│   │   ├── normalized_bonesupp_array_manual.pkl      # normalized array for bone suppressed image
│   ├── lung_crop
│   │   ├── cxr_reg_weights.best.hdf5                 # trained lung cropping model
```

# CAC classification steps:
Both the training and inference code can run directly from DCMs as it has the preprocessing code in it. However, it is recommened to run preprocessing first and then run training or inference so that one can check the preprocessed images.

# Preprocessing
To run the inference code, simply run:

```python process.py --df_path (root directory to dataframe of dcm paths) --processed_dir (directory_to_save_processed_images_to)```

> **'df_path'** should be a dataframe with a minimum of one column **"path"** that has the list of dicom file locations (the full path) to try and preprocess. Additionally, the dataframe can have another column **"label"**  ([0, 1]) that can be used for training or inference later on.

> The preprocessing will output a dataframe **"processed_df.csv"** that should contain the following columns: [path, label (optional), view_pred, CLAHE_pred, cropped_path, heart_path, supp_path]

The code will run as follows:
1. Collect all DICOMs from the root directory.
2. Evaluate all DICOM CXRs as AP or lateral view images.
3. Evaluate all DICOM CXRs as CLAHE, inverted, or normal intensity images.
4. Remove all CXRs determined to be lateral/non-normal intensity images.
5. Center crop to the lungs and save the cropped image and the inverse lung mask.
6. Generate a bone suppressed image of the cropped image.
7. Run binary CAC classification on the processed images.
8. Finally save the output as a CSV file 'cac_inference.csv'

# Training

### Make train/val/test splits

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# load the processed dataframe
processed_df = pd.read_csv('processed_df.csv')

# split into train/val/test
train, val = train_test_split(processed_df, test_size=0.4)
val, test = train_test_split(val, test_size=0.5)
# reset indices for all dataframes
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)
# save the dataframes
train.to_csv('train.csv')
val.to_csv('val.csv')
test.to_csv('test.csv')
```
> The df_path should have train.csv, val.csv, and test.csv with the processed images and labels. If the processing has gone correctly, the dataframe will have "cropped_path", "heart_path", "supp_path", and "label" columns.

### Train Model
To check if there is good signal on a dataset or to compare the trained models against newly trained models on an external dataset, simply run:

```python train.py --checkpoints_dir (root directory to save the model) --df_path (root directory where the train/val/test dataframes are)```
> It shouldn't be necessary but any other training options one wants to change like the model to be trained or learning rates change the options in 'train.json'.

> In the checkpoints_dir the model will be saved as: ./checkpoints_dir/test_name/model.pth

# Inference

To run the inference code, simply run:

```python evaluate_dcms.py --df_path (path to the inference dataframe)```
> The inference code will output a dataframe of the concatinated output of the inference dataframe and the output predictions as **"cac_inference.csv"**.

# Inference on the joint model (*note: run after the image model so that you can just use the processed images)

To run the code, simply run:

```python evaluate_dcms.py```
Note: the mapped_ehr.csv file should be updated and have rows where the first column is the path to the processed main image and the rest of the 32 columns should be the EHR information.


