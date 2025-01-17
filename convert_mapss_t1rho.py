from monai import transforms
import os
import pickle
#import nibabel as nib
import numpy as np

trans = transforms.Compose(
    [
        transforms.LoadImaged(
            keys="image", 
            image_only=True
        ),
        transforms.EnsureChannelFirstd(
            keys="image",
        ),
        transforms.Orientationd(
            keys="image", 
            axcodes="RAS",
        ),
        transforms.Spacingd(
            keys="image",
            pixdim=(0.6, 0.3, 0.3),
            mode="bilinear",
        ),
    ]
)

base_path = '/data/mskacquisition/howard_temp/MAPSS_data/MAPSS_T1rho/'

files = os.listdir(base_path)

data_dict = []

for f in files:
    path = os.path.join(base_path, f)
    img_dict = {"image": path}
#    data_dict.append({"image": path})

# print(data_dict[0])

    image = trans(img_dict)

    print(f"image shape: {image['image'].shape}")
    print(f"image pixdim:\n{image['image'].pixdim}")

    widget_state = {
        't1rho': np.squeeze(image['image']),
    }

    base_path = "/data/mskacquisition/howard_temp/MAPSS_data/MAPSS_T1rho/"
    f_name = f"{f}.pkl"
    save_path = os.path.join(base_path, f_name)

    with open(save_path, 'wb') as f:
        pickle.dump(widget_state, f)

