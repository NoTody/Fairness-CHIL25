from monai import data, transforms

from torch.utils.data import Dataset

import os
import argparse
import pandas as pd
from tqdm import tqdm


def seg_transforms(roi, mode='train'):
    # Define transforms for image and segmentation
    if mode == 'train':
        trans = transforms.Compose(
            [
                transforms.LoadImaged(
                    keys=["image", "label"], 
                    image_only=False,
                ),
                transforms.EnsureChannelFirstd(
                    keys=["image", "label"],
                ),
                transforms.Orientationd(
                    keys=["image", "label"], 
                    axcodes="RAS",
                ),
                transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(0.6, 0.3, 0.3),
                    mode=("bilinear", "nearest"),
                ),
                transforms.ScaleIntensityRangePercentilesd(
                    keys="image",
                    lower=0.5,
                    upper=99.5,
                    b_min=0,
                    b_max=1,
                    clip=True,
                ),
                transforms.SpatialPadd(
                    keys=["image", "label"], 
                    spatial_size=(roi[0], roi[1], roi[2]), 
                    mode='constant'
                ),
            ]
        )

    return trans

class DatasetCache(Dataset):
    def __init__(self, roi, csv_file, cache_dir=None):
        self.data = pd.read_csv(csv_file)
        self.load = seg_transforms(roi)

        base_path = '/data/mskacquisition/MAPSS_data'

        self.cache_dir = cache_dir
        self.cache_dataset = data.PersistentDataset(
            data=list([{"image": os.path.join(base_path, img), "label": os.path.join(base_path, cart)} for img, cart \
                       in zip(self.data['img_path'], self.data['cartilage_path'])]), 
            transform=self.load, 
            cache_dir=self.cache_dir, 
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.cache_dataset.__getitem__(idx)
        print(f"image: {image['image'].shape}")
        return image


parser = argparse.ArgumentParser(description='Example of a command-line argument parser')

# Positional argument
parser.add_argument('--start_idx', type=int, help='Path to the input file')

# Optional argument
parser.add_argument('--end_idx', type=int, help='Path to the input file')

# Flag argument (boolean)
args = parser.parse_args()

roi = [96, 96, 96]
csv_file = '/data/mskacquisition/TBRecon/dataset/MAPSS/final_binary.csv'
cache_dir = '/data/mskacquisition/MAPSS_data/MAPSS_cache_dir_binary'

train_ds = DatasetCache(
    roi, 
    csv_file=csv_file, 
    cache_dir=cache_dir, 
)

for idx in tqdm(range(args.start_idx, args.end_idx)):
    b = train_ds.__getitem__(idx)