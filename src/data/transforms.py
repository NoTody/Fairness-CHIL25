import numpy as np
from monai import transforms

def seg_transforms(config, mode='train'):
    roi = config.MODEL.ROI
    num_samples = config.MODEL.NUM_SAMPLES
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
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(roi[0], roi[1], roi[2]),
                    pos=2,
                    neg=1,
                    num_samples=num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], 
                    prob=0.5, 
                    spatial_axis=0
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], 
                    prob=0.5, 
                    spatial_axis=1
                ),
                transforms.RandFlipd(
                    keys=["image", "label"], 
                    prob=0.5, 
                    spatial_axis=2,
                ),
                transforms.RandScaleIntensityd(
                    keys="image", 
                    factors=0.1, 
                    prob=0.5,
                ),
                transforms.RandShiftIntensityd(
                    keys="image", 
                    offsets=0.1, 
                    prob=0.5,
                ),
                # transforms.NormalizeIntensityd(
                #     keys="image", 
                #     nonzero=True, 
                #     channel_wise=True
                # ),
                transforms.ToTensord(
                    keys=["image", "label"],
                ),
            ]
        )
    else:
        trans = transforms.Compose(
            [
                transforms.LoadImaged(
                    keys=["image", "label"], 
                    image_only=False
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
                    mode='constant',
                ),
                # transforms.NormalizeIntensityd(
                #     keys="image", 
                #     nonzero=True, 
                #     channel_wise=True,
                # ),
                transforms.ToTensord(
                    keys=["image", "label"],
                ),
            ]
        )

    return trans


def ssl_transforms(config, mode='train'):
    roi = config.MODEL.ROI
    num_samples = config.MODEL.NUM_SAMPLES
    # Define transforms for image and segmentation
    if mode == 'train':
        trans = transforms.Compose(
            [
                transforms.LoadImaged(
                    keys="image", 
                    image_only=False,
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
                transforms.ScaleIntensityRangePercentilesd(
                    keys="image",
                    lower=0.5,
                    upper=99.5,
                    b_min=0,
                    b_max=1,
                    clip=True,
                ),
                transforms.SpatialPadd(
                    keys="image", 
                    spatial_size=(roi[0], roi[1], roi[2]), 
                    mode='constant',
                ),
                transforms.RandSpatialCropSamplesd(
                    keys="image",
                    roi_size=[roi[0], roi[1], roi[2]],
                    num_samples=num_samples,
                    random_center=True,
                    random_size=False,
                ),
                # transforms.NormalizeIntensityd(
                #     keys="image", 
                #     nonzero=True, 
                #     channel_wise=True
                # ),
                transforms.RandScaleIntensityd(
                    keys="image", 
                    factors=0.1, 
                    prob=0.5,
                ),
                transforms.ToTensord(
                    keys="image",
                ),
            ]
        )
    else:
        trans = transforms.Compose(
            [
                transforms.LoadImaged(
                    keys="image", 
                    image_only=False,
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
                transforms.ScaleIntensityRangePercentilesd(
                    keys="image",
                    lower=0.5,
                    upper=99.5,
                    b_min=0,
                    b_max=1,
                    clip=True,
                ),
                transforms.SpatialPadd(
                    keys="image", 
                    spatial_size=(roi[0], roi[1], roi[2]), 
                    mode='constant',
                ),
                transforms.RandSpatialCropSamplesd(
                    keys="image",
                    roi_size=[roi[0], roi[1], roi[2]],
                    num_samples=num_samples,
                    random_center=True,
                    random_size=False,
                ),
                # transforms.NormalizeIntensityd(
                #     keys="image", 
                #     nonzero=True, 
                #     channel_wise=True
                # ),
                transforms.ToTensord(
                    keys="image",
                ),
            ]
        )

    return trans


def extract_transforms(config, mode='train'):
    roi = config.MODEL.ROI
    num_samples = config.MODEL.NUM_SAMPLES
    # Define transforms for image and segmentation
    trans = transforms.Compose(
        [
            transforms.LoadImaged(
                keys="image", 
                image_only=False,
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
            transforms.ScaleIntensityRangePercentilesd(
                keys="image",
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            transforms.SpatialPadd(
                keys="image", 
                spatial_size=(roi[0], roi[1], roi[2]), 
                mode='constant',
            ),
            # transforms.GridPatchd(
            #     keys="image",
            #     patch_size=(roi[0], roi[1], roi[2]),
            #     num_patches=num_samples,
            #     overlap=0.1,
            #     # start_pos=0,
            #     pad_mode="constant",
            # ),
            transforms.RandSpatialCropSamplesd(
                keys="image",
                roi_size=(roi[0], roi[1], roi[2]),
                num_samples=num_samples,
                random_center=True,
                random_size=False,
            ),
            transforms.ToTensord(
                keys="image",
            ),
        ]
    )

    return trans


class DataAugmentationDINO3D(object):
    def __init__(self, final_size, global_crops_size, local_crops_size, local_crops_number):
        
        flip_and_noise = transforms.Compose([transforms.RandFlip(prob=0.2, spatial_axis=0), 
                                    transforms.RandFlip(prob=0.2, spatial_axis=1), 
                                    transforms.RandFlip(prob=0.2, spatial_axis=2), 
                                    transforms.RandShiftIntensity(offsets=0.2, prob=0.5)
                                    ])
        normalize = transforms.ToTensor()
        
        # Global crop transform
        self.global_transform1 = transforms.Compose(
            transforms.CastToType(np.float32),
            transforms.RandSpatialCrop(final_size, random_center=True, random_size=True),
            flip_and_noise,
            transforms.RandGaussianSmooth(sigma_x=(0.5,1.0), sigma_y=(0.5,1.0), sigma_z=(0.5,1.0), prob=0.2), 
            normalize,
        )

        self.global_transform2 = transforms.Compose([
            transforms.RandSpatialCrop(final_size, random_center=True, random_size=True),
            flip_and_noise, 
            transforms.RandAdjustContrast(gamma=(0.2,1.),prob=0.2),
            normalize,
        ])

        # Local crop transform
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.CastToType(dtype=np.float32),
            transforms.RandSpatialCrop(final_size, random_center=True, random_size=True),
            transforms.RandSpatialCrop(local_crops_size, max_roi_size=global_crops_size, random_center=True, random_size=True),
            transforms.Resize(spatial_size=final_size),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops
    