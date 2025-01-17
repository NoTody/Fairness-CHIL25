import numpy as np
from scipy.io import loadmat
import nibabel as nib
import cv2

from collections import OrderedDict

# ****** Dataset mask key: 'datastruct' ******** #

imorphics_assign = OrderedDict([
    ('Background', 0),
    ('FemoralCartilage', 1),
    ('LateralTibialCartilage', 2),
    ('MedialTibialCartilage', 3),
    ('PatellarCartilage', 4),
    ('LateralMeniscus', 5),
    ('MedialMeniscus', 6),
])

# ********** Functions to load Imorphics segmentations ********** #
def load_mat(filename:str, key:str, struct_as_record:bool=True):
    """
    Load data from a .mat file
    Args:
        struct_as_record: default when using loadmat is true, false when loading imorphics
    """
    data = loadmat(filename, struct_as_record=struct_as_record)[key]
    return data

def get_mat(mat_path:str, key:str):
    """
    Load .mat files and perform additional dataset-specific processing.
    """
    data = load_mat(mat_path, key, struct_as_record=False)

    #if 'OAI' == self.dataset_id:
    data = extract_imorphics_mask(data)
    data = np.flip(data, axis=2) # reverse slice order to match imgs
    # Reorient to shape (num_slices, H, W) to match dicom nifti
    data = np.transpose(data, [2, 0, 1])
    return data 

def extract_imorphics_mask(masks_mat, margin = 0):
    """
    Extract mask from imorphic .mat structure using attributes.
    """
    def mask_from_mat(masks_mat, mask_shape, slice_idx, attr_name):
        mask = np.zeros(mask_shape, dtype=np.uint8)
        data = getattr(masks_mat[0][slice_idx], attr_name)
        if len(data.shape) > 0:
            for comp in range(data.shape[1]):
                cnt = data[0, comp][:, :2].copy()
                cnt[:, 1] = mask_shape[0] - cnt[:, 1]
                cntf = cnt.astype(int)
                cv2.drawContours(mask, [cntf], -1, (255, 255, 255), -1)
        mask = (mask > 0).astype(np.uint8)
        return mask

    # Initialize a list to store slice data
    slice_data_list = []
    num_slices  = masks_mat.shape[1]
    for slice_idx in range(num_slices):
        mask_proc = np.zeros((384, 384))  # Replace with actual image shape

        for part_name, part_value in reversed(imorphics_assign.items()):
            if part_name == 'Background':
                continue
            try:
                mask_temp = mask_from_mat(masks_mat, (384, 384), slice_idx, part_name)  # Replace with actual image shape
                mask_proc[mask_temp > 0] = part_value
            except AttributeError:
                print(f'Error accessing {part_name} in {slice_idx}')

        if margin != 0:
            mask_proc = mask_proc[margin:-margin, margin:-margin]

        slice_data_list.append(mask_proc)

    # Convert slice_data_list to a 3D array (W, H, num_slices)
    data = np.dstack(slice_data_list)

    return data

def save_nifti(data, file_name):
    #import pdb; pdb.set_trace()
    # Create an identity affine transformation matrix
    affine_ = np.eye(4)
    print('nifti', np.shape(data))
    return nib.save(nib.Nifti1Image(data, affine_), file_name)