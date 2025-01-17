import numpy as np
import nibabel as nib
import SimpleITK as sitk

# Hoyer 2024
# OAI Zuse Institute Berlin OAI Segmentations: 
# mask_labels:
#   0: background
#   1: femoral bone
#   2: femoral cartilage
#   3: tibial bone
#   4: tibial cartilage


def load_mhd(filename:str):
    """
    Load data from a .mdh file
    """
    # Load the .mhd file using SimpleITK
    img = sitk.ReadImage(filename)
    # Convert to NIfTI format
    data = sitk.GetArrayFromImage(img)
    #data = data.swapaxes(0, 1) 
    return data

def get_mhd(mhd_path:str):
    """
    Load .mhd files and perform additional dataset-specific processing.
    """
    # Load OAI-specific .mhd data - ZIB dataset
    data = load_mhd(mhd_path)
    #if 'OAI'== self.dataset_id:
    data = np.flip(data, axis=0)
    data = np.flip(data, axis=2) # reverse slice order to match imgs
    data = np.transpose(data, [2, 0, 1])
    return data

def save_nifti(data, file_name):
    # Create an identity affine transformation matrix
    affine_ = np.eye(4)
    print('nifti', np.shape(data))
    return nib.save(nib.Nifti1Image(data, affine_), file_name)