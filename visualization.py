import os
import sys
import re

import json
import time
#import fibrosis_analysis


import pandas as pd
import numpy as np
import pickle

from importlib import reload  
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import gc

import matplotlib.animation as manimation

def plot_slices_dual(volume, 
                     gt_np,
                     pred_np,
                     num_labels,
                     window="knee", 
                     max_slices=None, 
                     custom_bounds=None, 
                     title1="Ground Truth",
                     title2="Prediction"):
    # Extract numpy arrays from SimpleITK images
    #volume_np = np.array(volume)
    
    volume_np = volume

    # Generate alpha masks based on the segmentations
    gt_alpha_masks = [np.where(gt_np == i, 0.8, 0) for i in range(1, num_labels+1)]

    # Generate alpha masks based on the segmentations
    pred_alpha_masks = [np.where(pred_np == i, 0.8, 0) for i in range(1, num_labels+1)]
    
    # Set up the figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    lower = np.min(volume_np)
    upper = np.max(volume_np)

    if num_labels == 1:
        cmaps = ['winter']
    elif num_labels == 3:
        cmaps = ['winter', 'autumn', 'cool']
    elif num_labels == 6:
        cmaps = ['winter', 'autumn', 'cool', 'summer', 'hot', 'spring']
    
    im_vol1 = ax1.imshow(volume_np[0, :, :], animated=True, cmap="gray", \
                         vmin=lower, vmax=upper)
    im_segs1 = [
        ax1.imshow(gt_np[0, :, :], animated=True, cmap=cmaps[i], \
                   alpha=gt_alpha_masks[i][0, :, :], vmin=0, vmax=1)
        for i in range(num_labels)
    ]
    
    ax1.set_title(title1)
    ax1.axis('off')
    
    im_vol2 = ax2.imshow(volume_np[0, :, :], animated=True, cmap="gray", \
                         vmin=lower, vmax=upper)
    im_segs2 = [
        ax2.imshow(pred_np[0, :, :], animated=True, cmap=cmaps[i], \
                   alpha=pred_alpha_masks[i][0, :, :], vmin=0, vmax=1)
        for i in range(num_labels)
    ]
    
    ax2.set_title(title2)
    ax2.axis('off')
    
    depth = volume_np.shape[0]
    step = 1 if max_slices is None else max(1, depth // max_slices)
    
    # Initialize the text label and store its reference
    slice_label = ax2.text(volume_np.shape[2]-10, volume_np.shape[1]-10, f"0/{depth}", 
                           ha="right", va="bottom", color="white", fontsize=8, weight="bold")
    
    def update(i):
        ret = []
        
        im_vol1.set_array(volume_np[i, :, :])
        ret.append(im_vol1)
        
        for im_seg, gt_alpha_mask in zip(im_segs1, gt_alpha_masks):
            im_seg.set_array(gt_np[i, :, :])
            im_seg.set_alpha(gt_alpha_mask[i, :, :])
            ret.append(im_seg)
        
        im_vol2.set_array(volume_np[i, :, :])
        ret.append(im_vol2)
        
        for im_seg, pred_alpha_mask in zip(im_segs2, pred_alpha_masks):
            im_seg.set_array(pred_np[i, :, :])
            im_seg.set_alpha(pred_alpha_mask[i, :, :])
            ret.append(im_seg)
        
        slice_label.set_text(f"{i}/{depth}")
        
        return ret
    
    ani = manimation.FuncAnimation(fig, update, frames=range(0, depth, step), blit=True)
    
    return ani


def plot_slices_one(volume, 
                    gt_np,
                    num_labels,
                    window="knee", 
                    max_slices=None, 
                    custom_bounds=None, 
                    title1="Ground Truth",
                    ):
    # Extract numpy arrays from SimpleITK images
    #volume_np = np.array(volume)
    
    volume_np = volume

    # Generate alpha masks based on the segmentations
    gt_alpha_masks = [np.where(gt_np == i, 0.8, 0) for i in range(1, num_labels+1)]

    # Set up the figure and axis
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    lower = np.min(volume_np)
    upper = np.max(volume_np)

    if num_labels == 1:
        cmaps = ['winter']
    elif num_labels == 3:
        cmaps = ['winter', 'autumn', 'cool']
    elif num_labels == 4:
        cmaps = ['winter', 'autumn', 'cool', 'summer']
    elif num_labels == 6:
        cmaps = ['winter', 'autumn', 'cool', 'summer', 'hot', 'spring']
    
    im_vol1 = ax1.imshow(volume_np[0, :, :], animated=True, cmap="gray", \
                         vmin=lower, vmax=upper)
    im_segs1 = [
        ax1.imshow(gt_np[0, :, :], animated=True, cmap=cmaps[i], \
                   alpha=gt_alpha_masks[i][0, :, :], vmin=0, vmax=1)
        for i in range(num_labels)
    ]
    
    ax1.set_title(title1)
    ax1.axis('off')
    
    depth = volume_np.shape[0]
    step = 1 if max_slices is None else max(1, depth // max_slices)
    
    def update(i):
        ret = []
        
        im_vol1.set_array(volume_np[i, :, :])
        ret.append(im_vol1)
        
        for im_seg, gt_alpha_mask in zip(im_segs1, gt_alpha_masks):
            im_seg.set_array(gt_np[i, :, :])
            im_seg.set_alpha(gt_alpha_mask[i, :, :])
            ret.append(im_seg)
        
        return ret
    
    ani = manimation.FuncAnimation(fig, update, frames=range(0, depth, step), blit=True)
    
    return ani

def plot_slices_img(volume, 
                    window="knee", 
                    max_slices=None, 
                    custom_bounds=None, 
                    title1="Ground Truth",
                    ):
    # Extract numpy arrays from SimpleITK images
    #volume_np = np.array(volume)
    
    volume_np = volume
    
    # Set up the figure and axis
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    lower = np.min(volume_np)
    upper = np.max(volume_np)

    im_vol1 = ax1.imshow(volume_np[0, :, :], animated=True, cmap="gray", \
                         vmin=lower, vmax=upper)
    ax1.set_title(title1)
    ax1.axis('off')
    
    depth = volume_np.shape[0]
    step = 1 if max_slices is None else max(1, depth // max_slices)

    # Initialize the text label and store its reference
    slice_label = ax1.text(volume_np.shape[2]-10, volume_np.shape[1]-10, f"0/{depth}", 
                           ha="right", va="bottom", color="white", fontsize=8, weight="bold")
    
    def update(i):
        ret = []
        
        im_vol1.set_array(volume_np[i, :, :])
        ret.append(im_vol1)

        slice_label.set_text(f"{i}/{depth}")
        return ret
    
    ani = manimation.FuncAnimation(fig, update, frames=range(0, depth, step), blit=True)
    
    return ani


def plot_slices_tri(volume, 
                    gt_np,
                    pred1_np,
                    pred2_np, 
                    num_labels,
                    gt_num_labels,
                    window="knee", 
                    max_slices=None, 
                    custom_bounds=None, 
                    title1="Ground Truth",
                    title2="Prediction (Binary)",
                    title3="Prediction (Multi)"):
    # Extract numpy arrays from SimpleITK images
    #volume_np = np.array(volume)
    
    volume_np = volume

    # Generate alpha masks based on the segmentations
    gt_alpha_masks = [np.where(gt_np == i, 0.8, 0) for i in range(1, gt_num_labels+1)]

    # Generate alpha masks based on the segmentations
    pred1_alpha_masks = [np.where(pred1_np == i, 0.8, 0) for i in range(1, num_labels+1)]
    pred2_alpha_masks = [np.where(pred2_np == i, 0.8, 0) for i in range(1, gt_num_labels+1)]
    
    # Set up the figure and axis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    
    lower = np.min(volume_np)
    upper = np.max(volume_np)

    # Get the 'tab10' colormap
    tab10 = plt.get_cmap('tab10')
    
    cmaps1 = [tab10(i) for i in range(num_labels)]
    cmaps2 = [tab10(i) for i in range(gt_num_labels)]
    
    im_vol1 = ax1.imshow(volume_np[0, :, :], animated=True, cmap="gray", \
                         vmin=lower, vmax=upper)
    im_segs1 = [
        ax1.imshow(gt_np[0, :, :], animated=True, cmap=ListedColormap(cmaps2[i]), \
                   alpha=gt_alpha_masks[i][0, :, :], vmin=0, vmax=1)
        for i in range(gt_num_labels)
    ]
    ax1.set_title(title1)
    ax1.axis('off')
    
    im_vol2 = ax2.imshow(volume_np[0, :, :], animated=True, cmap="gray", \
                         vmin=lower, vmax=upper)
    im_segs2 = [
        ax2.imshow(pred1_np[0, :, :], animated=True, cmap=ListedColormap(cmaps1[i]), \
                   alpha=pred1_alpha_masks[i][0, :, :], vmin=0, vmax=1)
        for i in range(2)
    ]
    ax2.set_title(title2)
    ax2.axis('off')
    
    im_vol3 = ax3.imshow(volume_np[0, :, :], animated=True, cmap="gray", \
                         vmin=lower, vmax=upper)
    im_segs3 = [
        ax3.imshow(pred2_np[0, :, :], animated=True, cmap=ListedColormap(cmaps2[i]), \
                   alpha=pred2_alpha_masks[i][0, :, :], vmin=0, vmax=1)
        for i in range(gt_num_labels)
    ]
    ax3.set_title(title3)
    ax3.axis('off')
    
    depth = volume_np.shape[0]
    step = 1 if max_slices is None else max(1, depth // max_slices)
    
    # Initialize the text label and store its reference
    slice_label = ax2.text(volume_np.shape[2]-10, volume_np.shape[1]-10, f"0/{depth}", 
                           ha="right", va="bottom", color="white", fontsize=8, weight="bold")
    
    def update(i):
        ret = []
        
        im_vol1.set_array(volume_np[i, :, :])
        ret.append(im_vol1)
        
        for im_seg, gt_alpha_mask in zip(im_segs1, gt_alpha_masks):
            im_seg.set_array(gt_np[i, :, :])
            im_seg.set_alpha(gt_alpha_mask[i, :, :])
            ret.append(im_seg)
        
        im_vol2.set_array(volume_np[i, :, :])
        ret.append(im_vol2)
        
        for im_seg, pred_alpha_mask in zip(im_segs2, pred1_alpha_masks):
            im_seg.set_array(pred1_np[i, :, :])
            im_seg.set_alpha(pred_alpha_mask[i, :, :])
            ret.append(im_seg)

        im_vol3.set_array(volume_np[i, :, :])
        ret.append(im_vol3)
        
        for im_seg, pred_alpha_mask in zip(im_segs3, pred2_alpha_masks):
            im_seg.set_array(pred2_np[i, :, :])
            im_seg.set_alpha(pred_alpha_mask[i, :, :])
            ret.append(im_seg)
        
        slice_label.set_text(f"{i}/{depth}")
        
        return ret
    
    ani = manimation.FuncAnimation(fig, update, frames=range(0, depth, step), blit=True)
    
    return ani