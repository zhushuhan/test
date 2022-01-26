#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang (hz13@mail.ustc.edu.cn)
Date               : 2021-12-19 16:10
Last Modified By   : ZhenHuang (hz13@mail.ustc.edu.cn)
Last Modified Date : 2021-12-26 13:18
Description        : visualize nii file
-------- 
Copyright (c) 2021 Alibaba Inc. 
'''
import os
import os.path as osp
import SimpleITK as sitk
import cv2
import numpy as np
import random
import torch
from typing import DefaultDict, Union, Optional, List, Tuple, Text, BinaryIO
from PIL import Image, ImageDraw, ImageFont, ImageColor

def get_array_from_nifti(input_nii_path: str) -> np.ndarray:
    """Get np.ndarray from nifti file

    Parameters
    ----------
    input_nii_path : str
        path of the nifti file

    Returns
    -------
    np.ndarray

    Reference
    ---------
    https://github.com/SimpleITK/SimpleITK
    """
    if not input_nii_path.endswith('nii.gz'):
        raise ValueError('{} does not ends with nii.gz'.format(input_nii_path))
    nii_image = sitk.ReadImage(input_nii_path)
    array = sitk.GetArrayFromImage(nii_image)
    return array

def process_array(array: np.ndarray) -> np.ndarray:
    """Process the image array, make sure the intensities is within [0, 255]

    Parameters
    ----------
    array : np.ndarray

    Returns
    -------
    np.ndarray
    """
    # make the pixels values is above 0
    array = array + array.min()
    values = array.reshape(-1).copy()
    values.sort()
    # clip off the top 1% pixel values
    top = values[int(len(values) * 0.99)]
    # make sure the pixel is in [0, 255]
    array = np.clip(array, 0, top) / top * 255.
    return array[:, ::-1, :]

def write_nifti_as_image_series(input_nii_path: str, output_image_dir: str) -> None:
    """Load the nifti file and save the image series

    Parameters
    ----------
    input_nii_path : str
        path of the nifti file
    output_image_dir : str
        directory of the output image series
    """
    array = get_array_from_nifti(input_nii_path)
    array = process_array(array)
    if not osp.exists(output_image_dir):
        os.mkdir(output_image_dir)
    for i, x in enumerate(array):
        cv2.imwrite(os.path.join(output_image_dir, '{:0>3d}.jpeg'.format(i)), x)


def visualize_nifti_files(input_nii_dir: str, output_dir: str, num_patients: int=4):
    """Random sample `num_patients` examples, and show the bounding boxes and intervals

    Parameters
    ----------
    input_nii_dir : str
    output_dir : str
    num_patients : int, optional
        number of random patients, by default 4
    """
    # epe tumor
    epe_dir = osp.join(input_nii_dir, 'train_data_epe')
    assert osp.exists(epe_dir), 'dataset dir {} does not exist.'.format(epe_dir)
    epe_patients_list = os.listdir(epe_dir)
    epe_patients_ex = random.sample(epe_patients_list, num_patients)

    for patient in epe_patients_ex:
        patient_dir = osp.join(epe_dir, patient, 'N4BiasFieldCorrection')
        for mod in os.listdir(patient_dir):
            nii_file_path = osp.join(patient_dir, mod)
            output_image_dir = osp.join(output_dir, 'epe', patient, mod[:-7])
            os.makedirs(output_image_dir, exist_ok=True)
            write_nifti_as_image_series(nii_file_path, output_image_dir)

    # med tumor
    med_dir = osp.join(input_nii_dir, 'train_data_med')
    assert osp.exists(med_dir), 'dataset dir {} does not exist.'.format(med_dir)
    med_patients_list = os.listdir(med_dir)
    med_patients_ex = random.sample(med_patients_list, num_patients)

    for patient in med_patients_ex:
        patient_dir = osp.join(med_dir, patient, 'N4BiasFieldCorrection')
        for mod in os.listdir(patient_dir):
            nii_file_path = osp.join(patient_dir, mod)
            output_image_dir = osp.join(output_dir, 'med', patient, mod[:-7])
            os.makedirs(output_image_dir, exist_ok=True)
            write_nifti_as_image_series(nii_file_path, output_image_dir)


def annotate_boxes_for_nifti(input_nii_path: str, output_image_dir: str, width: int = 1, interval=[85, 250, 426, 460, 0, 23]) -> None:
    """ Annotate bounding boxes on image seris

    Parameters
    ----------
    input_nii_path : str
    output_image_dir : str
    width : int, optional
        width of the box line, by default 1
    interval : list, optional
        [left, up, right, down, begin, end], by default [10, 20, 30, 40, 50, 60]
    """
    array = get_array_from_nifti(input_nii_path)
    array = process_array(array) # [C, H, W] [182, 218, 182]
    if not osp.exists(output_image_dir):
        os.mkdir(output_image_dir)
    for i, x in enumerate(array):
        if interval[-2] <= i <= interval[-1]:
            # [218, 182]
            img_to_draw = Image.fromarray(x)
            draw = ImageDraw.Draw(img_to_draw)
            draw.rectangle(interval[:4], width=width, outline=(255,))
            cv2.imwrite(os.path.join(output_image_dir, '{:0>3d}.jpeg'.format(i)), np.array(img_to_draw))
        else:
            cv2.imwrite(os.path.join(output_image_dir, '{:0>3d}.jpeg'.format(i)), np.array(x))

def annotate_boxes_for_all(input_nii_path: str, output_image_path: str, interval=[85, 250, 426, 460, 0, 23]):
    assert osp.exists(input_nii_path), 'dataset dir {} does not exist.'.format(
        input_nii_path)
    if not osp.exists(output_image_path):
        os.mkdir(output_image_path)
    patient_list = os.listdir(input_nii_path)
    for patient in patient_list:
        inputfile1 = input_nii_path + '/' + patient
        output_file1 = output_image_path + '/' + patient
        if not osp.exists(output_file1):
            os.mkdir(output_file1)
        modality_list = os.listdir(inputfile1)
        for modality in modality_list:
            inputfile2 = inputfile1 + '/' + modality
            #print(outputfile2)
            output_file2 = output_file1 + '/' + modality
            if not osp.exists(output_file2):
                os.mkdir(output_file2)
            data = (os.listdir(inputfile2))[0]
            datafile = inputfile2 + '/' + data
            annotate_boxes_for_nifti(datafile, output_file2, width = 1, interval = interval)


def compute_dataset_statistics(inputdir: str):
    from collections import defaultdict
    assert osp.exists(inputdir), 'dataset dir {} does not exist.'.format(inputdir)
    tumor_count = defaultdict(int)
    mean_per_modality = defaultdict(int)
    std_per_modality = defaultdict(int)
    num_per_modality = defaultdict(int)
    for tumor_type in os.listdir(inputdir):
        tumor_dir = osp.join(inputdir, tumor_type)
        mod_count = defaultdict(int)
        for patient in os.listdir(tumor_dir):
            tumor_count[tumor_type] += 1
            patient_dir = osp.join(tumor_dir, patient,
                                    'N4BiasFieldCorrection')
            for mod in os.listdir(patient_dir):
                if not mod.endswith('nii.gz'):
                    continue
                nii_file_path = osp.join(patient_dir, mod)
                array = get_array_from_nifti(nii_file_path)
                array = process_array(array)
                if mod.startswith('T2Flair'):
                    mod = mod[:2] + '_' + mod[2:]
                mean_per_modality[mod[:-7]] += array.mean().astype(np.float16)
                std_per_modality[mod[:-7]] += array.std().astype(np.float16)
                num_per_modality[mod[:-7]] += 1
                mod_count[mod[:-7]] += 1

        print('\nTumor {} has {} patients in total.'.format(tumor_type, tumor_count[tumor_type]))
        for mod, num in mod_count.items():
            print('Modality {}: {}'.format(mod, num))

    print('Mean:')
    for mod, v in mean_per_modality.items():
        print('Modality {}: {}'.format(mod, v/num_per_modality[mod]))
    print('Std:')
    for mod, v in std_per_modality.items():
        print('Modality {}: {}'.format(mod, v/num_per_modality[mod]))


if __name__ == '__main__':
    # write_nifti_as_image_series('/Users/zhushuhan/Desktop/new_med_process/med/FU ZHENG XIANG/T1_E_Ax/T1_E_Ax.nii.gz', '/Users/zhushuhan/Desktop/med_visual')
    # visualize_nifti_files('/home/muzhao.hz/Projects/Medical/data/med_mpe-ax_T1_T2_E/', './image_series', num_patients=5)
    # annotate_boxes_for_all('/Users/zhushuhan/Desktop/new_med_process/med_resize', '/Users/zhushuhan/Desktop/image_draw')
    # compute_dataset_statistics('/home/muzhao.hz/Projects/Medical/data/med_epe-ax_T1_T2_E/')
    # write_nifti_as_image_series('/Users/zhushuhan/Desktop/redata/med/507024/N4BiasFieldCorrection/T1_E_Ax_reg.nii.gz','/Users/zhushuhan/Desktop/original_med_visual')
    # annotate_boxes_for_nifti('/Users/zhushuhan/Desktop/redata/med/507024/N4BiasFieldCorrection/T1_E_Ax_reg.nii.gz','//Users/zhushuhan/Desktop/original_image_draw')
    # write_nifti_as_image_series('/Users/zhushuhan/Desktop/new_med_process/med_resize/LI KE/T1_E_Ax/T1_E_Ax.nii.gz','/Users/zhushuhan/Desktop/image_draw_LI KE_T1_E_Ax')
    annotate_boxes_for_all('/Users/zhushuhan/Desktop/new_med_process/med_resize', '/Users/zhushuhan/Desktop/image_draw_large', interval=[58, 255 ,453, 498, 0, 23])


