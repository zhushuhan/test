#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang (hz13@mail.ustc.edu.cn)
Date               : 2021-12-20 20:49
Last Modified By   : ZhenHuang (hz13@mail.ustc.edu.cn)
Last Modified Date : 2021-12-26 17:00
Description        : Create HDF5 dataset
-------- 
Copyright (c) 2021 Multimedia Group USTC. 
'''
import os
import os.path as osp
import h5py
import numpy as np
from visualize_nifti import get_array_from_nifti, process_array
from collections import defaultdict


def create_hdf5_file_v1(inputdir: str,
                        output_path: str,
                        interval=[1, 100, 181, 211, 0, 60]):
    """Create HDF5 dataset from nifti files (v1)

    Parameters
    ----------
    inputdir : str
    output_path : str
        the path name of the output hdf5 file
    interval : list
        [xmin, ymin, xmax, ymax, cmin, cmax]

    Hierarchy of the nifti files
    ----------------------------
    inputdir
        - epe (tumor)
            - 1 (patient)
                - N4BiasFieldCorrection (preprocess && registration on the standard human brain)
                    - T1_Ax_reg.nii.gz (T1 MRI, Ax angle, after regulation)
                    - T1_E_Ax_reg.nii.gz (T1 Enhanced MRI, Ax angle, after regulation)
                    - T2_Ax_reg.nii.gz (T2 MRI, Ax angle, after regulation)
                    - T2_Flair_Ax_reg.nii.gz (Optional, T2 Flair MRI, Ax angle, after regulation)
                - Registration (registration on the standard human brain)
                    - T1_Ax_reg.nii.gz
                    ...
                - T1_Ax (raw nifti file, before registration, usually have less MR Images, smaller nifti file)
                    - T1_Ax.nii.gz (T1 MRI, Ax angle, raw)
                - T1_E_Ax
                - T2_Ax
                - T2_Flair_Ax (Optional)
            - 2
            ...
        - med (tumor)
            - 507024
                - N4BiasFieldCorrection
                - T1_Ax
                ...
            ...

    Attributes of HDF5 file
    -----------------------
    doc: summarized details of the dataset
    angles: of the MRIs
    modalities: of the MRIs
    registration: whether registrated to the normalized human brain
    num_arrays: total number of the MRI sequence (nifti file)
    Reference
    ---------
    https://docs.h5py.org/en/stable/quick.html
    """
    assert osp.exists(inputdir), 'dataset dir {} does not exist.'.format(
        inputdir)
    num_arrays = 0
    num_arrays_per_modality = defaultdict(int)
    mean_per_modality = defaultdict(int)
    std_per_modality = defaultdict(int)
    print('Loading {}...'.format(inputdir))
    with h5py.File(output_path, 'a') as dset:
        dset.attrs[
            'doc'] = 'angles: Ax, modalities: T1 + T1_E + T2 (+ T2_Flair), registration: True'
        print('Doc: {}'.format(dset.attrs['doc']))
        dset.attrs['angles'] = 'Ax'
        dset.attrs['modalities'] = 'T1_Ax_reg+T1_E_Ax_reg+T2_Ax_reg+T2_Flair_Ax_reg'
        dset.attrs['registration'] = True
        num_tumors = -1
        for tumor_type in os.listdir(inputdir):
            num_tumors += 1
            tumor_dir = osp.join(inputdir, tumor_type)
            tumor_grp = dset.create_group(tumor_type)
            num_patients = -1
            for patient in os.listdir(tumor_dir):
                ''' Use the after-registration nifti file, usually have more 180 channels'''
                num_patients += 1
                print('\nProcessing #{} tumor `{}`, #{} patient `{}`...'.format(
                    num_tumors, tumor_type, num_patients, patient))
                patient_dir = osp.join(tumor_dir, patient,
                                       'N4BiasFieldCorrection')
                patient_grp = tumor_grp.create_group(patient)
                for mod in os.listdir(patient_dir):
                    if not mod.endswith('nii.gz'):
                        continue
                    nii_file_path = osp.join(patient_dir, mod)
                    array = get_array_from_nifti(nii_file_path)
                    array = process_array(array)
                    array_to_write = array[interval[4]:interval[5],
                                           interval[0]:interval[2],
                                           interval[1]:interval[3]]
                    array_to_write = array_to_write.astype(np.int16)
                    if mod.startswith('T2Flair'):
                        mod = mod[:2] + '_' + mod[2:]
                    patient_grp.create_dataset(mod[:-7], data=array_to_write)

                    # statistics
                    num_arrays += 1
                    num_arrays_per_modality[mod[:-7]] += 1
                    mean_per_modality[mod[:-7]] += array_to_write.mean().astype(np.float16)
                    std_per_modality[mod[:-7]] += array_to_write.std().astype(np.float16)
                    print('num_arrays: ', num_arrays)
                    print('num_arrays of {}: {}'.format(mod[:-7], num_arrays_per_modality[mod[:-7]]))
                    print('array shape: {}'.format(array_to_write.shape))
                    print('mean of {}: {}'.format(mod[:-7], array_to_write.mean().astype(np.float16)))
                    print('std of {}: {}'.format(mod[:-7], array_to_write.std().astype(np.float16)))

        # write doc string in the attributes
        dset.attrs['num_arrays'] = num_arrays
        for mod in num_arrays_per_modality:
            dset.attrs['num_arrays_{}'.format(mod)] = num_arrays_per_modality[mod]
            dset.attrs['mean_{}'.format(mod)] = mean_per_modality[mod] / num_arrays_per_modality[mod]
            dset.attrs['std_{}'.format(mod)] = std_per_modality[mod] / num_arrays_per_modality[mod]
            print('Pixel mean of {}: {}'.format(mod, dset.attrs['mean_{}'.format(mod)]))
            print('Pixel std of {}: {}'.format(mod, dset.attrs['std_{}'.format(mod)]))


if __name__ == '__main__':
    create_hdf5_file_v1(
        '/Users/zhushuhan/Desktop/new_med_process',
        '/Users/zhushuhan/Desktop/new_med-Ax-T1_T1_E.hdf5')
