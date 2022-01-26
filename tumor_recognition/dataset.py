#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang (hz13@mail.ustc.edu.cn)
Date               : 2021-12-20 22:26
Last Modified By   : ZhenHuang (hz13@mail.ustc.edu.cn)
Last Modified Date : 2021-12-26 18:04
Description        : DataSet
-------- 
Copyright (c) 2021 Multimedia Group USTC. 
'''
import random
import torch
import logging
import torchvision
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import numpy as np
import h5py
from torchvision import transforms as T

_logger = logging.getLogger('vis')


def temporal_sampling(array: np.ndarray, num_frames: int,
                      step: int) -> np.ndarray:
    """Sample the numpy ndarray along the temporal dimension

    Parameters
    ----------
    array : np.ndarray
        [T, H, W] or [C, T, H, W] or [B, C, T, H, W]
    num_frames : int
        number of the output length of the temporal dimension
    step : int
        step size of the uniform sampling

    Returns
    -------
    np.ndarray
        # of dim equals to the input array. The length of the temporal dimension (the last third dimension) is `num_frames`
    """
    T = array.shape[-3]
    length = num_frames * step
    assert T >= length, 'num_frames({}) * step({}) shoud <= T({})'.format(
        num_frames, step, T)
    start = np.random.randint(0, T - length)
    end = start + length
    out_array = array[..., start:end:step, :, :]
    return out_array


class BrainDatasetV1(Dataset):
    """ Dataset for brain data for HDF5 file, created by `create_hdf5_file_v1` function

    Init Args
    ---------
    data_list : list
        list of the arrays and labels
    train : bool
        whether for training
    num_frames : int, optional
        number of frames in the dataset, by default 6
    sample_step : int, optional
        step between consecutive frames, by default 5
    rand_crop : bool, optional
        whether spatially random crop
    """
    def __init__(
        self,
        data_list: List[Tuple[np.ndarray, int]],
        train: bool = False,
        num_frames: int = 6,
        sample_step: int = 5,
        rand_crop: bool = False,
    ):
        super().__init__()
        self.data_list = data_list
        self.train = train
        self.num_frames = num_frames
        self.sample_step = sample_step
        self.rand_crop = rand_crop
        if rand_crop:
            raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        array, label = self.data_list[index]  # [3, T, H, W], 1
        if self.train:
            array = temporal_sampling(array, self.num_frames, self.sample_step)
        tensor = torch.from_numpy(array).float().div(255)
        return tensor, label


def get_brain_hdf5_dataset_v1(hdf5_path: str,
                              modality: List[str] = [
                                  'T1_Ax_reg', 'T1_E_Ax_reg', 'T2_Ax_reg'
                              ],
                              test_ratio: float = 0.2,
                              num_frames: int = 6,
                              sample_step: int = 5,
                              seed: int = 1) -> Tuple[Dataset, Dataset]:
    """ Get Dataset for brain hdf5 file

    Parameters
    ----------
    hdf5_path : str
        path of the hdf5 path
    modality : List[str]
        list of modalities you want to use
    test_ratio : float, optional
        ratio of the testset, by default 0.2
    num_frames : int, optional
        number of frames in the dataset, by default 6
    sample_step : int, optional
        step between consecutive frames, by default 5
    seed : int, optional
        random seed, by default 1, the division plan is fixed once the seed is fixed

    Returns
    -------
    Tuple[Dataset, Dataset]
        Train dataset, Test dataset
    """
    ''' load hdf5 file '''
    assert hdf5_path.endswith('.hdf5'), '{} does not ends with .hdf5'.format(
        hdf5_path)
    _logger.info('Load file {}'.format(hdf5_path))
    hdf5_file = h5py.File(hdf5_path, 'r')
    doc = hdf5_file.attrs['doc']
    _logger.info('Infos: {}'.format(doc))
    available_modalities = hdf5_file.attrs['modalities'].split('+')
    for m in modality:
        assert m in available_modalities, 'modality {} is not supported, maybe {}'.format(
            m, available_modalities)
    ''' load mean and std '''
    mean_per_modality = {
        m: hdf5_file.attrs['mean_{}'.format(m)].astype(np.float16)
        for m in modality
    }
    std_per_modality = {
        m: hdf5_file.attrs['std_{}'.format(m)].astype(np.float16)
        for m in modality
    }
    _logger.info('Pixel mean of each modality: {}'.format(mean_per_modality))
    _logger.info('Pixel std of each modality: {}'.format(std_per_modality))
    ''' create info list '''
    info_list = []
    tumor_label = -1
    for tumor_type, tumor_grp in hdf5_file.items():
        tumor_label += 1
        for patient, patient_grp in tumor_grp.items():
            # Note that, here I assume every modality array of a patient has the exactly the same shape.
            # Based on this, I could stack them along the `channel` dimension and make a sythesize a fake sequence
            array_list = []
            # Some patients do not have all the modalities, this situation does exist.
            # So I first get the shape of the available modality, and pad the missed modality with zeros
            # array_shape = patient_grp.values()[0].shape
            for _, _array in patient_grp.items():
                array_shape = _array.shape
                break

            for mod in modality:
                if mod in patient_grp:
                    # this patient does have this modality
                    mod_array = patient_grp[mod]
                    # normalize the array, based on the statistics of specific modality
                    mod_array -= mean_per_modality[mod]
                    mod_array /= std_per_modality[mod]
                else:
                    # this patient does not have this modality
                    # fill the array with zeros
                    _logger.warn(
                        'Tumor({}) - Patient({}) does not have modality {}'.
                        format(tumor_type, patient, mod))
                    mod_array = np.zeros(array_shape)
                array_list.append(mod_array)
            stacked_array = np.stack(array_list, axis=0)  # [C, T, H, W]
            info_list.append((stacked_array, tumor_label))

    # split train and test, the split plan is fixed once the seed is fixed
    random.seed(seed)
    random.shuffle(info_list)
    num_test = int(test_ratio * len(info_list))
    train_list, test_list = info_list[num_test:], info_list[:num_test]
    trainset = BrainDatasetV1(train_list,
                              train=True,
                              num_frames=num_frames,
                              sample_step=sample_step)
    testset = BrainDatasetV1(test_list,
                             train=False,
                             num_frames=num_frames,
                             sample_step=sample_step)
    return trainset, testset
