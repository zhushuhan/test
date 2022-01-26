import os
import json
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm  # tqdm是一个包，要import tqdm.tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision import transforms


def read_split_slice(root: str, val_rate: float = 0.3):  # 返回每个slice的path label

    assert os.path.exists(root), "dataset root: {} does not exist.".format(
        root)

    random.seed(0)

    # 1.以 subgroup 区分 训练集验证集
    molecular_class = [
        cla for cla in os.listdir(root)
        if os.path.isdir(os.path.join(root, cla))
    ]  # SHH, G3_G4, WNT

    # 排序，保证顺序一致
    molecular_class.sort()
    # 生成类别名称及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(molecular_class))
    json_str = json.dumps(dict(
        (val, key) for key, val in class_indices.items()),
                          indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = ['.jpg', '.png']

    # 遍历每个文件夹下的文件
    for cla in molecular_class:
        cla_path = os.path.join(root, cla)
        # 包含所有sample的列表
        sample_lst = os.listdir(cla_path)
        # 按比例随机采样val data
        eval_sample_lst = random.sample(sample_lst,
                                        k=int(len(sample_lst) * val_rate))
        # 获取该类别索引
        image_class = class_indices[cla]

        for sample in sample_lst:
            sample_path = os.path.join(cla_path, sample)
            if sample in eval_sample_lst:
                img_lst = os.listdir(sample_path)
                for img in img_lst:
                    img_path = os.path.join(sample_path, img)
                    val_images_path.append(img_path)
                    val_images_label.append(image_class)
            else:
                img_lst = os.listdir(sample_path)
                for img in img_lst:
                    img_path = os.path.join(sample_path, img)
                    train_images_path.append(img_path)
                    train_images_label.append(image_class)

    num_med = 0
    num_epe = 0
    label_lst = train_images_label + val_images_label

    for i in label_lst:
        if (i == 0):
            num_epe += 1
        else:
            num_med += 1

    num = num_epe + num_med
    print(
        "{} images were found in the dataset, including {} ependy slices, {} medull slices.\n"
        .format(num, num_epe, num_med))

    return train_images_path, train_images_label, val_images_path, val_images_label


def read_split_sample(root: str,
                      val_rate: float = 0.3):  # 返回每个slice的path label

    assert os.path.exists(root), "dataset root: {} does not exist.".format(
        root)

    random.seed(0)

    # 1.以 subgroup 区分 训练集验证集
    molecular_class = [
        cla for cla in os.listdir(root)
        if os.path.isdir(os.path.join(root, cla))
    ]  # SHH, G3_G4, WNT

    # 排序，保证顺序一致
    molecular_class.sort()
    # 生成类别名称及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(molecular_class))
    json_str = json.dumps(dict(
        (val, key) for key, val in class_indices.items()),
                          indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_samples_path = []
    train_samples_label = []
    val_samples_path = []
    val_samples_label = []
    every_class_num = []
    supported = ['.jpg', '.png']

    # 遍历每个文件夹下的文件
    for cla in molecular_class:
        cla_path = os.path.join(root, cla)
        # 包含所有sample的列表
        sample_lst = os.listdir(cla_path)
        # 按比例随机采样val data
        eval_sample_lst = random.sample(sample_lst,
                                        k=int(len(sample_lst) * val_rate))
        # 获取该类别索引
        sample_class = class_indices[cla]

        for sample in sample_lst:
            sample_path = os.path.join(cla_path, sample)
            if sample in eval_sample_lst:
                val_samples_path.append(sample_path)
                val_samples_label.append(sample_class)
            else:
                train_samples_path.append(sample_path)
                train_samples_label.append(sample_class)

    num_med = 0
    num_epe = 0
    label_lst = train_samples_label + val_samples_label

    for i in label_lst:
        if (i == 0):
            num_epe += 1
        else:
            num_med += 1

    num = num_epe + num_med
    print(
        "{} samples were found in the dataset, including {} ependy samples, {} medull samples.\n"
        .format(num, num_epe, num_med))

    return train_samples_path, train_samples_label, val_samples_path, val_samples_label


# 读取图片并转换为灰度图
def load_img(img_path):
    img = Image.open(img_path)
    img_gray = transforms.Grayscale(1)(img)
    return img_gray


def resize_img(img):
    img = transforms.Resize([256, 256])(img)
    return img


def calculate_mean_std(root, cal=False):
    if (cal == False):
        return (0.310, 0.083)
    else:
        img_mean = []
        img_std = []
        dir_lst = os.listdir(path)
        for dir in dir_lst:
            dir_path = os.path.join(root, dir)
            for sample in os.listdir(dir_path):
                sample_path = os.path.join(dir_path, sample)
                for img in os.listdir(sample_path):
                    img_path = os.path.join(sample_path, img)
                    img = load_img(img_path)
                    img = resize_img(img)
                    img = np.array(img, dtype=np.float32) / 255
                    img_mean.append(np.mean(img))
                    img_std.append(np.std(img))
    mean = np.mean(img_mean)
    std = np.mean(img_std)
    return mean, std


def train_one_epoch(model, optimizer, data_loader, device):

    model.train()

    loss_function = torch.nn.CrossEntropyLoss()

    mean_loss = torch.zeros(1).to(device)
    correct_num = torch.zeros(1).to(device)
    total_num = len(data_loader.dataset)

    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):

        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        pred_fc = model(images)
        pred = torch.max(pred_fc, dim=1)[1]  # 分类预测结果
        correct_num += torch.eq(pred, labels).sum()  #统计分类正确的个数

        loss = loss_function(pred_fc, labels)  #分类损失函数

        optimizer.zero_grad()  # 梯度归零

        loss.backward()  # 反向传播， 求参数的梯度

        optimizer.step()  # 参数更新

        mean_loss = (mean_loss * step + loss.detach()) / (
            step + 1)  # update mean losses

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    mean_acc = correct_num / total_num

    return mean_loss.item(), mean_acc.item()


def setup_seed(seed):
    random.seed(seed)  # python随机因素
    np.random.seed(seed)  # numpy随机因素
    torch.manual_seed(seed)  # pytorch cpu 随机因素
    torch.cuda.manual_seed(seed)  # pytorch gpu 随机因素
    torch.cuda.manual_seed_all(seed)  # pytorch gpu 随机因素
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    mean_loss = torch.zeros(1).to(device)
    correct_num = torch.zeros(1).to(device)
    total_num = len(data_loader.dataset)

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        print(' true labels: ', labels)
        pred_fc = model(images)

        preds = torch.max(pred_fc, dim=-1)[-1]
        print(' pred labels: ', preds)

        correct_num += preds.eq(labels).sum()

        loss = loss_function(pred_fc, labels)

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

    mean_acc = correct_num / total_num
    return mean_loss.item(), mean_acc.item()


# path ='data'
# mean , std = calculate_mean_std(path, cal=True)
# print(mean, std)