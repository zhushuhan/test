import os
import time
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import torchvision.models as models
from MyModel import Net
from MyDataset import BrainDataset_slice, BrainDataset_sample
from utils import read_split_slice, read_split_sample, calculate_mean_std, train_one_epoch, evaluate, setup_seed


def main(args):

    # 设置随机数种子
    setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('random seed: {}.'.format(args.seed))
    print('using {} device.'.format(device))
    print('Start Tensorboard with "tensorboard --logdir=/runs"')
    tb_writer = SummaryWriter(
        comment="{}+seed {}+random slice random val slice".format(
            args.data_path[10:], args.seed))  #tb_path保存tensorboard记录
    if os.path.exists("weights") is False:
        os.makedirs("weights")

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_slice(args.data_path)
    train_samples_path, train_samples_label, val_samples_path, val_samples_label = read_split_sample(
        args.data_path)

    mean, std = calculate_mean_std(args.data_path)

    data_transform = {
        "train":
        transforms.Compose([
            transforms.Grayscale(1),  # 转灰度图
            transforms.Resize([256, 256]),
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor(),  # 从0~255转换为0~1
            transforms.Normalize(mean, std),
        ]),
        "val":
        transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    }

    # # 1.输入为每个sample的每个slice
    # train_dataset = BrainDataset_slice(images_path=train_images_path,
    #                              images_class=train_images_label,
    #                              transform=data_transform["train"])

    # # img = transforms.ToPILImage()(train_dataset[2][0])
    # # print(img)
    # # plt.imshow(img, cmap='gray')
    # # plt.show()
    # # exit(0)

    # val_dataset = BrainDataset_slice(images_path=val_images_path,
    #                            images_class=val_images_label,
    #                            transform=data_transform["val"])

    # 2. 输入为每个sample的随机一张slice
    train_dataset = BrainDataset_sample(samples_path=train_samples_path,
                                        samples_class=train_samples_label,
                                        mode='train',
                                        transform=data_transform["train"])

    val_dataset = BrainDataset_sample(samples_path=val_samples_path,
                                      samples_class=val_samples_label,
                                      mode='val',
                                      transform=data_transform["val"])

    batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    resnet = models.resnet101(pretrained=True)
    model = Net(resnet, args.num_class)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    # optimizer =  optim.Adam(params,lr = args.lr,  weight_decay=1E-4)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (
        1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    max_val_acc = 0

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader,
                                                device)
        print("epoch {} train mean loss: {} , train accuracy: {}\n".format(
            epoch, round(train_loss, 4), round(train_acc, 4)))

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device)
        print("epoch {} val mean loss: {} , val accuracy: {}".format(
            epoch, round(val_loss, 4), round(val_acc, 4)))
        if val_acc >= max_val_acc:
            max_val_acc = val_acc
            print("current best model's acc: {}\n".format(round(
                max_val_acc, 4)))
            torch.save(
                model.state_dict(), args.weights + '/' +
                '{}+{}.pkl'.format(args.data_path[10:], args.seed))
        else:
            print("current best model's acc: {}\n".format(round(
                max_val_acc, 4)))
        tags = [
            "learning rate", "train loss", "train accuracy", "val loss",
            "val accuracy"
        ]
        tb_writer.add_scalar(tags[0], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[1], train_loss, epoch)
        tb_writer.add_scalar(tags[2], train_acc, epoch)
        tb_writer.add_scalar(tags[3], val_loss, epoch)
        tb_writer.add_scalar(tags[4], val_acc, epoch)
    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrf', type=float, default=1e-2)
    parser.add_argument('--device',
                        default='cuda',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    # 设置随机种子
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    # 数据集所在根目录
    parser.add_argument('--data_path',
                        type=str,
                        default="./dataset/large_dataset_edit")
    # 网络权重位置
    parser.add_argument('--weights',
                        type=str,
                        default='./weights',
                        help='initial weights path')

    opt = parser.parse_args()

    begin = time.time()
    main(opt)
    end = time.time()

    print("train processing finished, spent {} min".format(
        round(((end - begin) / 60), 3)))
