import torch

pretrained_weights = torch.load('detr-r50-e632da11.pth')

# pretrained_weights = torch.load('resnet34-333f7ec4.pth')

num_class = 1 + 1
pretrained_weights["model"]["class_embed.weight"].resize_(num_class+1 , 256)
pretrained_weights["model"]["class_embed.bias"].resize_(num_class+1)
torch.save(pretrained_weights, "detr-r50-%d.pth"%num_class)
