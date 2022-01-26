import math

from PIL import Image
import requests
import matplotlib.pyplot as plt

#import ipywidgets as widgets
#from IPython.display import display, clear_output

import torch
import os
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from hubconf import*
from util.misc import nested_tensor_from_tensor_list
torch.set_grad_enabled(False);

# COCO classes
CLASSES = [
    'background', 'tumor'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, ID, num,  prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    dir_out1 = '../../AI-Pediatric/emd_slice_res/' + '/{}'.format(ID)
    dir_out2 = '../../AI-Pediatric/emd_slice_crop/' + '/{}'.format(ID)
    if not os.path.exists(dir_out1):
        os.makedirs(dir_out1)
    if not os.path.exists(dir_out2):
        os.makedirs(dir_out2)
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig(dir_out1 + '/{}.jpg'.format(num)) 
    
    cropped = pil_img.crop((xmin, ymin, xmax, ymax))
    cropped.save(dir_out2 + '/{}.jpg'.format(num))

def predict(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    anImg = transform(im)
    data = nested_tensor_from_tensor_list([anImg])

    # propagate through the model
    outputs = model(data)
    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]

    keep = probas.max(-1).values > 0.99
    # print(probas[keep])

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    model= detr_resnet50(False, 1+1)
    state_dict = torch.load("outputs/checkpoint.pth",map_location='cpu')
    model.load_state_dict(state_dict["model"])
    model.eval()

    root_dir = '../../AI-Pediatric/emd_slice'
    for sample in os.listdir(root_dir):
        sample_path = os.path.join(root_dir, sample)
        for img in os.listdir(sample_path):
            if (img[-4:] == '.jpg'):
                img_num = img[:-4]
                img_path = os.path.join(sample_path, img)
                print("current img path:", img_path)
                im = Image.open(img_path)
                scores, boxes = predict(im,  model, transform)
                if (len(boxes) != 0):
                    plot_results(im, sample, img_num, scores, boxes)






