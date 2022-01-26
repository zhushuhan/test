import os
import shutil

root = './med_slice_crop'

dir_out = './data/med_large'

for sample in os.listdir(root):
    out = dir_out + '/{}'.format(sample)
    if not os.path.exists(out):
        os.makedirs(out)
    sample_path = os.path.join(root, sample)
    img_lst = os.listdir(sample_path)
    sample_len = len(img_lst)
    img_num = len(os.listdir(sample_path))
    for i in range(0, img_num):
        img = img_lst[i]
        img_path = os.path.join(sample_path, img)
        move_path = os.path.join(out, img)
        if (i%5 == 0 or img == '16.jpg'):
            shutil.copy(img_path, move_path)

