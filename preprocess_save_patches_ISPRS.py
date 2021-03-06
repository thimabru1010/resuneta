from utils import load_npy_image, get_boundary_label, get_distance_label #, data_augmentation
import numpy as np
import argparse
import os
import gc
import psutil
import cv2
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler # , LabelBinarizer
from sklearn.model_selection import train_test_split
from skimage.util.shape import view_as_windows

import tensorflow as tf

import matplotlib.pyplot as plt

def extract_patches(image, reference, patch_size, stride):
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(image,
                                             window_shape_array, step=stride))

    patches_ref = np.array(view_as_windows(reference,
                                           window_shape_ref, step=stride))

    print('Patches extraidos')
    print(patches_array.shape)
    num_row, num_col, p, row, col, depth = patches_array.shape

    print('fazendo reshape')
    patches_array = patches_array.reshape(num_row*num_col, row, col, depth)
    print(patches_array.shape)
    patches_ref = patches_ref.reshape(num_row*num_col, row, col)
    print(patches_ref.shape)

    return patches_array, patches_ref


def RGB2Categories(img_ref_rgb, label_dict):
    # Convert Reference Image in RGB to a single channel integer category
    w = img_ref_rgb.shape[0]
    h = img_ref_rgb.shape[1]
    # c = img_train_ref.shape[2]
    cat_img_train_ref = np.full((w, h), -1, dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            r = img_ref_rgb[i][j][0]
            g = img_ref_rgb[i][j][1]
            b = img_ref_rgb[i][j][2]
            rgb = (r, g, b)
            rgb_key = str(rgb)
            cat_img_train_ref[i][j] = label_dict[rgb_key]

    return cat_img_train_ref


parser = argparse.ArgumentParser()
parser.add_argument("--norm_type",
                    help="Choose type of normalization to be used", type=int,
                    default=1, choices=[1, 2, 3])
parser.add_argument("--patch_size",
                    help="Choose size of patches", type=int, default=256)
parser.add_argument("--stride",
                    help="Choose stride to be using on patches extraction",
                    type=int, default=32)
parser.add_argument("--num_classes",
                    help="Choose number of classes to convert \
                    labels to one hot encoding", type=int, default=5)
parser.add_argument("--data_aug",
                    help="Choose number of classes to convert \
                    labels to one hot encoding", action='store_true', default=False)
args = parser.parse_args()

print('='*50)
print('Parameters')
print(f'patch size={args.patch_size}')
print(f'stride={args.stride}')
print(f'Number of classes={args.num_classes} ')
print('='*50)

root_path = './DATASETS/ISPRS_npy'
# Load images
img_train_path = 'Image_Train.npy'
img_train = load_npy_image(os.path.join(root_path,
                                        img_train_path))
# Convert shape from C x H x W --> H x W x C
img_train = img_train.transpose((1, 2, 0))
# img_train_normalized = normalization(img_train)
print('Imagem RGB')
print(img_train.shape)

# Load reference
img_train_ref_path = 'Reference_Train.npy'
img_train_ref = load_npy_image(os.path.join(root_path, img_train_ref_path))
# Convert from C x H x W --> H x W x C
img_train_ref = img_train_ref.transpose((1, 2, 0))
print('Imagem de referencia')
print(img_train_ref.shape)

label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1,
              '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4}

# Convert from H x W x C --> C x H x W
binary_img_train_ref = RGB2Categories(img_train_ref, label_dict)
print(binary_img_train_ref.shape)
del img_train_ref

# stride = patch_size
patches_tr, patches_tr_ref = extract_patches(img_train,
                                             binary_img_train_ref,
                                             args.patch_size, args.stride)
print('patches extraidos!')
process = psutil.Process(os.getpid())
print('[CHECKING MEMORY]')
# print(process.memory_info().rss)
print(process.memory_percent())
del binary_img_train_ref, img_train
# print(process.memory_info().rss)
print(process.memory_percent())
gc.collect()
print('[GC COLLECT]')
print(process.memory_percent())

# Convert from B x H x W x C --> B x C x H x W
# print('[Checking channels]')
# print(patches_tr.shape)
# print(patches_tr_ref.shape)
# patches_tr = patches_tr.transpose((0, 3, 1, 2))
# patches_tr_ref = patches_tr_ref.transpose((0, 3, 1, 2))
# print(patches_tr.shape)
# print(patches_tr_ref.shape)


def create_folders(folder_path, mode='train'):
    if not os.path.exists(os.path.join(folder_path, mode)):
        # os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, mode, 'imgs'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/seg'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/bound'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/dist'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/color'))


print(f'Total number of patches: {len(patches_tr)}')

patches_tr, patches_val, patches_tr_lb, patches_val_lb = train_test_split(patches_tr, patches_tr_ref, test_size=0.2, random_state=42)

print('saving images...')
folder_path = f'./DATASETS/patch_size={args.patch_size}_' + \
            f'stride={args.stride}_norm_type={args.norm_type}_data_aug={args.data_aug}'

create_folders(folder_path, mode='train')
create_folders(folder_path, mode='val')


def filename(i):
    # return f'patch_{i}.npy'
    return str(i).zfill(6) + '.npy'


print(f'Number of train patches: {len(patches_tr)}')
print(f'Number of val patches: {len(patches_val)}')
# if args.data_aug:
#     print(f'Number of patches expected: {len(patches_tr)*5}')

# Performs the one hot encoding
# label_binarizer = LabelBinarizer()
# label_binarizer.fit(range(args.num_classes))
# b = label_binarizer.transform(a)

def save_patches(patches_tr, patches_tr_ref, folder_path, mode='train'):
    for i in tqdm(range(len(patches_tr))):
        # Expand dims (Squeeze) to receive data_augmentation. Depreceated ?
        img_aug, label_aug = np.expand_dims(patches_tr[i], axis=0), np.expand_dims(patches_tr_ref[i], axis=0)
        # label_aug_h = label_binarizer.transform(label_aug)
        # Performs the one hot encoding
        label_aug_h = tf.keras.utils.to_categorical(label_aug, args.num_classes)
        # Convert from B x H x W x C --> B x C x H x W
        # label_aug_h = label_aug_h.transpose((0, 3, 1, 2))
        for j in range(len(img_aug)):
            # Input image RGB
            # Float32 its need to train the model
            img_float = img_aug[j].astype(np.float32)
            # img_normalized = normalize_rgb(img_float, norm_type=args.norm_type)
            # np.save(os.path.join(folder_path, mode, 'imgs', filename(i*5 + j)),
            #         img_float.transpose((2, 0, 1)))
            np.save(os.path.join(folder_path, mode, 'imgs', filename(i*5 + j)),
                    img_float)
            # All multitasking labels are saved in one-hot
            # Segmentation
            np.save(os.path.join(folder_path, mode, 'masks/seg', filename(i*5 + j)),
                    label_aug_h[j].astype(np.float32))
            # Boundary
            bound_label_h = get_boundary_label(label_aug_h[j]).astype(np.float32)
            np.save(os.path.join(folder_path, mode, 'masks/bound', filename(i*5 + j)),
                    bound_label_h)
            # Distance
            dist_label_h = get_distance_label(label_aug_h[j]).astype(np.float32)
            np.save(os.path.join(folder_path, mode, 'masks/dist', filename(i*5 + j)),
                    dist_label_h)
            # Color
            # print(f'Checking if rgb img is in uint8 before hsv: {img_aug[j].dtype}')
            hsv_patch = cv2.cvtColor(img_aug[j],
                                     cv2.COLOR_RGB2HSV).astype(np.float32)
            # Float32 its need to train the model
            # hsv_patch = normalize_hsv(hsv_patch, norm_type=args.norm_type)
            # np.save(os.path.join(folder_path, mode, 'masks/color', filename(i*5 + j)),
            #         hsv_patch.transpose((2, 0, 1)))
            np.save(os.path.join(folder_path, mode, 'masks/color', filename(i*5 + j)),
                    hsv_patch)


save_patches(patches_tr, patches_tr_lb, folder_path, mode='train')
save_patches(patches_val, patches_val_lb, folder_path, mode='val')
