import mxnet as mx
from mxnet import gluon, autograd
import mxnet.ndarray as nd
import numpy as np

import argparse
import os

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

import ast
import cv2
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb
from sklearn.preprocessing import StandardScaler

# isprs dataset
from resuneta.src.NormalizeDataset import Normalize
from resuneta.src.ISPRSDataset import ISPRSDataset
from mxnet.gluon.data.vision import transforms

# from resuneta.models.resunet_d7_causal_mtskcolor_ddist import *
from resuneta.models.resunet_d6_causal_mtskcolor_ddist import ResUNet_d6
from resuneta.models.Unet import UNet
from utils import load_npy_image, get_boundary_label, get_distance_label
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

def Test(model, patches, args):
    num_patches, weight, height, _ = patches.shape
    preds = model.predict(patches, batch_size=1)
    if args.use_multitasking:
        print('Multitasking Enabled!')
        return preds
    else:
        print(preds.shape)
        predicted_class = np.argmax(preds, axis=-1)
        print(predicted_class.shape)
        return predicted_class


def compute_metrics(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    # avg_accuracy = 100*accuracy_score(true_labels, predicted_labels, average=None)
    f1score = 100*f1_score(true_labels, predicted_labels, average=None)
    recall = 100*recall_score(true_labels, predicted_labels, average=None)
    precision = 100*precision_score(true_labels, predicted_labels, average=None)
    return accuracy, f1score, recall, precision


def pred_recostruction(patch_size, pred_labels, binary_img_test_ref, img_type=1):
    # Patches Reconstruction
    if img_type == 1:
        stride = patch_size

        height, width, _= binary_img_test_ref.shape

        num_patches_h = height // stride
        num_patches_w = width // stride
        #print(num_patches_h, num_patches_w)

        new_shape = (height, width)
        img_reconstructed = np.zeros(new_shape)
        cont = 0
        # rows
        for h in range(num_patches_h):
            # columns
            for w in range(num_patches_w):
                img_reconstructed[h*stride:(h+1)*stride, w*stride:(w+1)*stride] = pred_labels[cont]
                cont += 1
        print('Reconstruction Done!')
    if img_type == 2:
        stride = patch_size

        height, width = binary_img_test_ref.shape

        num_patches_h = height // stride
        num_patches_w = width // stride

        new_shape = (height, width, 3)
        img_reconstructed = np.zeros(new_shape)
        cont = 0
        # rows
        for h in range(num_patches_h):
            # columns
            for w in range(num_patches_w):
                img_reconstructed[h*stride:(h+1)*stride, w*stride:(w+1)*stride, :] = pred_labels[cont]
                cont += 1
        print('Reconstruction Done!')
    return img_reconstructed

def convert_preds2rgb(img_reconstructed, label_dict):
    reversed_label_dict = {value:key for (key, value) in label_dict.items()}
    print(reversed_label_dict)
    height, width = img_reconstructed.shape
    img_reconstructed_rgb = np.zeros((height, width, 3))
    for h in range(height):
        for w in range(width):
            pixel_class = img_reconstructed[h, w]
            img_reconstructed_rgb[h, w, :] = ast.literal_eval(reversed_label_dict[pixel_class])
    print('Conversion to RGB Done!')
    return img_reconstructed_rgb.astype(np.uint8)

def colorbar(mappable, ax, fig):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    # ax = mappable.axes
    # fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def convert_hsvpatches2rgb(patches):
    (amount, h, w, c) = patches.shape
    color_patches = np.full([amount, h, w, c], -1)
    for i in range(amount):
        color_patches[i] = hsv_to_rgb(patches[i])*255

    return color_patches


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

def extract_patches(img_complete, patch_size, img_type=1):
    stride = patch_size

    if img_type == 1:
        height, width, channel = img_complete.shape
    else:
        height, width = img_complete.shape

    #print(height, width)

    num_patches_h = height // stride
    num_patches_w = width // stride
    #print(num_patches_h, num_patches_w)

    if img_type == 1:
        new_shape = (num_patches_h*num_patches_w, patch_size, patch_size, channel)
    else:
        new_shape = (num_patches_h*num_patches_w, patch_size, patch_size)
    new_img = np.zeros(new_shape)
    cont = 0
    # rows
    for h in range(num_patches_h):
        # columns
        for w in range(num_patches_w):
            new_img[cont] = img_complete[h*stride:(h+1)*stride, w*stride:(w+1)*stride]
            cont += 1
    #print(cont)

    return new_img

def gather_preds(pred):
    pred = np.array(pred)
    pred = np.concatenate(pred, axis=0)
    pred = pred.transpose((0, 2, 3, 1))
    return pred

parser = argparse.ArgumentParser()
parser.add_argument("--use_multitasking",
                    help="Choose resunet-a model or not", action='store_true')
parser.add_argument("--model_path",
                    help="Model's filepath .h5", type=str, required=True)
parser.add_argument("--dataset_path",
                    help="Dataset directory path", type=str, required=True)
parser.add_argument("-ps", "--patch_size",
                    help="Size of Patches extracted from image and reference",
                    type=int, default=256)
parser.add_argument("--norm_type", choices=[1, 2, 3],
                    help="Types of normalization. Be sure to select the same \
                    type used in your training. 1 --> [0,1]; 2 --> [-1,1]; \
                    3 --> StandardScaler() from scikit",
                    type=int, default=1)
parser.add_argument("--num_classes",
                    help="Number of classes",
                    type=int, default=5)
parser.add_argument("--output_path",
                    help="Path to where save predictions",
                    type=str, default='results/preds_run')
parser.add_argument("--model", help="Path to where save predictions",
                    type=str, default='resuneta', choices=['resuneta', 'unet'])
args = parser.parse_args()

# Test model
root_path = args.dataset_path

# # Load images
img_input_path = 'Image_Test.npy'
img_input = load_npy_image(os.path.join(args.dataset_path, img_input_path)).astype(np.float32)
# # Transform the image into W x H x C shape
img_input = img_input.transpose((1, 2, 0))
print(img_input.shape)
# img_test_normalized = img_test_normalized.transpose((1, 2, 0))
# print(img_test_normalized.shape)
#
# # Load reference
img_ref_path = 'Reference_Test.npy'
img_ref = load_npy_image(os.path.join(args.dataset_path, img_ref_path))
# Transform the image into W x H x C shape
img_ref = img_ref.transpose((1, 2, 0))
print(img_ref.shape)
# img_test_ref = img_test_ref.transpose((1, 2, 0))
# print(img_test_ref.shape)

# Dictionary used in training
label_dict = {'(255, 255, 255)': 0, '(0, 255, 0)': 1,
              '(0, 255, 255)': 2, '(0, 0, 255)': 3, '(255, 255, 0)': 4}

cat_img_ref = RGB2Categories(img_ref, label_dict)

# Put the patch size according to you training here
input_patches = extract_patches(img_input, args.patch_size, img_type=1)
ref_patches = extract_patches(cat_img_ref, args.patch_size, img_type=2)

assert len(input_patches) == len(ref_patches), "Input patches and Reference patches have a different lenght"

# Load model
ctx = mx.gpu(0)
# ctx = mx.cpu()
if args.model == 'resuneta':
    nfilters_init = 32
    Nbatch = 8
    net = ResUNet_d6('tanimoto', nfilters_init, args.num_classes)
else:
    net = UNet(args.num_classes, nfilter=64)
    args.use_multitasking = False

net.initialize()
net.collect_params().initialize(force_reinit=True, ctx=ctx)
net.load_parameters(args.model_path, ctx=ctx)
# net.summary(mx.nd.random.uniform(shape=(Nbatch, 3, 256, 256)))
net.hybridize()

tnorm = Normalize()

# dataset = ISPRSDataset(root=args.dataset_path, mode='train', color=True, mtsk=True, norm=tnorm)
# datagen = gluon.data.DataLoader(dataset, batch_size=Nbatch, shuffle=False)

preds = []
# seg_preds = np.zeros((len(datagen), Nbatch, args.num_classes, args.patch_size, args.patch_size))
seg_preds = []
seg_preds2 = []
bound_preds = []
dist_preds = []
color_preds = []
seg_refs = []
patches_test = []

for i in range(len(input_patches)):
    img = input_patches[i]
    # plt.imshow(img.astype(np.uint8))
    # plt.show()
    img_normed = mx.ndarray.array(tnorm(img).transpose((2, 0, 1)))
    # plt.imshow(tnorm.restore(img_normed.asnumpy().transpose((1, 2, 0))))
    # plt.show()
    img_normed = mx.nd.expand_dims(img_normed, axis=0)

    if args.use_multitasking:
        preds1, preds2, preds3, preds4 = net(img_normed.copyto(ctx))

        seg_preds.append(preds1.asnumpy())
        bound_preds.append(preds2.asnumpy())
        dist_preds.append(preds3.asnumpy())
        color_preds.append(preds4.asnumpy())
    else:
        preds1 = net(img_normed.copyto(ctx))
        print(preds1)
        seg_preds.append(preds1.asnumpy())


print('='*40)
print('[TEST]')

if args.use_multitasking:
    seg_preds = gather_preds(seg_preds)
    print(f'seg shape: {seg_preds.shape}')
    seg_pred = np.argmax(seg_preds, axis=-1)
    print(f'seg shape argmax: {seg_pred.shape}')
    patches_pred = [seg_preds, gather_preds(bound_preds), gather_preds(dist_preds), gather_preds(color_preds)]
else:
    seg_preds = gather_preds(seg_preds)
    print(f'seg shape: {seg_preds.shape}')
    seg_pred = np.argmax(seg_preds, axis=-1)
    print(f'seg shape argmax: {seg_pred.shape}')

# Metrics
print(ref_patches.shape)
print(seg_pred.shape)
true_labels = np.reshape(ref_patches, (ref_patches.shape[0] *
                                            ref_patches.shape[1] *
                                            ref_patches.shape[2]))

predicted_labels = np.reshape(seg_pred, (seg_pred.shape[0] *
                                         seg_pred.shape[1] *
                                         seg_pred.shape[2]))

# Metrics
metrics = compute_metrics(true_labels, predicted_labels)
confusion_matrix = confusion_matrix(true_labels, predicted_labels)

print('Confusion  matrix \n', confusion_matrix)
print()
print('Accuracy: ', metrics[0])
print('F1score: ', metrics[1])
print('Recall: ', metrics[2])
print('Precision: ', metrics[3])

# Reconstruct entire image segmentation predction
img_reconstructed = pred_recostruction(args.patch_size, seg_pred,
                                       img_ref, img_type=1)
img_reconstructed_rgb = convert_preds2rgb(img_reconstructed,
                                          label_dict)

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

plt.imsave(os.path.join(args.output_path, 'pred_seg_reconstructed.jpeg'),
           img_reconstructed_rgb)

print('Image Saved!')

print(f'Input patches: {type(input_patches)}')
print(f'Input patches: {input_patches.dtype}')

# Visualize inference per class
if args.use_multitasking:

    for i in range(len(input_patches)):
        print(f'Patch: {i}')
        # Plot predictions for each class and each task; Each row corresponds to a
        # class and has its predictions of each task
        fig1, axes = plt.subplots(nrows=args.num_classes, ncols=7, figsize=(15, 10))
        img = input_patches[i].astype(np.uint8)
        # img = (img * np.array([255, 255, 255])).astype(np.uint8)
        # print(img)
        # img = tnorm.restore(img)

        img_ref = ref_patches[i]
        img_ref_h = to_categorical(img_ref, args.num_classes)
        bound_ref_h = get_boundary_label(img_ref_h)
        dist_ref_h = get_distance_label(img_ref_h)
        # Put the first plot as the patch to be observed on each row
        for n_class in range(args.num_classes):
            axes[n_class, 0].imshow(img)
            # Loop the columns to display each task prediction and reference
            # Remeber we are not displaying color preds here, since this task
            # do not use classes
            # Multiply by 2 cause its always pred and ref side by side
            for task in range(len(patches_pred) - 1):
                task_pred = patches_pred[task]
                col_ref = (task + 1)*2
                print(task_pred.shape)
                axes[n_class, col_ref].imshow(task_pred[i, :, :, n_class],
                                              cmap=cm.Greys_r)
                col = col_ref - 1
                if task == 0:
                    # Segmentation
                    axes[n_class, col].imshow(img_ref_h[:, :, n_class],
                                              cmap=cm.Greys_r)
                elif task == 1:
                    # Boundary
                    print(f' bound class: {n_class}')
                    axes[n_class, col].imshow(bound_ref_h[:, :, n_class],
                                              cmap=cm.Greys_r)
                elif task == 2:
                    # Distance Transform
                    axes[n_class, col].imshow(dist_ref_h[:, :, n_class],
                                              cmap=cm.Greys_r)
        axes[0, 0].set_title('Patch')
        axes[0, 1].set_title('Seg Ref')
        axes[0, 2].set_title('Seg Pred')
        axes[0, 3].set_title('Bound Ref')
        axes[0, 4].set_title('Bound Pred')
        axes[0, 5].set_title('Dist Ref')
        axes[0, 6].set_title('Dist Pred')

        for n_class in range(args.num_classes):
            axes[n_class, 0].set_ylabel(f'Class {n_class}')

        plt.savefig(os.path.join(args.output_path, f'pred{i}_classes.jpg'))

        # Color
        fig2, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        ax1.set_title('Original')
        ax1.imshow(img)
        ax2.set_title('Pred HSV in RGB')
        task = 3
        hsv_pred = patches_pred[task][i]
        # print(f'HSV max {i}: {hsv_patch.max()}, HSV min: {hsv_patch.min()}')
        # As long as the normalization process was just img = img / 255
        hsv_patch = (hsv_pred * np.array([179, 255, 255])).astype(np.uint8)
        rgb_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2RGB)
        ax2.imshow(rgb_patch)
        ax3.set_title('Difference between both')
        hsv_label = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        diff = np.mean(hsv_patch - hsv_label, axis=-1)
        diff = 2*(diff-diff.min())/(diff.max()-diff.min()) - np.ones_like(diff)
        im = ax3.imshow(diff, cmap=cm.Greys_r)
        colorbar(im, ax3, fig2)

        plt.savefig(os.path.join(args.output_path, f'pred{i}_color.jpg'))
        plt.show()
        plt.close()
