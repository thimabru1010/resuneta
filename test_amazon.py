import mxnet as mx
import numpy as np

import argparse
import os

from sklearn.metrics import confusion_matrix, f1_score, precision_score, \
    recall_score, accuracy_score, roc_auc_score, precision_recall_curve, \
    average_precision_score

import ast
import cv2
from matplotlib import cm
# from sklearn.preprocessing import StandardScaler

# isprs dataset
from resuneta.src.NormalizeDataset import Normalize

# from resuneta.models.resunet_d7_causal_mtskcolor_ddist import *
from resuneta.models.resunet_d6_causal_mtskcolor_ddist import ResUNet_d6
from resuneta.models.Unet import UNet
from resuneta.models.Unet_small import UNet_small
from utils import load_npy_image, get_boundary_label, get_distance_label, \
    check_memory, normalization, mask_no_considered
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import itertools
from metrics_amazon import compute_def_metrics, prepare4metrics


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if title == 'Metrics':
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    if title == 'Metrics':
        classes = ['F1score', 'Recall', 'Precision']
        plt.yticks(tick_marks, classes)
    else:
        plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    if title == 'Metrics':
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j],  '.2f'),
                     horizontalalignment="center", #  color="black")
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        # plt.ylabel('Metric')
        # plt.xlabel('Predicted label')
    else:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j],  '.5f'),
                     horizontalalignment="center", color="red")
                     # color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')



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


def pred_recostruction(patch_size, pred_labels, binary_img_test_ref, img_type=1,
                       cont=0):
    # Patches Reconstruction
    if img_type == 1:
        stride = patch_size

        height, width = binary_img_test_ref.shape

        num_patches_h = height // stride
        num_patches_w = width // stride
        #print(num_patches_h, num_patches_w)

        new_shape = (height, width)
        img_reconstructed = np.full(new_shape, -1, dtype=np.float32)
        # cont = 0
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
    return img_reconstructed, cont


def reconstruct_patches2tiles(tiles, mask_amazon, image_ref, patch_size, preds):
    new_image_ref = np.full((image_ref.shape[0], image_ref.shape[1]), -1)
    print('Reconstructing tiles')
    cont = 0
    for num_tile in tiles:
        # print('='*60)
        # print(num_tile)
        rows, cols = np.where(mask_amazon == num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)

        # tile_img = input_image[x1:x2+1, y1:y2+1, :]
        tile = image_ref[x1:x2+1, y1:y2+1]
        tile_rec, cont = pred_recostruction(patch_size, preds, tile, cont=cont)

        new_image_ref[x1:x2+1, y1:y2+1] = tile_rec.copy()

    return new_image_ref


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


def extract_tiles2patches(tiles, mask_amazon, input_image, image_ref, patch_size):
    patches_out = []
    labels_out = []
    check_memory()
    print('Extracting tiles')
    for num_tile in tiles:
        # print('='*60)
        # print(num_tile)
        rows, cols = np.where(mask_amazon == num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)

        tile_img = input_image[x1:x2+1, y1:y2+1, :]
        tile_ref = image_ref[x1:x2+1, y1:y2+1]

        # Extract patches for each tile
        print(tile_img.shape)
        patches_img = extract_patches(tile_img, patch_size)
        patches_ref = extract_patches(tile_ref, patch_size, img_type=2)

        print(f'Patches of tile {num_tile} extracted!')
        assert len(patches_img) == len(patches_ref), "Train: Input patches and reference \
        patches don't have the same numbers"

        #print(type(patches_img))
        # print(patches_img.shape)
        # print(patch_ref.shape)
        print(len(patches_img))
        if len(patches_img) > 0:
            patches_out.append(patches_img)
            labels_out.append(patches_ref)

        # check_memory()
        # del patches_img, patches_ref
        # print('Variables deleted')
        # check_memory()

    # print(patches_out)
    patches_out = np.concatenate(patches_out, axis=0)
    labels_out = np.concatenate(labels_out, axis=0)
    return patches_out, labels_out


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
                    help="Model's filepath .params", type=str, required=True)
parser.add_argument("--dataset_path",
                    help="Dataset directory path", type=str, required=True)
parser.add_argument("-ps", "--patch_size",
                    help="Size of Patches extracted from image and reference",
                    type=int, default=128)
parser.add_argument("--norm_type", choices=[1, 2, 3],
                    help="Types of normalization. Be sure to select the same \
                    type used in your training. 1 --> [0,1]; 2 --> [-1,1]; \
                    3 --> StandardScaler() from scikit",
                    type=int, default=1)
parser.add_argument("--num_classes",
                    help="Number of classes",
                    type=int, default=3)
parser.add_argument("--output_path",
                    help="Path to where save predictions",
                    type=str, default='results/preds_run')
parser.add_argument("--model", help="Path to where save predictions",
                    type=str, default='resuneta', choices=['resuneta', 'unet', 'unet_small'])
parser.add_argument("--groups", help="Groups to be used in convolutions",
                    type=int, default=1)
args = parser.parse_args()

# Test model
# root_path = './DATASETS/Amazon_npy_corrigido'
root_path = args.dataset_path

# # Load images
img_t1_path = 'cut_raster_2018_ok.npy'
img_t2_path = 'cut_raster_2019_ok.npy'
img_t1 = load_npy_image(os.path.join(root_path, img_t1_path)).astype(np.float32)
img_t2 = load_npy_image(os.path.join(root_path, img_t2_path)).astype(np.float32)
# Convert shape from C x H x W --> H x W x C
img_t1 = img_t1.transpose((1, 2, 0))
img_t2 = img_t2.transpose((1, 2, 0))
# img_train_normalized = normalization(img_train)
print('Image 7 bands')
print(img_t1.shape)
print(img_t2.shape)
# Concatenation of images
input_image = np.concatenate((img_t1, img_t2), axis=-1)
# input_image = input_image[:6100, :6600]
input_image = input_image[:3328, :5248]
h_, w_, channels = input_image.shape
print(f"Input image shape: {input_image.shape}")
check_memory()
scaler, input_image = normalization(input_image, norm_type=args.norm_type)
check_memory()

# Load deforastation reference ---------------------------------------------
'''
    0 --> No deforastation
    1 --> Deforastation
'''
img_ref_path = 'cut_ref_2019_ok.npy'
image_ref = load_npy_image(os.path.join(root_path,
                                        'labels', img_ref_path)).astype(np.float32)
# Clip to fit tiles of your specific image
# image_ref = image_ref[:6100, :6600]
image_ref = image_ref[:3328, :5248]
# image_ref[img_mask_ref == -99] = -1
print(f"Image reference shape: {image_ref.shape}")

# Load past deforastation reference ----------------------------------------
# past_ref1_path = 'binary_clipped_2013_2018.npy'
past_ref1_path = 'cut_ref_1988_2007_ok.npy'
past_ref1 = load_npy_image(os.path.join(root_path,
                                        'labels', past_ref1_path)).astype(np.float32)
# past_ref2_path = 'binary_clipped_1988_2012.npy'
past_ref2_path = 'cut_ref_2008_2018_ok.npy'
past_ref2 = load_npy_image(os.path.join(root_path,
                                        'labels', past_ref2_path)).astype(np.float32)
past_ref_sum = past_ref1 + past_ref2
# Clip to fit tiles of your specific image
# past_ref_sum = past_ref_sum[:6100, :6600]
past_ref_sum = past_ref_sum[:3328, :5248]
print(f"Past reference shape: {past_ref_sum.shape}")

#  Creation of buffer
buffer = 2
# Gather past deforestation with actual deforastation
'''
    0 --> No deforastation
    1 --> Deforastation
    2 --> Past deforastation (No considered)
'''
final_mask = mask_no_considered(image_ref, buffer, past_ref_sum)

check_memory()
del img_t1, img_t2, past_ref1, past_ref2 #  , image_ref
print('Images deleted!')
check_memory()

# Put the patch size according to you training here
# input_patches = extract_patches(input_image, args.patch_size, img_type=1)
# ref_patches = extract_patches(final_mask, args.patch_size, img_type=2)

# # Separate per tiles
# tile_number = np.ones((1040, 1680))
# mask_c_1 = np.concatenate((tile_number, 2*tile_number, 3*tile_number), axis=1)
# mask_c_2 = np.concatenate((4*tile_number, 5*tile_number, 6*tile_number), axis=1)
# mask_c_3 = np.concatenate((7*tile_number, 8*tile_number, 9*tile_number), axis=1)
# mask_c_4 = np.concatenate((10*tile_number, 11*tile_number, 12*tile_number), axis=1)
# mask_c_5 = np.concatenate((13*tile_number, 14*tile_number, 15*tile_number), axis=1)
# mask_tiles = np.concatenate((mask_c_1, mask_c_2, mask_c_3, mask_c_4, mask_c_5), axis=0)
#
# tst1 = 5 # 357
# tst2 = 11 # 257
# tst3 = 14
# tst4 = 15
# tst5 = 13
# tst6 = 12
#
# # Testing tiles
# tst_tiles = [tst1, tst2, tst3, tst4]
# input_patches, ref_patches = extract_tiles2patches(tst_tiles, mask_tiles, input_image,
#                                                    final_mask, args.patch_size)

input_patches = extract_patches(input_image, args.patch_size, img_type=1)
ref_patches = extract_patches(final_mask, args.patch_size, img_type=2)

assert len(input_patches) == len(ref_patches), "Input patches and Reference patches have a different lenght"

# Load model
ctx = mx.gpu(0)
# ctx = mx.cpu()
if args.model == 'resuneta':
    nfilters_init = 32
    args.use_multitasking = True
    # Nbatch = 8
    net = ResUNet_d6('amazon', nfilters_init, args.num_classes,
                     patch_size=args.patch_size,
                     multitasking=args.use_multitasking)
elif args.model == 'unet':
    net = UNet(args.num_classes, groups=args.groups, nfilter=64)
    args.use_multitasking = False
elif args.model == 'unet_small':
    net = UNet_small(args.num_classes, groups=args.groups, nfilter=32)
    args.use_multitasking = False

net.initialize()
net.collect_params().initialize(force_reinit=True, ctx=ctx)
net.load_parameters(args.model_path, ctx=ctx)
# net.summary(mx.nd.random.uniform(shape=(Nbatch, 3, 256, 256)))
net.hybridize()

tnorm = Normalize()


preds = []
# seg_preds = np.zeros((len(datagen), Nbatch, args.num_classes, args.patch_size, args.patch_size))
seg_preds = []
seg_preds2 = []
bound_preds = []
dist_preds = []
color_preds = []
seg_refs = []
patches_test = []

for i in tqdm(range(len(input_patches))):
    img_float = input_patches[i].astype(np.float32)
    # plt.imshow(img.astype(np.uint8))
    # plt.show()
    # img_reshaped = img_float.reshape((img_float.shape[0] * img_float.shape[1]),
    #                                img_float.shape[2])
    # img_normed = scaler.transform(img_reshaped)
    # img_float = img_normed.reshape(img_float.shape[0], img_float.shape[1], img_float.shape[2])
    img_float = img_float.transpose((2, 0, 1))
    img_normed = mx.ndarray.array(img_float)
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
        # print(preds1)
        seg_preds.append(preds1.asnumpy())


print('='*40)
print('[TEST]')

if args.use_multitasking:
    seg_preds = gather_preds(seg_preds)
    print(f'seg shape: {seg_preds.shape}')
    seg_preds_def = seg_preds[:, :, :, 1]
    seg_pred = np.argmax(seg_preds, axis=-1)
    print(f'seg shape argmax: {seg_pred.shape}')
    patches_pred = [seg_preds, gather_preds(bound_preds), gather_preds(dist_preds), gather_preds(color_preds)]
else:
    seg_preds = gather_preds(seg_preds)
    print(f'seg shape: {seg_preds.shape}')
    seg_preds_def = seg_preds[:, :, :, 1]
    print(seg_preds_def.shape)
    seg_pred = np.argmax(seg_preds, axis=-1)
    print(f'seg shape argmax: {seg_pred.shape}')

# Metrics
# print(ref_patches.shape)
# print(seg_pred.shape)
true_labels = np.reshape(ref_patches, (ref_patches.shape[0] *
                                            ref_patches.shape[1] *
                                            ref_patches.shape[2]))

predicted_labels = np.reshape(seg_pred, (seg_pred.shape[0] *
                                         seg_pred.shape[1] *
                                         seg_pred.shape[2]))


metrics = compute_metrics(true_labels, predicted_labels)
confusion_matrix = confusion_matrix(true_labels, predicted_labels)
np.set_printoptions(precision=2)
class_names = ['No def', 'Def', 'Past def']

print('Confusion  matrix \n', confusion_matrix)
print()
print('Accuracy: ', metrics[0])
print('F1score: ', metrics[1])
print('Recall: ', metrics[2])
print('Precision: ', metrics[3])

fig = plt.figure()
plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=True)
# plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(os.path.join(args.output_path, 'confusion_matrix.jpg'), dpi=300,
            bbox_inches="tight")
plt.show()
plt.close()

print(metrics[1].shape)
print(metrics[2].shape)
print(metrics[3].shape)
new_metrics = np.stack((metrics[1], metrics[2], metrics[3]), axis=0)
fig = plt.figure()
plot_confusion_matrix(new_metrics, classes=class_names, title='Metrics')
# plt.gcf().subplots_adjust(bottom=0.2)
fig.savefig(os.path.join(args.output_path, 'metrics.jpg'), dpi=300,
            bbox_inches="tight")
plt.show()
plt.close()

check_memory()
del predicted_labels, true_labels
check_memory()

# Dictionary used in training
# -1 --> tiles not used
label_dict = {'(255, 255, 255)': -1, '(0, 255, 0)': 0, '(255, 0, 0)': 1, '(0, 0, 255)': 2}

# Reconstruct entire image segmentation predction
print(final_mask.shape)
final_mask = np.expand_dims(final_mask, axis=-1)
print(final_mask.shape)
# img_reconstructed = pred_recostruction(args.patch_size, seg_pred,
#                                        final_mask, img_type=1)

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# print(tst_tiles)
# tst_tiles.reverse()
# print(tst_tiles)
print(np.squeeze(final_mask, axis=-1).shape)
# print(seg_preds_def)
img_reconstructed, _ = pred_recostruction(args.patch_size, seg_preds_def,
                                       np.squeeze(final_mask, axis=-1))
# img_reconstructed = reconstruct_patches2tiles(tst_tiles, mask_tiles, final_mask,
#                                               args.patch_size, seg_pred)
# img_reconstructed_rgb = convert_preds2rgb(img_reconstructed,
#                                           label_dict)
#
#
# plt.imsave(os.path.join(args.output_path, 'pred_seg_reconstructed.jpeg'),
#            img_reconstructed_rgb)

print('Shapes')
print(img_reconstructed.shape)
# print(img_reconstructed)
fig, axes = plt.subplots(nrows=1, ncols=2,
                          figsize=(10, 5))
axes[0].set_title('Reference')
axes[1].set_title('Def pred')

print(final_mask.shape)
# axes[0].imshow(final_mask[:, :, 0])
axes[0].imshow(image_ref)
im = axes[1].imshow(img_reconstructed)
colorbar(im, axes[1], fig)
plt.show()
plt.close()
fig.savefig(os.path.join(args.output_path, 'seg_pred_def&ref.jpg'))

plt.imsave('seg_pred_def2.jpeg', img_reconstructed)

# Metrics
final_mask = np.squeeze(final_mask, axis=-1)
ref1 = np.ones_like(final_mask).astype(np.float32)
ref1[final_mask == 2] = 0
# TileMask = mask_amazon_ts * ref1
TileMask = ref1
GTTruePositives = (final_mask == 1)

Npoints = 100
Pmax = np.max(img_reconstructed[GTTruePositives * TileMask == 1])
# ProbList = np.linspace(Pmax, 0, Npoints)
ProbList = np.arange(start=0, stop=1, step=0.02).tolist()
ProbList.reverse()
# print(final_mask.shape)
print(ProbList)
# ProbList = [0.2, 0.5, 0.8]
def_metrics, prec, recall, tpr, fpr = compute_def_metrics(ProbList, img_reconstructed, final_mask)
print('Image Saved!')

# Calculates AUC ROC and AP for precisionXrecall

refs, probs = prepare4metrics(img_reconstructed, final_mask)

roc_auc = roc_auc_score(refs, probs)
ap = average_precision_score(refs, probs)

fig_pr = plt.figure()
# plt.plot(fpr, tpr, color='darkorange',
#         lw=lw, label=f'ROC curve (area = {auc*100:.2f}%)')

# Precision x Recall curve
plt.plot(recall, prec, color='darkorange', lw=2)
# plt.plot([0, 0], [1, 0], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision x Recall (mAP: {ap:.2f})')
# plt.legend(loc="lower right")

fig_roc = plt.figure()
# ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC curve (AUC: {roc_auc:.2f})%')
plt.legend(loc="lower right")

plt.show()
plt.close()

fig_pr.savefig(os.path.join(args.output_path, 'precisionXrecall.jpg'),
               dpi=300)

fig_roc.savefig(os.path.join(args.output_path, 'ROC.jpg'),
                dpi=300)

label_name = {0: 'No def', 1: 'Actual def', 2: 'Past def'}

# Visualize inference per class
if args.use_multitasking:

    for i in range(len(input_patches)):
        print(f'Patch: {i}')
        # Plot predictions for each class and each task; Each row corresponds to a
        # class and has its predictions of each task
        fig1, axes = plt.subplots(nrows=args.num_classes, ncols=8,
                                  figsize=(30, 30))
        # fig1, axes = plt.subplots(nrows=args.num_classes, ncols=8)
        # fig1.tight_layout()
        image = input_patches[i]  # .astype(np.uint8)
        image_reshaped = image.reshape((image.shape[0] * image.shape[1]),
                                       image.shape[2])
        image_unnormalized = scaler.inverse_transform(image_reshaped)
        img = image_unnormalized.reshape(image.shape[0], image.shape[1], image.shape[2])
        img_t1 = img[:, :, 0:7]
        img_t2 = img[:, :, 7:]
        # Convert from BGR 2 RGB
        img_t1_bgr = img_t1[:, :, 1:4].astype(np.uint8)
        # img_t1_rgb = img_t1_bgr[:, :, ::-1]
        img_t1_rgb = img_t1_bgr

        img_t2_bgr = img_t2[:, :, 1:4].astype(np.uint8)
        # img_t2_rgb = img_t2_bgr[:, :, ::-1]
        img_t2_rgb = img_t2_bgr
        print(img_t1_rgb.shape)
        print(img_t2_rgb.shape)
        img_both_rgb = np.concatenate((img_t1_rgb, img_t2_rgb), axis=-1)
        # print(img)
        # img = tnorm.restore(img)

        img_ref = ref_patches[i]
        img_ref_h = to_categorical(img_ref, args.num_classes)
        bound_ref_h = get_boundary_label(img_ref_h)
        dist_ref_h = get_distance_label(img_ref_h)
        # Put the first plot as the patch to be observed on each row
        for n_class in range(args.num_classes):
            axes[n_class, 0].imshow(img_t1_rgb)
            axes[n_class, 1].imshow(img_t2_rgb)
            # Loop the columns to display each task prediction and reference
            # Remeber we are not displaying color preds here, since this task
            # do not use classes
            # Multiply by 2 cause its always pred and ref side by side
            for task in range(len(patches_pred) - 1):
                task_pred = patches_pred[task]
                col_ref = (task + 1)*2 + 1
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
        axes[0, 0].set_title('Patch T1')
        axes[0, 1].set_title('Patch T2')

        axes[0, 2].set_title('Seg Ref')
        axes[0, 3].set_title('Seg Pred')
        axes[0, 4].set_title('Bound Ref')
        axes[0, 5].set_title('Bound Pred')
        axes[0, 6].set_title('Dist Ref')
        axes[0, 7].set_title('Dist Pred')

        for n_class in range(args.num_classes):
            axes[n_class, 0].set_ylabel(f'{label_name[n_class]}')

        # Color
        fig2, axes_c = plt.subplots(nrows=1, ncols=5, figsize=(10, 5))
        print(axes_c.shape)
        axes_c[0].set_title('Original T1')
        axes_c[0].imshow(img_t1_rgb)

        axes_c[1].set_title('Pred T1 HSV in RGB')
        task = 3
        hsv_pred = patches_pred[task][i]
        # print(f'HSV max {i}: {hsv_patch.max()}, HSV min: {hsv_patch.min()}')
        # As long as the normalization process was just img = img / 255
        # Talvez de problemas ao mudar para np uint8
        hsv_pred_t1 = (hsv_pred[:, :, :3] * np.array([179, 255, 255])).astype(np.uint8)
        rgb_pred_t1 = cv2.cvtColor(hsv_pred_t1, cv2.COLOR_HSV2RGB)
        axes_c[1].imshow(rgb_pred_t1)

        axes_c[2].set_title('Original T2')
        axes_c[2].imshow(img_t2_rgb)

        axes_c[3].set_title('Pred T2 HSV in RGB')
        hsv_pred_t2 = (hsv_pred[:, :, 3:] * np.array([179, 255, 255])).astype(np.uint8)
        rgb_pred_t2 = cv2.cvtColor(hsv_pred_t1, cv2.COLOR_HSV2RGB)
        axes_c[3].imshow(rgb_pred_t2)

        axes_c[4].set_title('Difference between both')
        print(img_t1_rgb.min(), img_t1_rgb.max())
        hsv_t1_label = cv2.cvtColor(img_t1_rgb, cv2.COLOR_RGB2HSV)
        hsv_t2_label = cv2.cvtColor(img_t2_rgb, cv2.COLOR_RGB2HSV)
        hsv_label = np.concatenate((hsv_t1_label, hsv_t2_label), axis=-1)

        hsv_patch = np.concatenate((hsv_pred_t1, hsv_pred_t2), axis=-1)

        diff = np.mean(hsv_patch - hsv_label, axis=-1)
        diff = 2*(diff-diff.min())/(diff.max()-diff.min()) - np.ones_like(diff)
        im = axes_c[4].imshow(diff, cmap=cm.Greys_r)
        colorbar(im, axes_c[4], fig2)


        plt.tight_layout()

        fig2.savefig(os.path.join(args.output_path, f'pred{i}_color.jpg'))
        plt.subplots_adjust(top=0.99, left=0.05, hspace=0.01, wspace=0.4)
        plt.show()
        fig1.savefig(os.path.join(args.output_path, f'pred{i}_classes.jpg'),
                     dpi=300)
        plt.close()
