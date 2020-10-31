from utils import load_npy_image
import tensorflow as tf
import numpy as np

from utils import get_boundary_label, get_distance_label
import argparse
import os

from skimage.util.shape import view_as_windows

import gc
import psutil
import cv2
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.morphology import disk
import skimage


def normalization(image, norm_type=1):
    image = image.reshape((image.shape[0] * image.shape[1]),
                                   image.shape[2])
    if(norm_type == 1):
        scaler = StandardScaler()
    if(norm_type == 2):
        scaler = MinMaxScaler(feature_range=(0, 1))
    if(norm_type == 3):
        scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(image)
    # print(scaler.mean_)
    # print(scaler.)
    # image_normalized = scaler.fit_transform(image_reshaped)
    # image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
    return scaler


def create_folders(folder_path, mode='train'):
    if not os.path.exists(os.path.join(folder_path, mode)):
        # os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, mode, 'imgs'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/seg'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/bound'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/dist'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/color'))


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

    # print('fazendo reshape')
    check_memory()
    del image, reference
    check_memory()
    patches_array = patches_array.reshape(num_row*num_col, row, col, depth)
    # print(patches_array.shape)
    patches_ref = patches_ref.reshape(num_row*num_col, row, col)
    # print(patches_ref.shape)

    return patches_array, patches_ref


def count_deforastation(image_ref, image_mask_ref):
    total_no_def = 0
    total_def = 0

    # Make this to count the deforastation area
    image_ref[img_mask_ref == -99] = -1

    total_no_def += len(image_ref[image_ref == 0])
    total_def += len(image_ref[image_ref == 1])
    # Print number of samples of each class
    print('Total no-deforestaion class is {}'.format(len(image_ref[image_ref == 0])))
    print('Total deforestaion class is {}'.format(len(image_ref[image_ref == 1])))
    print('Percentage of deforestaion class is {:.2f}'.format((len(image_ref[image_ref == 1])*100)/len(image_ref[image_ref == 0])))

    image_ref[img_mask_ref == -99] = 0


def mask_no_considered(image_ref, buffer, past_ref):
    '''
        Creation of buffer for pixel no considered
    '''
    image_ref_ = image_ref.copy()
    im_dilate = skimage.morphology.dilation(image_ref_, disk(buffer))
    outer_buffer = im_dilate - image_ref_
    outer_buffer[outer_buffer == 1] = 2
    # 1 deforestation, 2 past deforastation
    final_mask = image_ref_ + outer_buffer
    final_mask[past_ref == 1] = 2
    return final_mask


def filter_patches(patches_img, patches_ref, percent):
    filt_patches_img = []
    filt_patches_ref = []
    for i in range(len(patches_img)):
        unique, counts = np.unique(patches_ref[i], return_counts=True)
        counts_dict = dict(zip(unique, counts))
        if 0 not in counts_dict.keys():
            counts_dict[0] = 0
        if 1 not in counts_dict.keys():
            counts_dict[1] = 0
        if 2 not in counts_dict.keys():
            counts_dict[2] = 0
        # print(counts_dict)
        if -1 in counts_dict.keys():
            continue
        deforastation = counts_dict[1] / (counts_dict[0] + counts_dict[1] + counts_dict[2])
        if deforastation * 100 > percent:
            filt_patches_img.append(patches_img[i])
            filt_patches_ref.append(patches_ref[i])

    print(len(filt_patches_img))
    print(type(filt_patches_img))
    # print(type(filt_patches_img[0]))
    if len(filt_patches_img) > 0:
        filt_patches_img = np.stack(filt_patches_img, axis=0)
        print(type(filt_patches_img))
        filt_patches_ref = np.stack(filt_patches_ref, axis=0)
        print(filt_patches_img.shape)
        print(filt_patches_ref.shape)
    return filt_patches_img, filt_patches_ref


def extract_patches2(img, img_ref, patch_size, stride, percent):
    # Extract patches manually

    height, width, channel = img.shape
    #print(height, width)

    num_patches_h = height // stride
    num_patches_w = width // stride
    #print(num_patches_h, num_patches_w)

    # new_shape = (num_patches_h*num_patches_w, patch_size, patch_size, channel)
    # new_shape_ref = (num_patches_h*num_patches_w, patch_size, patch_size)
    # patches_img = np.zeros(new_shape)
    # patches_ref = np.zeros(new_shape_ref)
    patches_img = []
    patches_ref = []
    n_patch = 0
    # rows
    for h in range(num_patches_h):
        # columns
        for w in range(num_patches_w):
            # patch_img = img[h*stride:(h+1)*stride, w*stride:(w+1)*stride, :]
            # patch_ref = img_ref[h*stride:(h+1)*stride, w*stride:(w+1)*stride]
            patch_img = img[h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size, :]
            patch_ref = img_ref[h*patch_size:(h+1)*patch_size, w*patch_size:(w+1)*patch_size]
            # print(patch_img.shape)
            # print(patch_ref.shape)
            patch_shape = patch_img.shape
            if (patch_shape[0], patch_shape[1]) == (patch_size, patch_size):
                print(patch_img.shape)
                print(patch_ref.shape)
                unique, counts = np.unique(patch_ref, return_counts=True)
                counts_dict = dict(zip(unique, counts))
                if 0 not in counts_dict.keys():
                    counts_dict[0] = 0
                if 1 not in counts_dict.keys():
                    counts_dict[1] = 0
                if 2 not in counts_dict.keys():
                    counts_dict[2] = 0
                # print(counts_dict)
                if -1 in counts_dict.keys():
                    continue
                deforastation = counts_dict[1] / (counts_dict[0] + counts_dict[1] + counts_dict[2])
                if deforastation * 100 > percent:
                    # patches_img[n_patch] = img[h*stride:(h+1)*stride, w*stride:(w+1)*stride, :]
                    # patches_ref[n_patch] = img_ref[h*stride:(h+1)*stride, w*stride:(w+1)*stride]
                    patches_img.append(patch_img)
                    patches_ref.append(patch_ref)

            # n_patch += 1

    if len(patches_img) > 0:
        filt_patches_img = np.stack(patches_img, axis=0)
        # print(type(filt_patches_img))
        filt_patches_ref = np.stack(patches_ref, axis=0)
        # print(filt_patches_img.shape)
        # print(filt_patches_ref.shape)
        return filt_patches_img, filt_patches_ref
    else:
        print("Error: Couldn't extract patches." +
              "Maybe there wasn't enough deforastation or " +
              "it was out of right region (with -1)")
        return [], []


def extract_tiles2patches(tiles, mask_amazon, input_image, image_ref, patch_size,
                          stride, percent):
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

        tile_img = input_image[x1:x2+1,y1:y2+1,:]
        tile_ref = image_ref[x1:x2+1,y1:y2+1]
        # Check deforastation percentage for each tile
        unique, counts = np.unique(tile_ref, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        # print(counts_dict)
        if 0 not in counts_dict.keys():
            counts_dict[0] = 0
        if 1 not in counts_dict.keys():
            counts_dict[1] = 0
        if 2 not in counts_dict.keys():
            counts_dict[2] = 0
        # deforastation = counts_dict[1] / (counts_dict[0] + counts_dict[1] + counts_dict[2])
        # print(f"Deforastation of tile {num_tile}: {deforastation * 100}")
        # Extract patches for each tile
        print(tile_img.shape)
        # patches_img, patches_ref = extract_patches(tile_img, tile_ref, patch_size,
        #                                             stride)
        patches_img, patches_ref = extract_patches2(tile_img, tile_ref, patch_size,
                                                   stride, percent)
        print(f'Patches of tile {num_tile} extracted!')
        assert len(patches_img) == len(patches_ref), "Train: Input patches and reference \
        patches don't have the same numbers"
        # patches_img, patches_ref = filter_patches(patches_img, patches_ref, percent)

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


def show_deforastation_per_tile(tiles, mask_amazon, image_ref):
    defs = []
    for num_tile in tiles:
        print('='*60)
        print(f'Tile: {num_tile}')
        rows, cols = np.where(mask_amazon == num_tile)
        x1 = np.min(rows)
        y1 = np.min(cols)
        x2 = np.max(rows)
        y2 = np.max(cols)

        tile_ref = image_ref[x1:x2+1, y1:y2+1]
        # Check deforastation percentage for each tile
        unique, counts = np.unique(tile_ref, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        print(counts_dict)
        if 0 not in counts_dict.keys():
            counts_dict[0] = 0
        if 1 not in counts_dict.keys():
            counts_dict[1] = 0
        if 2 not in counts_dict.keys():
            counts_dict[2] = 0
        deforastation = counts_dict[1] / (counts_dict[0] + counts_dict[1] + counts_dict[2])
        print(f"Deforastation of tile {num_tile}: {deforastation * 100}%")
        defs.append(deforastation*100)
    print('-'*60)
    print(f'Total deforastation per tile: {sum(defs)}')
    print(sorted(zip(defs, tiles), reverse=True))
    print('='*60)

def filename(i):
    return f'patch_{i}.npy'


def save_patches(patches_tr, patches_tr_ref, folder_path, scaler, mode='train'):
    for i in tqdm(range(len(patches_tr))):
        # Expand dims (Squeeze) to receive data_augmentation. Depreceated ?
        img_aug, label_aug = np.expand_dims(patches_tr[i], axis=0), np.expand_dims(patches_tr_ref[i], axis=0)
        # label_aug_h = label_binarizer.transform(label_aug)
        # Performs the one hot encoding
        label_aug_h = tf.keras.utils.to_categorical(label_aug, args.num_classes)
        # Convert from B x H x W x C --> B x C x H x W
        # label_aug_h = label_aug_h.transpose((0, 3, 1, 2))
        for j in range(len(img_aug)):
            # Input image 7 bands of Staelite
            # Float32 its need to train the model
            img_float = img_aug[j].astype(np.float32)
            img_reshaped = img_float.reshape((img_float.shape[0] * img_float.shape[1]),
                                           img_float.shape[2])
            img_normed = scaler.transform(img_reshaped)
            img_float = img_normed.reshape(img_float.shape[0], img_float.shape[1], img_float.shape[2])
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
            # Get only BGR from Aerial Image
            hsv_patch = cv2.cvtColor(img_aug[j][:, :, 1:4].astype(np.uint8),
                                     cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_patch = hsv_patch * np.array([1./179, 1./255, 1./255])
            # Float32 its need to train the model
            np.save(os.path.join(folder_path, mode, 'masks/color', filename(i*5 + j)),
                    hsv_patch)


def check_memory():
    process = psutil.Process(os.getpid())
    print('-'*50)
    print('[CHECKING MEMORY]')
    # print(process.memory_info().rss)
    print(process.memory_percent())
    # print(process.memory_info().rss)
    gc.collect()
    print('[GC COLLECT]')
    print(process.memory_percent())
    print('-'*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm_type",
                        help="Choose type of normalization to be used",
                        type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--patch_size",
                        help="Choose size of patches", type=int, default=256)
    parser.add_argument("--stride",
                        help="Choose stride to be using on patches extraction",
                        type=int, default=32)
    parser.add_argument("--num_classes",
                        help="Choose number of classes to convert \
                        labels to one hot encoding", type=int, default=3)
    parser.add_argument("--data_aug",
                        help="Choose number of classes to convert \
                        labels to one hot encoding", action='store_true', default=False)
    parser.add_argument("--def_percent",
                        help="Choose minimum percentage of Deforastation",
                        type=int, default=5)
    args = parser.parse_args()

    print('='*50)
    print('Parameters')
    print(f'patch size={args.patch_size}')
    print(f'stride={args.stride}')
    print(f'Number of classes={args.num_classes} ')
    print(f'Norm type: {args.norm_type}')
    print(f'Using data augmentation? {args.data_aug}')
    print('='*50)

    root_path = './DATASETS/Amazon_npy'
    # Load images --------------------------------------------------------------
    img_t1_path = 'clipped_raster_004_66_2018.npy'
    img_t2_path = 'clipped_raster_004_66_2019.npy'
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
    input_image = input_image[:6100, :6600]
    h_, w_, channels = input_image.shape
    print(f"Input image shape: {input_image.shape}")
    check_memory()
    scaler = normalization(input_image)
    check_memory()

    # Load Mask area -----------------------------------------------------------
    # Mask constains exactly location of region since the satelite image
    # doesn't fill the entire resolution (Kinda rotated with 0 around)
    img_mask_ref_path = 'mask_ref.npy'
    img_mask_ref = load_npy_image(os.path.join(root_path, img_mask_ref_path)).astype(np.float32)
    img_mask_ref = img_mask_ref[:6100, :6600]
    print(f"Mask area reference shape: {img_mask_ref.shape}")

    # Load deforastation reference ---------------------------------------------
    '''
        0 --> No deforastation
        1 --> Deforastation
    '''
    image_ref = load_npy_image(os.path.join(root_path,
                                            'labels/binary_clipped_2019.npy')).astype(np.float32)
    # Clip to fit tiles of your specific image
    image_ref = image_ref[:6100, :6600]
    # image_ref[img_mask_ref == -99] = -1
    print(f"Image reference shape: {image_ref.shape}")

    count_deforastation(image_ref, img_mask_ref)

    # Load past deforastation reference ----------------------------------------
    past_ref1 = load_npy_image(os.path.join(root_path,
                                            'labels/binary_clipped_2013_2018.npy')).astype(np.float32)
    past_ref2 = load_npy_image(os.path.join(root_path,
                                            'labels/binary_clipped_1988_2012.npy')).astype(np.float32)
    past_ref_sum = past_ref1 + past_ref2
    # Clip to fit tiles of your specific image
    past_ref_sum = past_ref_sum[:6100, :6600]
    # past_ref_sum[img_mask_ref==-99] = -1
    # Doing the sum, there are some pixels with value 2 (Case when both were deforastation).
    # past_ref_sum[past_ref_sum == 2] = 1
    # Same thing for background area (different from no deforastation)
    # past_ref_sum[past_ref_sum==-2] = -1
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

    final_mask[img_mask_ref == -99] = -1
    unique, counts = np.unique(final_mask, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print(f'Class pixels of final mask: {counts_dict}')
    deforastation = counts_dict[1] / (counts_dict[0] + counts_dict[1] + counts_dict[2])
    print(f"Total Deforastation: {deforastation * 100}%")

    # Calculates weights for weighted cross entropy
    total_pixels = counts_dict[0] + counts_dict[1] + counts_dict[2]
    weight0 = total_pixels / counts_dict[0]
    weight1 = total_pixels / counts_dict[1]

    check_memory()
    del img_t1, img_t2, image_ref, past_ref1, past_ref2
    print('Images deleted!')
    check_memory()


    # Mask with tiles
    # Divide tiles in 5 rows and 3 columns. Total = 15 tiles
    # tile.shape --> (6100/5, 6600/3) = (1220, 2200)
    # [NEW] Divide tiles in 5 rows and 5 columns. Total = 25 tiles
    # [NEW] tile.shape --> (6100/5, 6600/5) = (1220, 1320)
    # tile_number = np.ones((1220, 1320))
    # mask_c_1 = np.concatenate((tile_number, 2*tile_number, 3*tile_number, 16*tile_number, 17*tile_number), axis=1)
    # mask_c_2 = np.concatenate((4*tile_number, 5*tile_number, 6*tile_number, 18*tile_number, 19*tile_number), axis=1)
    # mask_c_3 = np.concatenate((7*tile_number, 8*tile_number, 9*tile_number, 20*tile_number, 21*tile_number), axis=1)
    # mask_c_4 = np.concatenate((10*tile_number, 11*tile_number, 12*tile_number, 22*tile_number, 23*tile_number), axis=1)
    # mask_c_5 = np.concatenate((13*tile_number, 14*tile_number, 15*tile_number, 24*tile_number, 25*tile_number), axis=1)
    tile_number = np.ones((1220, 2200))
    mask_c_1 = np.concatenate((tile_number, 2*tile_number, 3*tile_number), axis=1)
    mask_c_2 = np.concatenate((4*tile_number, 5*tile_number, 6*tile_number), axis=1)
    mask_c_3 = np.concatenate((7*tile_number, 8*tile_number, 9*tile_number), axis=1)
    mask_c_4 = np.concatenate((10*tile_number, 11*tile_number, 12*tile_number), axis=1)
    mask_c_5 = np.concatenate((13*tile_number, 14*tile_number, 15*tile_number), axis=1)
    mask_tiles = np.concatenate((mask_c_1, mask_c_2, mask_c_3, mask_c_4, mask_c_5), axis=0)

    mask_tr_val = np.zeros((mask_tiles.shape))
    tr1 = 5
    tr2 = 8
    tr3 = 13
    tr4 = 7
    tr5 = 11
    tr6 = 1
    tr7 = 14
    tr8 = 3
    tr9 = 9

    val1 = 2
    val2 = 10
    val3 = 4
    val4 = 6
    # Putting 15 and 12 to validation but don't have expressive deforastation %
    val5 = 15
    val6 = 12

    mask_tr_val[mask_tiles == tr1] = 1
    mask_tr_val[mask_tiles == tr2] = 1
    mask_tr_val[mask_tiles == tr3] = 1
    mask_tr_val[mask_tiles == tr4] = 1
    mask_tr_val[mask_tiles == tr5] = 1
    mask_tr_val[mask_tiles == tr6] = 1
    mask_tr_val[mask_tiles == tr7] = 1
    mask_tr_val[mask_tiles == tr8] = 1
    mask_tr_val[mask_tiles == tr8] = 1
    mask_tr_val[mask_tiles == val1] = 2
    mask_tr_val[mask_tiles == val2] = 2
    mask_tr_val[mask_tiles == val3] = 2
    mask_tr_val[mask_tiles == val4] = 2
    mask_tr_val[mask_tiles == val5] = 2
    mask_tr_val[mask_tiles == val6] = 2

    all_tiles = [i for i in range(1, 16)]
    print(f'All tiles: {all_tiles}')
    # final_mask[img_mask_ref == -99] = -1
    show_deforastation_per_tile(all_tiles, mask_tiles, final_mask)

    # Trainig tiles
    tr_tiles = [tr1, tr2, tr3, tr4, tr5, tr6, tr7, tr8, tr9]

    # [TODO] Create a function to show deforestaion for all the tiles
    patches_tr, patches_tr_ref = extract_tiles2patches(tr_tiles, mask_tiles, input_image,
                                                       final_mask, args.patch_size,
                                                       args.stride, args.def_percent)

    assert len(patches_tr) == len(patches_tr_ref), "Train: Input patches and reference \
    patches don't have the same numbers"

    # Validation tiles
    val_tiles = [val1, val2, val3, val4, val5, val6]

    patches_val, patches_val_ref = extract_tiles2patches(val_tiles, mask_tiles, input_image,
                                                         final_mask, args.patch_size, args.stride,
                                                         args.def_percent)

    assert len(patches_val) == len(patches_val_ref), "Val: Input patches and reference \
    patches don't have the same numbers"

    print('patches extracted!')

    print('saving images...')
    folder_path = f'./DATASETS/patch_size={args.patch_size}_' + \
                f'stride={args.stride}_norm_type={args.norm_type}_data_aug={args.data_aug}'

    create_folders(folder_path, mode='train')
    create_folders(folder_path, mode='val')

    print(f'Number of train patches: {len(patches_tr)}')
    print(f'Number of val patches: {len(patches_val)}')

    save_patches(patches_tr, patches_tr_ref, folder_path, scaler, mode='train')
    save_patches(patches_val, patches_val_ref, folder_path, scaler, mode='val')
