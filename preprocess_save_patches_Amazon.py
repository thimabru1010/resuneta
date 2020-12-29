import tensorflow as tf
import numpy as np

from utils import load_npy_image, get_boundary_label, get_distance_label, \
    data_augmentation, check_memory, normalization, mask_no_considered
import argparse
import os

from skimage.util.shape import view_as_windows

import cv2
from tqdm import tqdm

# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def compute_cva(img_t1, img_t2, th):
    # _, image_t1 = normalization(img_t1, norm_type=2)
    # _, image_t2 = normalization(img_t2, norm_type=2)

    image_t1 = img_t1
    image_t2 = img_t2

    blue_t1 = image_t1[:, :, 1]
    red_t1 = image_t1[:, :, 3]
    nir_t1 = image_t1[:, :, 4]
    swir_t1 = image_t1[:, :, 5]

    blue_t2 = image_t2[:, :, 1]
    red_t2 = image_t2[:, :, 3]
    nir_t2 = image_t2[:, :, 4]
    swir_t2 = image_t2[:, :, 5]

    # NDVI and BI index
    ndvi1 = (nir_t1-red_t1)/(nir_t1+red_t1)
    # print(ndvi1)
    bi1 = (swir_t1+red_t1)-(nir_t1+blue_t1)/(swir_t1+red_t1)+(nir_t1+blue_t1)

    ndvi2 = (nir_t2-red_t2)/(nir_t2+red_t2)
    bi2 = (swir_t2+red_t2)-(nir_t2+blue_t2)/(swir_t2+red_t2)+(nir_t2+blue_t2)
    print(np.min(ndvi1), np.max(ndvi1))
    print(np.min(bi1), np.max(bi1))

    # Calculating the change:
    S = (ndvi2-ndvi1)**2+(bi2-bi1)**2
    S1 = np.sqrt(S)
    S1_ref = S1.copy()
    S1_ref[S1 >= th] = 1
    S1_ref[S1 < th] = 0
    return S1_ref


def create_folders(folder_path, mode='train'):
    if not os.path.exists(os.path.join(folder_path, mode)):
        # os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, mode, 'imgs'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/seg'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/bound'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/dist'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/color'))
        os.makedirs(os.path.join(folder_path, mode, 'masks/cva'))


def extract_patches(image, reference, patch_size, stride, CVA_ref=None):
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape, image.shape[2])
    window_shape_ref = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(image,
                                             window_shape_array, step=stride))

    patches_ref = np.array(view_as_windows(reference,
                                           window_shape_ref, step=stride))
    if CVA_ref is not None:
        patches_cva = np.array(view_as_windows(CVA_ref,
                                               window_shape_ref, step=stride))
    else:
        patches_cva = None

    print('Patches extraidos')
    # print(patches_array.shape)
    num_row, num_col, p, row, col, depth = patches_array.shape

    # print('fazendo reshape')
    check_memory()
    del image, reference
    check_memory()
    patches_array = patches_array.reshape(num_row*num_col, row, col, depth)
    # print(patches_array.shape)
    patches_ref = patches_ref.reshape(num_row*num_col, row, col)
    # print(patches_ref.shape)
    if CVA_ref is not None:
        patches_cva = patches_cva.reshape(num_row*num_col, row, col)

    return patches_array, patches_ref, patches_cva


def count_deforastation(image_ref, image_mask_ref):
    total_no_def = 0
    total_def = 0

    # Make this to count the deforastation area
    if image_mask_ref is not None:
        image_ref[img_mask_ref == -99] = -1

    total_no_def += len(image_ref[image_ref == 0])
    total_def += len(image_ref[image_ref == 1])
    # Print number of samples of each class
    print('Total no-deforestaion class is {}'.format(len(image_ref[image_ref == 0])))
    print('Total deforestaion class is {}'.format(len(image_ref[image_ref == 1])))
    print('Percentage of deforestaion class is {:.2f}'.format((len(image_ref[image_ref == 1])*100)/len(image_ref[image_ref == 0])))

    if image_mask_ref is not None:
        image_ref[img_mask_ref == -99] = 0


def filter_patches(patches_img, patches_ref, patches_cva, percent):
    filt_patches_img = []
    filt_patches_ref = []
    filt_patches_cva = []
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
            filt_patches_cva.append(patches_cva[i])

    # print(type(filt_patches_img))
    # print(type(filt_patches_img[0]))
    if len(filt_patches_img) > 0:
        filt_patches_img = np.stack(filt_patches_img, axis=0)
        # print(type(filt_patches_img))
        filt_patches_ref = np.stack(filt_patches_ref, axis=0)
        # print(filt_patches_img.shape)
        # print(filt_patches_ref.shape)
        filt_patches_cva = np.stack(filt_patches_cva, axis=0)
    return filt_patches_img, filt_patches_ref, filt_patches_cva


def extract_tiles2patches(tiles, mask_amazon, input_image, image_ref, CVA_ref,
                          patch_size, stride, percent):
    patches_out = []
    labels_out = []
    cva_out = []
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
        cva_tile_ref = CVA_ref[x1:x2+1, y1:y2+1]

        # Extract patches for each tile
        print(tile_img.shape)
        patches_img, patches_ref, patches_cva = extract_patches(tile_img, tile_ref, patch_size,
                                                   stride, cva_tile_ref)

        print(f'Patches of tile {num_tile} extracted!')
        assert len(patches_img) == len(patches_ref), "Train: Input patches and reference \
        patches don't have the same numbers"
        patches_img, patches_ref, patches_cva = filter_patches(patches_img,
                                                               patches_ref,
                                                  patches_cva, percent)
        print(f'Filtered patches: {len(patches_img)}')

        #print(type(patches_img))
        # print(patches_img.shape)
        # print(patch_ref.shape)
        print(len(patches_img))
        if len(patches_img) > 0:
            patches_out.append(patches_img)
            labels_out.append(patches_ref)
            cva_out.append(patches_cva)

        # check_memory()
        # del patches_img, patches_ref
        # print('Variables deleted')
        # check_memory()

    # print(patches_out)
    patches_out = np.concatenate(patches_out, axis=0)
    labels_out = np.concatenate(labels_out, axis=0)
    cva_out = np.concatenate(cva_out, axis=0)
    return patches_out, labels_out, cva_out


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


def save_patches(patches_tr, patches_tr_ref, patches_tr_cva,
                 folder_path, scaler, data_aug, classes_dict=None,
                 mode='train'):
    if classes_dict is None:
        classes_dict = {0: 0, 1: 0, 2: 0}
    for i in tqdm(range(len(patches_tr))):
        # Expand dims (Squeeze) to receive data_augmentation. Depreceated ?
        if data_aug:
            img_aug, label_aug, cva_aug = data_augmentation(patches_tr[i], patches_tr_ref[i],
                                                   patches_tr_cva[i])
            unique, counts = np.unique(label_aug, return_counts=True)
            for clss in unique:
                clss = int(clss)
                classes_dict[clss] += counts[clss]
        else:
            img_aug, label_aug = np.expand_dims(patches_tr[i], axis=0), np.expand_dims(patches_tr_ref[i], axis=0)
            cva_aug = np.expand_dims(patches_tr_cva[i], axis=0)
            unique, counts = np.unique(label_aug[0], return_counts=True)
            for clss in unique:
                clss = int(clss)
                classes_dict[clss] += counts[clss]
        # Performs the one hot encoding
        label_aug_h = tf.keras.utils.to_categorical(label_aug, args.num_classes)
        cva_aug_h = tf.keras.utils.to_categorical(cva_aug, 2)
        for j in range(len(img_aug)):
            # Input image 14 bands of Staelite
            # Float32 its need to train the model
            img_float = img_aug[j].astype(np.float32)
            # print(f'Checking input image shape: {img_float.shape}')
            # img_reshaped = img_float.reshape((img_float.shape[0] * img_float.shape[1]),
            #                                img_float.shape[2])
            # img_normed = scaler.transform(img_reshaped)
            # img_float = img_normed.reshape(img_float.shape[0], img_float.shape[1], img_float.shape[2])
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
            try:
                img_aug_unnorm = scaler.inverse_transform(img_aug[j])
            except:
                img_aug_unnorm = img_aug[j]
            # img_t1_patch = img_aug[j][:, :, 0:7]
            # img_t2_patch = img_aug[j][:, :, 7:]
            img_t1_patch = img_aug_unnorm[:, :, 0:7]
            img_t2_patch = img_aug_unnorm[:, :, 7:]
            assert img_t1_patch.shape == (args.patch_size, args.patch_size, 7), "Img T1 shape not matching"
            assert img_t2_patch.shape == (args.patch_size, args.patch_size, 7), "Img T2 shape not matching"
            # Convert from BGR 2 RGB
            img_t1_patch_bgr = (img_t1_patch[:, :, 1:4]).astype(np.uint8)

            img_t2_patch_bgr = (img_t2_patch[:, :, 1:4]).astype(np.uint8)

            # print(img_t1_patch_bgr.shape)
            # print(img_t2_patch_bgr.shape)
            assert img_t1_patch_bgr.shape == (args.patch_size, args.patch_size, 3), "BGR T1 shape not matching"
            assert img_t2_patch_bgr.shape == (args.patch_size, args.patch_size, 3), "BGR T2 shape not matching"

            img_t1_patch_hsv = cv2.cvtColor(img_t1_patch_bgr,
                                     cv2.COLOR_BGR2HSV).astype(np.float32)
            img_t1_patch_hsv = img_t1_patch_hsv * np.array([1./179, 1./255, 1./255])

            img_t2_patch_hsv = cv2.cvtColor(img_t2_patch_bgr,
                                     cv2.COLOR_BGR2HSV).astype(np.float32)
            img_t2_patch_hsv = img_t2_patch_hsv * np.array([1./179, 1./255, 1./255])
            # print(hsv_patch.shape)
            img_both_patch_hsv = np.concatenate((img_t1_patch_hsv,
                                                 img_t2_patch_hsv), axis=-1)
            # Float32 its need to train the model
            np.save(os.path.join(folder_path, mode, 'masks/color', filename(i*5 + j)),
                    img_both_patch_hsv)

            # CVA segmentation
            np.save(os.path.join(folder_path, mode, 'masks/cva', filename(i*5 + j)),
                    cva_aug_h[j].astype(np.float32))
    print(classes_dict)
    class0 = classes_dict[0] / (classes_dict[0] + classes_dict[1] + classes_dict[2])
    class0 = round(class0, 5)
    print(f'class 0 %: {class0*100}')
    class1 = classes_dict[1] / (classes_dict[0] + classes_dict[1] + classes_dict[2])
    class1 = round(class1, 5)
    print(f'class 1 %: {class1*100}')
    class2 = classes_dict[2] / (classes_dict[0] + classes_dict[1] + classes_dict[2])
    class2 = round(class2, 5)
    print(f'class 2 %: {class2*100}')

    return classes_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm_type",
                        help="Choose type of normalization to be used",
                        type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--patch_size",
                        help="Choose size of patches", type=int, default=128)
    parser.add_argument("--stride",
                        help="Choose stride to be using on patches extraction",
                        type=int, default=32)
    parser.add_argument("--num_classes",
                        help="Choose number of classes to convert \
                        labels to one hot encoding", type=int, default=3)
    parser.add_argument("--data_aug",
                        help="Choose number of classes to convert \
                        labels to one hot encoding", action='store_true')
    parser.add_argument("--def_percent",
                        help="Choose minimum percentage of Deforastation",
                        type=int, default=2)
    # Img 66 cva th = 0.26074
    parser.add_argument("--cva_th", help="Choose CVA threshold",
                        type=float, default=0.34)
    args = parser.parse_args()

    print('='*50)
    print('Parameters')
    print(f'patch size={args.patch_size}')
    print(f'stride={args.stride}')
    print(f'Number of classes={args.num_classes} ')
    print(f'Norm type: {args.norm_type}')
    print(f'Using data augmentation? {args.data_aug}')
    print(f'CVA threshold {args.cva_th}')
    print('='*50)

    root_path = './DATASETS/Amazon_npy_corrigido'
    # Load images --------------------------------------------------------------
    # img_t1_path = 'clipped_raster_004_66_2018.npy'
    # img_t2_path = 'clipped_raster_004_66_2019.npy'
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
    # Need to be in float32 to be normalized
    input_image = np.concatenate((img_t1, img_t2), axis=-1).astype(np.float32)
    # input_image = input_image[:6100, :6600]
    input_image = input_image[:5200, :5040]
    h_, w_, channels = input_image.shape
    print(f"Input image shape: {input_image.shape}")
    check_memory()
    # scaler, input_image = normalization(input_image, norm_type=args.norm_type)
    check_memory()

    # Load Mask area -----------------------------------------------------------
    # Mask constains exactly location of region since the satelite image
    # doesn't fill the entire resolution (Kinda rotated with 0 around)
    # img_mask_ref_path = 'mask_ref.npy'
    # img_mask_ref = load_npy_image(os.path.join(root_path, img_mask_ref_path)).astype(np.float32)
    # img_mask_ref = img_mask_ref[:6100, :6600]
    # print(f"Mask area reference shape: {img_mask_ref.shape}")
    img_mask_ref = None

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
    # image_ref = image_ref[:5200, :5000]
    # image_ref[img_mask_ref == -99] = -1
    print(f"Image reference shape: {image_ref.shape}")

    count_deforastation(image_ref, img_mask_ref)

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
    # past_ref_sum = past_ref_sum[:5200, :5000]
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
    final_mask = final_mask[:5200, :5040]

    # final_mask[img_mask_ref == -99] = -1
    unique, counts = np.unique(final_mask, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print(f'Class pixels of final mask: {counts_dict}')
    deforastation = counts_dict[1] / (counts_dict[0] + counts_dict[1] + counts_dict[2])
    print(f"Total Deforastation: {deforastation * 100}%")

    # Calculates weights for weighted cross entropy
    total_pixels = counts_dict[0] + counts_dict[1] + counts_dict[2]
    weight0 = total_pixels / counts_dict[0]
    weight1 = total_pixels / counts_dict[1]
    weight2 = total_pixels / counts_dict[2]
    print('weights')
    print(weight0)
    print(weight1)
    print(weight2)

    CVA_ref = compute_cva(img_t1, img_t2, args.cva_th)

    check_memory()
    del img_t1, img_t2, image_ref, past_ref1, past_ref2
    print('Images deleted!')
    check_memory()


    # Mask with tiles
    # Divide tiles in 5 rows and 3 columns. Total = 15 tiles
    # (5909, 3067, 7) --> (5900, 3060, 7) --> tiles = (1180, 1020)
    # tile_number = np.ones((1040, 1680))
    tile_number = np.ones((1180, 1020))
    mask_c_1 = np.concatenate((tile_number, 2*tile_number, 3*tile_number), axis=1)
    mask_c_2 = np.concatenate((4*tile_number, 5*tile_number, 6*tile_number), axis=1)
    mask_c_3 = np.concatenate((7*tile_number, 8*tile_number, 9*tile_number), axis=1)
    mask_c_4 = np.concatenate((10*tile_number, 11*tile_number, 12*tile_number), axis=1)
    mask_c_5 = np.concatenate((13*tile_number, 14*tile_number, 15*tile_number), axis=1)
    mask_tiles = np.concatenate((mask_c_1, mask_c_2, mask_c_3, mask_c_4, mask_c_5), axis=0)

    # all_tiles = [i for i in range(1, 16)]
    # tr_tiles = [1, 2, 3, 6, 7, 8, 10, 11, 12]
    tr_tiles = [2, 3, 6, 7, 8, 10, 11, 12]
    # val_tiles = [4, 9]
    val_tiles = [4, 9, 15]
    # tst_tiles = [5, 15, 13, 14]
    tst_tiles = [5, 1, 13, 14]
    all_tiles = tr_tiles + val_tiles
    print(f'All tiles: {all_tiles}')
    # final_mask[img_mask_ref == -99] = -1
    show_deforastation_per_tile(all_tiles, mask_tiles, final_mask)

    # patches_tr, patches_tr_ref = extract_tiles2patches(all_tiles, mask_tiles, input_image,
    #                                                    final_mask, args.patch_size,
    #                                                    args.stride, args.def_percent)
    #
    # patches_tr, patches_val, patches_tr_ref, patches_val_ref = train_test_split(patches_tr,
    #                                                                           patches_tr_ref,
    #                                                                           test_size=0.2, random_state=42)

    # Trainig tiles ------------------------------------------------------------

    patches_tr, patches_tr_ref, patches_tr_cva = extract_tiles2patches(tr_tiles, mask_tiles, input_image,
                                                       final_mask, CVA_ref, args.patch_size,
                                                       args.stride, args.def_percent)

    # Validation tiles ---------------------------------------------------------

    patches_val, patches_val_ref, patches_val_cva = extract_tiles2patches(val_tiles, mask_tiles, input_image,
                                                         final_mask, CVA_ref, args.patch_size, args.stride,
                                                         args.def_percent)

    assert len(patches_tr) == len(patches_tr_ref), "Train: Input patches and reference \
    patches don't have the same numbers"
    assert len(patches_tr) == len(patches_tr_cva), "Train: Input patches and CVA \
    patches don't have the same numbers"
    assert len(patches_val) == len(patches_val_ref), "Val: Input patches and reference \
    patches don't have the same numbers"
    assert len(patches_val) == len(patches_val_cva), "Val: Input patches and CVA \
    patches don't have the same numbers"

    print('patches extracted!')

    print('saving images...')
    folder_path = f'./DATASETS/Amazon_Mabel_patch_size={args.patch_size}_' + \
                  f'stride={args.stride}_norm_type={args.norm_type}' + \
                  f'_data_aug={args.data_aug}_def_percent={args.def_percent}' + \
                  f'_cva_th={args.cva_th}'

    create_folders(folder_path, mode='train')
    create_folders(folder_path, mode='val')

    print(f'Number of train patches: {len(patches_tr)}')
    print(f'Number of val patches: {len(patches_val)}')

    scaler = None
    classes_dict = save_patches(patches_tr, patches_tr_ref, patches_tr_cva, folder_path, scaler, args.data_aug, mode='train')
    classes_dict = save_patches(patches_val, patches_val_ref, patches_val_cva, folder_path, scaler, args.data_aug, classes_dict, mode='val')

    # Save a pie graph with classes proportions
    my_labels = 'No deforastation', 'Deforastation', 'Past Deforastation'
    fig = plt.figure()
    plt.pie(classes_dict.values(), labels=my_labels, autopct='%1.1f%%')
    plt.title('Classes occurrences')
    plt.axis('equal')
    fig.savefig('./classes_occurrences_TrainVal.jpg', dpi=300)
