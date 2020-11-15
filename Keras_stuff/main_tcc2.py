import time
from utils import np, plt, load_tiff_image, load_SAR_image, compute_metrics, data_augmentation, unet, normalization, \
RGB_image, extract_patches, patch_tiles, bal_aug_patches, extrac_patch2, test_FCN, pred_recostruction, \
weighted_categorical_crossentropy, mask_no_considered, tf, Adam, prediction, load_model, confusion_matrix, \
EarlyStopping, ModelCheckpoint, identity_block, ResNet50, color_map, load_npy_image
import os
# from utils2 import patch_tiles2, bal_aug_patches2, bal_aug_patches3, patch_tiles3

from sklearn.model_selection import train_test_split

import argparse
import psutil
import gc
# gc.set_debug(gc.DEBUG_SAVEALL)
# print(gc.get_count())

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

    # print(type(filt_patches_img))
    # print(type(filt_patches_img[0]))
    if len(filt_patches_img) > 0:
        filt_patches_img = np.stack(filt_patches_img, axis=0)
        # print(type(filt_patches_img))
        filt_patches_ref = np.stack(filt_patches_ref, axis=0)
        # print(filt_patches_img.shape)
        # print(filt_patches_ref.shape)
    return filt_patches_img, filt_patches_ref


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

        tile_img = input_image[x1:x2+1, y1:y2+1, :]
        tile_ref = image_ref[x1:x2+1, y1:y2+1]

        # Extract patches for each tile
        print(tile_img.shape)
        patches_img, patches_ref = extract_patches(tile_img, tile_ref,
                                                   patch_size, stride)

        print(f'Patches of tile {num_tile} extracted!')
        assert len(patches_img) == len(patches_ref), "Train: Input patches and reference \
        patches don't have the same numbers"
        patches_img, patches_ref = filter_patches(patches_img, patches_ref, percent)
        print(f'Filtered patches: {len(patches_img)}')

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



root_path = '../DATASETS/Amazon_npy_corrigido'
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
input_image = np.concatenate((img_t1, img_t2), axis=-1)
# input_image = input_image[:6100, :6600]
input_image = input_image[:5200, :5040]
h_, w_, channels = input_image.shape
print(f"Input image shape: {input_image.shape}")
check_memory()
input_image = normalization(input_image, norm_type=1)
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
# image_ref = image_ref[:5200, :5000]
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
print('weights')
print(weight0)
print(weight1)


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
tile_number = np.ones((1040, 1680))
mask_c_1 = np.concatenate((tile_number, 2*tile_number, 3*tile_number), axis=1)
mask_c_2 = np.concatenate((4*tile_number, 5*tile_number, 6*tile_number), axis=1)
mask_c_3 = np.concatenate((7*tile_number, 8*tile_number, 9*tile_number), axis=1)
mask_c_4 = np.concatenate((10*tile_number, 11*tile_number, 12*tile_number), axis=1)
mask_c_5 = np.concatenate((13*tile_number, 14*tile_number, 15*tile_number), axis=1)
mask_tiles = np.concatenate((mask_c_1, mask_c_2, mask_c_3, mask_c_4, mask_c_5), axis=0)

mask_tr_val = np.zeros((mask_tiles.shape))

tr_tiles = [2, 8, 13, 7, 11, 1, 14, 13, 9, 6, 10]
# val_tiles = [val3, val4]
val_tiles = [5, 4, 15, 12]

total_no_def = 0
total_def = 0

total_no_def += len(image_ref[image_ref == 0])
total_def += len(image_ref[image_ref == 1])
# Print number of samples of each class
print('Total no-deforestaion class is {}'.format(len(image_ref[image_ref == 0])))
print('Total deforestaion class is {}'.format(len(image_ref[image_ref == 1])))
print('Percentage of deforestaion class is {:.2f}'.format((len(image_ref[image_ref == 1])*100)/len(image_ref[image_ref == 0])))

del img_t1, img_t2, image_ref, past_ref1, past_ref2
print('Images deleted!')

#image_ref[img_mask_ref==-99] = 0
#%% Patches extraction
patch_size = 128
#stride = patch_size
stride = 16

print("="*40)
print(f'Patche size: {patch_size}')
print(f'Stride: {stride}')
print("="*40)

# Percent of class deforestation
percent = 2
# 0 -> No-DEf, 1-> Def, 2 -> No considered
number_class = 3

# Trainig tiles
print('extracting training patches....')
# tr_tiles = [tr1, tr2, tr3, tr4, tr5, tr6]

patches_tr, patches_tr_ref = extract_tiles2patches(tr_tiles, mask_tiles, input_image,
                                         final_mask, patch_size, stride, percent)

print(f"Trainig patches size: {patches_tr.shape}")
print(f"Trainig ref patches size: {patches_tr_ref.shape}")
patches_tr_aug = patches_tr
patches_tr_ref_aug = patches_tr_ref
# patches_tr_aug, patches_tr_ref_aug = bal_aug_patches(percent, patch_size, patches_tr, patches_tr_ref)
#
# print(f"Trainig patches size with data aug: {patches_tr_aug.shape}")
# print(f"Trainig ref patches sizewith data aug: {patches_tr_ref_aug.shape}")

patches_tr_ref_aug_h = tf.keras.utils.to_categorical(patches_tr_ref_aug, number_class)

# Validation tiles
print('extracting validation patches....')
#Validation train_test_split
# patches_tr_aug, patches_val_aug, patches_tr_ref_aug_h, patches_val_ref_aug_h = train_test_split(patches_tr_aug, patches_tr_ref_aug_h, test_size=0.2, random_state=42)

# val_tiles = [val1, val2]
# # patches_val, patches_val_ref = patch_tiles(val_tiles, mask_tiles, image_array, final_mask, patch_size, stride)
# patches_val, patches_val_ref = patch_tiles2(val_tiles, mask_tiles, image_array, final_mask, img_mask_ref, patch_size, stride, percent)
#
# print(f"Validation patches size: {patches_val.shape}")
# print(f"Validation ref patches size: {patches_val_ref.shape}")
#
# patches_val_aug, patches_val_ref_aug = bal_aug_patches2(percent, patch_size, patches_val, patches_val_ref)
# patches_val_ref_aug_h = tf.keras.utils.to_categorical(patches_val_ref_aug, number_class)

patches_val, patches_val_ref = extract_tiles2patches(val_tiles, mask_tiles, input_image,
                                         final_mask, patch_size, stride, percent)
patches_val_aug = patches_val
patches_val_ref_aug = patches_val_ref
patches_val_ref_aug_h = tf.keras.utils.to_categorical(patches_val_ref_aug, number_class)
# print(f"Validation patches size with data aug: {patches_val_aug.shape}")
# print(f"Validation ref patches sizewith data aug: {patches_val_ref_aug_h.shape}")

#%%
start_time = time.time()
exp = 1
rows = patch_size
cols = patch_size
adam = Adam(lr=1e-3, beta_1=0.9)
batch_size = 64

weights = [weight0, weight1, 0]
#print('='*80)
#print(gc.get_count())
loss = weighted_categorical_crossentropy(weights)

model = unet((rows, cols, channels), num_classes=3)
#model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
# print model information
model.summary()

print(f"Class Weights CE: {weights}")

filepath = './models/'
if not os.path.exists(filepath):
    os.makedirs(filepath)
# define early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath+'unet_exp_'+str(exp)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [earlystop, checkpoint]

# train the model
start_training = time.time()
model_info = model.fit(patches_tr_aug, patches_tr_ref_aug_h,
                       batch_size=batch_size, epochs=100,
                       callbacks=callbacks_list, verbose=1,
                       validation_data=(patches_val_aug, patches_val_ref_aug_h))
end_training = time.time() - start_time
#%% Test model
# Creation of mask with test tiles
mask_ts_ = np.zeros((mask_tiles.shape))
ts1 = 1
ts2 = 2
ts3 = 3
ts4 = 4
ts5 = 6
ts6 = 9
ts7 = 12
ts8 = 14
ts9 = 15
mask_ts_[mask_tiles == ts1] = 1
mask_ts_[mask_tiles == ts2] = 1
mask_ts_[mask_tiles == ts3] = 1
mask_ts_[mask_tiles == ts4] = 1
mask_ts_[mask_tiles == ts5] = 1
mask_ts_[mask_tiles == ts6] = 1
mask_ts_[mask_tiles == ts7] = 1
mask_ts_[mask_tiles == ts8] = 1
mask_ts_[mask_tiles == ts9] = 1

#% Load model
model = load_model(filepath+'unet_exp_'+str(exp)+'.h5', compile=False)
# model.summary()
area = 11
# Prediction
img_ref_path = 'cut_ref_2019_ok.npy'
image_ref = load_npy_image(os.path.join(root_path,
                                        'labels', img_ref_path)).astype(np.float32)
ref_final, pre_final, prob_recontructed, ref_reconstructed, mask_no_considered_, mask_ts, time_ts = prediction(model, input_image, image_ref, final_mask, mask_ts_, patch_size, area)

# Metrics
cm = confusion_matrix(ref_final, pre_final)
metrics = compute_metrics(ref_final, pre_final)
print('Confusion  matrix \n', cm)
print('Accuracy: ', metrics[0])
print('F1score: ', metrics[1])
print('Recall: ', metrics[2])
print('Precision: ', metrics[3])

# Alarm area
total = (cm[1,1]+cm[0,1])/len(ref_final)*100
print('Area to be analyzed',total)

print('training time', end_training)
print('test time', time_ts)

#%% Show the results
# prediction of the whole image
fig1 = plt.figure('whole prediction')
plt.imshow(prob_recontructed)
plt.imsave('whole_pred.jpg', prob_recontructed)
# Show the test tiles
fig2 = plt.figure('prediction of test set')
plt.imshow(prob_recontructed*mask_ts)
plt.imsave('pred_test_set.jpg', prob_recontructed*mask_ts)
# plt.show()
