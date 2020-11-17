import numpy as np
from osgeo import gdal
import os
import cv2
from skimage.transform import resize


def load_tiff_image(patch):
    # Read tiff Image
    print(patch)
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    return img

# Load images --- Mabel
# root_path = './'
# img_t1 = load_tiff_image(root_path+'images/18_08_2017_image'+'.tif')
# print(img_t1.dtype)
# np.save('dataset_npy/18_08_2017_image_float32.npy', img_t1)
#
# img_t2 = load_tiff_image(root_path+'images/21_08_2018_image'+'.tif').astype(np.float32)
# np.save('dataset_npy/21_08_2018_image_float32.npy', img_t2)
#
# image_ref1 = load_tiff_image(root_path+'images/REFERENCE_2018_EPSG4674'+'.tif')
# np.save('dataset_npy/REFERENCE_2018_EPSG4674.npy', image_ref1)
#
# past_ref1 = load_tiff_image(root_path+'images/PAST_REFERENCE_FOR_2018_EPSG4674'+'.tif')
# np.save('dataset_npy/PAST_REFERENCE_FOR_2018_EPSG4674.npy', past_ref1)


# Dataset TCC
# root_path = 'DATASETS/Amazon'
# output_path = 'DATASETS/Amazon_npy'
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
#     os.makedirs(os.path.join(output_path, 'labels'))
# img_t1_path = 'clipped_raster_004_66_2018.tif'
# img_t2_path = 'clipped_raster_004_66_2019.tif'
#
# # Load images
# img_t1 = load_tiff_image(os.path.join(root_path, img_t1_path))
# print(img_t1.dtype)
# np.save(os.path.join(output_path, 'clipped_raster_004_66_2018.npy'), img_t1)
# del img_t1
#
# img_t2 = load_tiff_image(os.path.join(root_path, img_t2_path))
# print(img_t2.dtype)
# np.save(os.path.join(output_path, 'clipped_raster_004_66_2019.npy'), img_t2)
# del img_t2
#
# img_mask_ref_path = 'mask_ref.tif'
# img_mask_ref = load_tiff_image(os.path.join(root_path, img_mask_ref_path))
# print(img_mask_ref.dtype)
# np.save(os.path.join(output_path, 'mask_ref.npy'), img_mask_ref)
# del img_mask_ref
#
# # Load deforastation reference
# image_ref = load_tiff_image(os.path.join(root_path, 'labels/binary_clipped_2019.tif'))
# print(image_ref.dtype)
# np.save(os.path.join(output_path, 'labels/binary_clipped_2019.npy'), image_ref)
# del image_ref
#
# # Load past deforastation reference
# past_ref1 = load_tiff_image(os.path.join(root_path, 'labels/binary_clipped_2013_2018.tif'))
# print(past_ref1.dtype)
# np.save(os.path.join(output_path, 'labels/binary_clipped_2013_2018.npy'), past_ref1)
# del past_ref1
#
# past_ref2 = load_tiff_image(os.path.join(root_path, 'labels/binary_clipped_1988_2012.tif'))
# print(past_ref2.dtype)
# np.save(os.path.join(output_path, 'labels/binary_clipped_1988_2012.npy'), past_ref2)
# del past_ref2

# ISPRS
# img_train = load_tiff_image('DATASETS/homework3/Image_Train.tif')
# print(img_train.shape)
# # img_train = resize(img_train, (4500, 4000))
# np.save('DATASETS/ISPRS_npy/Image_Train.npy', img_train)
# print('img train saved')
# del img_train
#
# ref_train = load_tiff_image('DATASETS/homework3/Reference_Train.tif')
# print(ref_train.shape)
# # ref_train = resize(ref_train, (4500, 4000))
# np.save('DATASETS/ISPRS_npy/Reference_Train.npy', ref_train)
# print('ref train saved')
# del ref_train
#
# img_test = load_tiff_image('DATASETS/homework3/Image_Test.tif')
# np.save('DATASETS/ISPRS_npy/Image_Test.npy', img_test)
# print('img test saved')
# del img_test
#
# ref_test = load_tiff_image('DATASETS/homework3/Reference_Test.tif')
# np.save('DATASETS/ISPRS_npy/Reference_Test.npy', ref_test)
# print('ref test saved')


# Dataset TCC corrigido

# root_path = '/media/thimabru/ssd/TCC/imagens_satelite_dataset/cut_img_66_new'
root_path = '/media/thimabru/ssd/TCC/imagens_satelite_dataset/cut_227_63'
output_path = 'DATASETS/Amazon_npy_cut_test'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, 'labels'))
img_t1_path = 'cut_raster_2018_ok.tif'
img_t2_path = 'cut_raster_2019_ok.tif'

# Load images
img_t1 = load_tiff_image(os.path.join(root_path, img_t1_path))
print(img_t1.dtype)
np.save(os.path.join(output_path, img_t1_path[:-4] + '.npy'), img_t1)
del img_t1

img_t2 = load_tiff_image(os.path.join(root_path, img_t2_path))
print(img_t2.dtype)
np.save(os.path.join(output_path, img_t2_path[:-4] + '.npy'), img_t2)
del img_t2

# img_mask_ref_path = 'mask_ref.tif'
# img_mask_ref = load_tiff_image(os.path.join(root_path, img_mask_ref_path))
# print(img_mask_ref.dtype)
# np.save(os.path.join(output_path, 'mask_ref.npy'), img_mask_ref)
# del img_mask_ref

# Load deforastation reference
img_ref_path = 'cut_ref_2019_ok.tif'
image_ref = load_tiff_image(os.path.join(root_path, img_ref_path))
print(image_ref.dtype)
np.save(os.path.join(output_path, 'labels', img_ref_path[:-4] + '.npy'), image_ref)
del image_ref

# Load past deforastation reference
past_ref1_path = 'cut_ref_1988_2007_ok.tif'
past_ref1 = load_tiff_image(os.path.join(root_path, past_ref1_path))
print(past_ref1.dtype)
np.save(os.path.join(output_path, 'labels', past_ref1_path[:-4] + '.npy'), past_ref1)
del past_ref1

past_ref2_path = 'cut_ref_2008_2018_ok.tif'
past_ref2 = load_tiff_image(os.path.join(root_path, past_ref2_path))
print(past_ref2.dtype)
np.save(os.path.join(output_path, 'labels', past_ref2_path[:-4] + '.npy'), past_ref2)
del past_ref2
