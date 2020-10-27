import numpy as np
import cv2
# from skimage.morphology import disk
# import skimage
# from osgeo import ogr, gdal

# Functions

def load_tiff_image(patch):
    # Read tiff Image
    print(patch)
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    return img


def load_npy_image(patch):
    # Read npy Image converted from tiff
    print(patch)
    img = np.load(patch)
    return img

def data_augmentation(image, labels):
    aug_imgs = np.zeros((5, image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint8)
    aug_lbs = np.zeros((5, image.shape[0], image.shape[1]), dtype=np.uint8)

    for i in range(0, len(aug_imgs)):
        aug_imgs[0, :, :, :] = image
        aug_imgs[1, :, :, :] = np.rot90(image, 1)
        aug_imgs[2, :, :, :] = np.rot90(image, 2)
        #aug_imgs[3, :, :, :] = np.rot90(image, 3)
        #horizontal_flip = np.flip(image,0)
        aug_imgs[3, :, :, :] = np.flip(image,0)
        aug_imgs[4, :, :, :] = np.flip(image, 1)
        #aug_imgs[6, :, :] = np.rot90(horizontal_flip, 2)
        #aug_imgs[7, :, :] =np.rot90(horizontal_flip, 3)

    for i in range(0, len(aug_lbs)):
        aug_lbs[0, :, :] = labels
        aug_lbs[1, :, :] = np.rot90(labels, 1)
        aug_lbs[2, :, :] = np.rot90(labels, 2)
        #aug_lbs[3, :, :] = np.rot90(labels, 3)
        #horizontal_flip_lb = np.flip(labels,0)
        aug_lbs[3, :, :] = np.flip(labels,0)
        aug_lbs[4, :, :] = np.flip(labels, 1)
        #aug_lbs[6, :, :] = np.rot90(horizontal_flip_lb, 2)
        #aug_lbs[7, :, :] =np.rot90(horizontal_flip_lb, 3)

    return aug_imgs, aug_lbs


def get_boundary_label(label, kernel_size=(3, 3)):
    _, _, channel = label.shape
    bounds = np.empty_like(label, dtype=np.float32)
    for c in range(channel):
        tlabel = label.astype(np.uint8)
        # Apply filter per channel
        temp = cv2.Canny(tlabel[:, :, c], 0, 1)
        tlabel = cv2.dilate(temp,
                            cv2.getStructuringElement(
                                cv2.MORPH_CROSS,
                                kernel_size),
                            iterations=1)
        # Convert to be used on training (Need to be float32)
        tlabel = tlabel.astype(np.float32)
        # Normalize between [0, 1]
        tlabel /= 255.
        bounds[:, :, c] = tlabel
    return bounds


def get_distance_label(label):
    '''
        Input: label in one-hot encoding. Img formato --> H x W x C
        Output: Distance tranform for each class.
    '''
    label = label.copy()
    dists = np.empty_like(label, dtype=np.float32)
    for channel in range(label.shape[2]):
        patch = label[:, :, channel].astype(np.uint8)
        dist = cv2.distanceTransform(patch, cv2.DIST_L2, 0)
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dists[:, :, channel] = dist

    return dists


# def mask_no_considered(image_ref, buffer, past_ref):
#     # Creation of buffer for pixel no considered
#     image_ref_ = image_ref.copy()
#     im_dilate = skimage.morphology.dilation(image_ref_, disk(buffer))
#     outer_buffer = im_dilate - image_ref_
#     outer_buffer[outer_buffer == 1] = 2
#     # 1 deforestation, 2 past deforastation
#     final_mask = image_ref_ + outer_buffer
#     final_mask[past_ref == 1] = 2
#     return final_mask
