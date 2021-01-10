import numpy as np
import os

from metrics_amazon import compute_def_metrics
# from test_amazon import pred_recostruction
from utils import load_npy_image, mask_no_considered
import matplotlib.pyplot as plt


def pred_recostruction(patch_size, pred_labels, binary_img_test_ref, img_type=1,
                       cont=0):
    # Patches Reconstruction
    if img_type == 1:
        # Single channel images
        stride = patch_size

        height, width = binary_img_test_ref.shape

        num_patches_h = height // stride
        num_patches_w = width // stride
        #print(num_patches_h, num_patches_w)

        new_shape = (height, width)
        img_reconstructed = np.full(new_shape, 0, dtype=np.float32)
        # cont = 0
        # rows
        for h in range(num_patches_h):
            # columns
            for w in range(num_patches_w):
                img_reconstructed[h*stride:(h+1)*stride, w*stride:(w+1)*stride] = pred_labels[cont]
                cont += 1
        print('Reconstruction Done!')
    if img_type == 2:
        # Reconstruct RGB images
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
    print(f'Reconstructed Image shape: {img_reconstructed.shape}')
    return img_reconstructed, cont

root_path = 'DATASETS/Amazon_npy_MabelNormalized'

h_bound = 5900
w_bound = 3060

# Load deforastation reference ---------------------------------------------
'''
    0 --> No deforastation
    1 --> Deforastation
'''
img_ref_path = 'cut_ref_2019_ok.npy'
image_ref = load_npy_image(os.path.join(root_path,
                                        'labels', img_ref_path)).astype(np.float32)
# Clip to fit tiles of your specific image

image_ref = image_ref[:h_bound, :w_bound]


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

past_ref_sum = past_ref_sum[:h_bound, :w_bound]

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


folders = ['resuneta_run3_Baseline_NoCVA', 'resuneta_run2', 'resuneta_run4',
           'resuneta_run5']
# folders = ['resuneta_run3_Baseline_NoCVA', 'resuneta_run2']

labels = dict(zip(folders, ['Baseline', 'Baseline+CVA', 'Baseline+CVA-Dist',
                            'Baseline+CVA-Bound']))

ProbList = np.arange(start=0, stop=1, step=0.02).tolist()
ProbList.reverse()
print(ProbList)

# Separate per tiles
h_tiles = 1180
w_tiles = 1020
# tst_tiles = [5, 13, 11, 8, 6]
# tst_tiles = [5, 13, 11, 8, 6, 4]
tst_tiles = [5, 13, 8, 6]

tile_number = np.ones((h_tiles, w_tiles))
mask_c_1 = np.concatenate((tile_number, 2*tile_number, 3*tile_number), axis=1)
mask_c_2 = np.concatenate((4*tile_number, 5*tile_number, 6*tile_number), axis=1)
mask_c_3 = np.concatenate((7*tile_number, 8*tile_number, 9*tile_number), axis=1)
mask_c_4 = np.concatenate((10*tile_number, 11*tile_number, 12*tile_number), axis=1)
mask_c_5 = np.concatenate((13*tile_number, 14*tile_number, 15*tile_number), axis=1)
mask_tiles = np.concatenate((mask_c_1, mask_c_2, mask_c_3, mask_c_4, mask_c_5), axis=0)

mask_tst = np.zeros_like(mask_tiles)
for tst_tile in tst_tiles:
    mask_tst[mask_tiles == tst_tile] = 1

fig_pr = plt.figure()

for folder in folders:
    seg_preds = np.load(os.path.join('results', folder, 'seg_preds.npy'))

    seg_preds_def = seg_preds[:, :, :, 1]

    def_probs_reconstructed, _ = pred_recostruction(128, seg_preds_def,
                                                    final_mask)

    def_metrics, prec, recall = compute_def_metrics(ProbList,
                                                    def_probs_reconstructed,
                                                    final_mask,
                                                    mask_tst)

    Recall = def_metrics[:, 0]
    Precision = def_metrics[:, 1]

    DeltaR = Recall[1:]-Recall[:-1]
    ap = np.sum(Precision[:-1]*DeltaR)
    print('mAP:', ap)

    # Precision x Recall curve
    plt.plot(def_metrics[:, 0], def_metrics[:, 1], lw=2,
             label=labels[folder] + f' (mAP: {ap:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision x Recall')
# plt.legend([f'Original', f'Interpolated'], loc="lower right")
plt.legend()

plt.show()
plt.close()

fig_pr.savefig(os.path.join('results', 'precisionXrecall_AllTasks.jpg'),
               dpi=300)
