from utils import normalization, load_npy_image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import roc_curve, roc_auc_score

root_path = sys.argv[1]

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

# Normed images
_, image_t1 = normalization(img_t1, norm_type=2)
_, image_t2 = normalization(img_t2, norm_type=2)

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


'''
    0 --> No deforastation
    1 --> Deforastation
'''
image_ref = load_npy_image(os.path.join(root_path, 'labels',
                                        'cut_ref_2019_ok.npy'))

print(f'References Min: {np.min(image_ref)}, Max: {np.max(image_ref)}')

# print(S1.shape)
# plt.figure(figsize=(10,5))
# ax = sns.heatmap(S1, cmap="jet")
# ax.set_axis_off()

ref_final = np.reshape(image_ref, (image_ref.shape[0] * image_ref.shape[1]))

ref_final = ref_final.astype(int)
pred_final = np.reshape(S1, (S1.shape[0] * S1.shape[1]))

fpr, tpr, thresholds = roc_curve(ref_final, pred_final)
print(len(thresholds))
print(thresholds)
auc = roc_auc_score(ref_final, pred_final)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
plt.close()


optimal_idx = np.argmax(tpr - fpr)
print(len(tpr))
print(optimal_idx)
optimal_threshold = thresholds[optimal_idx]

print(f'Optimal Threshold: {optimal_threshold}')

S1_normed = np.copy(S1)
th = optimal_threshold
S1_normed[S1 >= th] = 1
S1_normed[S1 < th] = 0
print(S1_normed)
print(f'S1 normed Min: {np.min(S1_normed)}, Max: {np.max(S1_normed)}')

unique, counts = np.unique(S1_normed, return_counts=True)
counts_dict = dict(zip(unique, counts))
print(f'CVA pixels: {counts_dict}')

unique, counts = np.unique(image_ref, return_counts=True)
counts_dict = dict(zip(unique, counts))
print(f'Image ref pixels: {counts_dict}')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].imshow(S1_normed, cmap='jet')
axes[1].imshow(image_ref, cmap='jet')
plt.show()
plt.close()
