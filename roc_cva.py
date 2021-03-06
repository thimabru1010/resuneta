from utils import load_npy_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib import colors
import os
import sys
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def compute_roc(thresholds, img_predicted, img_labels):
    ''' INPUTS:
        thresholds = Vector of threshold values
        img_predicted = predicted maps (with probabilities)
        img_labels = image with labels (0-> no def, 1-> def, 2-> past def)
        mask_amazon_ts = binary tile mask (0-> train + val, 1-> test)
        px_area = not considered area (<69 pixels)

        OUTPUT:
        recall and precision for each threshold
    '''

    prec = []
    recall = []
    tpr = []
    fpr = []

    for thr in tqdm(thresholds):
        print('-'*60)
        print(f'Threshold: {thr}')

        img_predicted_ = img_predicted.copy()
        img_predicted_[img_predicted_ >= thr] = 1
        img_predicted_[img_predicted_ < thr] = 0

        ref_final = img_labels.copy()
        pre_final = img_predicted_

        # Metrics
        cm = confusion_matrix(ref_final, pre_final)

        TN = cm[0, 0]
        FN = cm[1, 0]
        TP = cm[1, 1]
        FP = cm[0, 1]
        precision_ = TP/(TP+FP)
        recall_ = TP/(TP+FN)

        # TruePositiveRate = TruePositives / (TruePositives + False Negatives)
        TPR = TP / (TP + FN)
        # FalsePositiveRate = FalsePositives / (FalsePositives + TrueNegatives)
        FPR = FP / (FP + TN)

        # print(f' Precision: {precision_}')
        # print(f' Recall: {recall_}')
        print(f'TPR: {TPR}')
        print(f'FPR: {FPR}')

        tpr.append(TPR)
        fpr.append(FPR)
        prec.append(precision_)
        recall.append(recall_)

    print('-'*60)

    return prec, recall, tpr, fpr


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
# _, image_t1 = normalization(img_t1, norm_type=2)
# _, image_t2 = normalization(img_t2, norm_type=2)

image_t1 = img_t1
image_t2 = img_t2

print(f'Img T1 min: {np.min(image_t1)}, max: {np.max(image_t1)}')
print(f'Img T2 min: {np.min(image_t2)}, max: {np.max(image_t2)}')

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
print(f'NVDI min: {np.min(ndvi1)}, max: {np.max(ndvi1)}')
print(f'Bi: min: {np.min(bi1)}, max: {np.max(bi1)}')

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
print(f'S1 Min: {np.min(S1)}, Max: {np.max(S1)}')
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

# thresholds = np.arange(start=np.min(S1), stop=np.max(S1), step=0.02).tolist()
# thresholds.reverse()
#
# _, _, tpr, fpr = compute_roc(thresholds, pred_final, ref_final)

tpr = np.array(tpr)
fpr = np.array(fpr)


optimal_idx = np.argmax(tpr - fpr)
print(len(tpr))
print(optimal_idx)
optimal_threshold = thresholds[optimal_idx]
opt_tpr, opt_fpr = tpr[optimal_idx], fpr[optimal_idx]

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.scatter(opt_fpr, opt_tpr, color='blue', label='Optimal threshold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
plt.close()

print(f'Optimal Threshold: {optimal_threshold}')

S1_normed = np.copy(S1)
th = optimal_threshold

# fig = plt.figure()
# plt.imshow(image_ref, cmap='jet')
# plt.tick_params(axis="x", which = "both", bottom = False, top = False)
# plt.tick_params(axis="y", which = "both", left = False, top = False)
# plt.xticks([])
# plt.yticks([])
# plt.show()
# fig.savefig('def_ref_only.jpg')

desired_S1 = {}
desired_th = [optimal_threshold, 0.5, 1.0]

for th in [optimal_threshold, 0.5, 0.7, 1.0, 1.5]:
    print('-------------------------------------------------------')
    print(f'TH: {th}')
    S1_normed[S1 >= th] = 1
    S1_normed[S1 < th] = 0
    if th in desired_th:
        desired_S1[th] = S1_normed
    print(f'S1 normed Min: {np.min(S1_normed)}, Max: {np.max(S1_normed)}')

    unique, counts = np.unique(S1_normed, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print(f'CVA pixels: {counts_dict}')

    unique, counts = np.unique(image_ref, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print(f'Image ref pixels: {counts_dict}')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].set_title(f'CVA Ref - Threshold = {th:.4f}')
    axes[1].set_title('Def Ref')
    axes[0].imshow(S1_normed, cmap='jet')
    axes[1].imshow(image_ref, cmap='jet')
    plt.show()
    plt.close()
    # fig.savefig(f'cva_ref&def_th={th}.jpg')

    # fig = plt.figure()
    # plt.imshow(S1_normed, cmap='jet')
    # plt.tick_params(axis="x", which="both", bottom=False, top=False)
    # plt.tick_params(axis="y", which="both", left=False, top=False)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # fig.savefig(f'cva_th={th}.jpg')


fig2, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 15))
matplotlib.rcParams.update({'font.size': 15})
# remove the x and y ticks
# for ax in axes:
#     ax.set_xticks([])
    # ax.set_yticks([])
plt.setp(axes, xticks=[], yticks=[])
plt.subplots_adjust(0, 0, 1, 1)

axes[0, 0].imshow(image_ref, cmap='jet')
axes[0, 0].set_title('Actual Deforestation Reference')

th = optimal_threshold
S1_normed[S1 >= th] = 1
S1_normed[S1 < th] = 0
axes[0, 1].imshow(S1_normed, cmap='jet')
axes[0, 1].set_title(f'CVA magnitude with threshold={optimal_threshold:.2f}')

th = 0.5
S1_normed[S1 >= th] = 1
S1_normed[S1 < th] = 0
axes[1, 0].imshow(S1_normed, cmap='jet')
axes[1, 0].set_title('CVA magnitude with threshold=0.5')

th = 1.0
S1_normed[S1 >= th] = 1
S1_normed[S1 < th] = 0
axes[1, 1].imshow(S1_normed, cmap='jet')
axes[1, 1].set_title('CVA magnitude with threshold=1.0')

plt.tight_layout()
fig2.savefig('results/CVA_magnitude&DefRef.jpg')
plt.show()
plt.close()
