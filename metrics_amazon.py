import numpy as np
import skimage
from sklearn.metrics import confusion_matrix


def compute_def_metrics(thresholds, img_predicted, img_labels,
                        mask_amazon_ts=None, px_area=69):
    ''' INPUTS:
        thresholds = Vector of threshold values
        img_predicted = predicted maps (with probabilities)
        img_labels = image with labels (0-> no def, 1-> def, 2-> past def)
        mask_amazon_ts = binary tile mask (0-> train + val, 1-> test)
        px_area = not considered area (<69 pixels)

        OUTPUT:
        recall and precision for each threshold
    '''
    metrics_all = []

    for thr in thresholds:
        print(f'Threshold: {thr}')

        img_predicted_ = img_predicted.copy()
        img_predicted_[img_predicted_ >= thr] = 1
        img_predicted_[img_predicted_ < thr] = 0

        # Mask of the small regions (<69 px)
        mask_areas_pred = np.ones_like(img_labels)
        small_area = skimage.morphology.area_opening(img_predicted_,
                                                     area_threshold=px_area,
                                                     connectivity=1)
        area_no_consider = img_predicted_ - small_area
        # print(mask_areas_pred.shape)
        # print(area_no_consider.shape)
        mask_areas_pred[area_no_consider == 1] = 0

        # Mask areas no considered reference (borders and buffer)
        mask_borders = np.ones_like(img_predicted_)
        #ref_no_consid = np.zeros((img_labels.shape))
        mask_borders[img_labels == 2] = 0

        # Final mask of no-considered regions
        mask_no_consider = mask_areas_pred * mask_borders
        ref_consider = mask_no_consider * img_labels
        pred_consider = mask_no_consider*img_predicted_

        # Pixels filtered
        # ref_final = ref_consider[mask_amazon_ts==1]
        # pre_final = pred_consider[mask_amazon_ts==1]

        ref_final = ref_consider
        pre_final = pred_consider
        # print(ref_final.shape)
        #
        # unique, counts = np.unique(ref_final, return_counts=True)
        # counts_dict = dict(zip(unique, counts))
        # print(f'Ref final: {counts_dict}')
        #
        # print(pre_final.shape)
        # unique, counts = np.unique(pre_final, return_counts=True)
        # counts_dict = dict(zip(unique, counts))
        # print(f'Pre final: {counts_dict}')

        ref_final = np.reshape(ref_final, (ref_final.shape[0] *
                                                    ref_final.shape[1]))

        pre_final = np.reshape(pre_final, (pre_final.shape[0] *
                                                 pre_final.shape[1]))

        # Metrics
        cm = confusion_matrix(ref_final, pre_final)

        print(cm)
        #TN = cm[0,0]
        FN = cm[1, 0]
        TP = cm[1, 1]
        FP = cm[0, 1]
        precision_ = TP/(TP+FP)
        recall_ = TP/(TP+FN)

        print(f' Precision: {precision_}')
        print(f' Recall: {recall_}')

        mm = np.hstack((recall_, precision_))
        metrics_all.append(mm)
    metrics_ = np.asarray(metrics_all)
    return metrics_

#%% **** Example ****
#
#
# ref1 = np.ones_like(img_labels).astype(np.float32)
# ref1[img_labels == 2] = 0
# TileMask = mask_amazon_ts * ref1
# GTTruePositives = img_labels==1
#
# Npoints = 100
# Pmax = np.max(mean_prob[GTTruePositives * TileMask ==1])
# ProbList = np.linspace(Pmax,0,Npoints)
# px_area = 69
#
# metrics = compute_def_metrics(ProbList, img_predicted, img_labels, mask_amazon_ts, px_area)
#
#
# #%% ************ CVA ************
# import math
# blue_t1 = image_t1[:,:,1]
# red_t1 = image_t1[:,:,3]
# nir_t1 = image_t1[:,:,4]
# swir_t1 = image_t1[:,:,5]
#
# blue_t2 = image_t2[:,:,1]
# red_t2 = image_t2[:,:,3]
# nir_t2 = image_t2[:,:,4]
# swir_t2 = image_t2[:,:,5]
#
# # NDVI and BI index
# ndvi1 = (nir_t1-red_t1)/(nir_t1+red_t1)
# bi1 = (swir_t1+red_t1)-(nir_t1+blue_t1)/(swir_t1+red_t1)+(nir_t1+blue_t1)
#
# ndvi2 = (nir_t2-red_t2)/(nir_t2+red_t2)
# bi2 = (swir_t2+red_t2)-(nir_t2+blue_t2)/(swir_t2+red_t2)+(nir_t2+blue_t2)
# print(np.min(ndvi1), np.min(bi1))
#
# # Calculating the change:
# S =(ndvi2-ndvi1)**2+(bi2-bi1)**2
# S1 = np.sqrt(S)
#
# print(S1.shape)
# plt.figure(figsize=(10,5))
# ax = sns.heatmap(S1, cmap="jet")
# ax.set_axis_off()
