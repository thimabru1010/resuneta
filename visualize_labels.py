import mxnet as mx
from resuneta.src.ISPRSDataset import ISPRSDataset
from mxnet import gluon
from pickle import load
import numpy as np
import matplotlib.pyplot as plt

dataset_path = '/media/thimabru/ssd/TCC/Resuneta_multasking_amazon/DATASETS/Amazon_patch_size=128_stride=16_norm_type=1_data_aug=True_def_percent=2'
nclasses = 3

train_dataset = ISPRSDataset(root=dataset_path,
                             mode='train', color=True, mtsk=False)
val_dataset = ISPRSDataset(root=dataset_path,
                           mode='val', color=True, mtsk=False)

dataloader = {}
dataloader['train'] = gluon.data.DataLoader(train_dataset,
                                           batch_size=1,
                                           shuffle=False)
dataloader['val'] = gluon.data.DataLoader(val_dataset,
                                         batch_size=1,
                                         shuffle=False)

# load the scaler
scaler = load(open('scaler.pkl', 'rb'))
print(scaler.scale_)
print(scaler.mean_)

for data, label in dataloader['train']:
    image = mx.nd.squeeze(data).asnumpy().transpose((1, 2, 0))
    print(image.shape)
    image_reshaped = image.reshape((image.shape[0] * image.shape[1]),
                               image.shape[2])
    print(image)
    img = scaler.inverse_transform(image_reshaped)
    # print(img)
    print(image.shape)
    img = img.reshape(image.shape[0], image.shape[1], image.shape[2])
    img_t1 = img[:, :, 0:7]
    img_t2 = img[:, :, 7:14]
    img_t1_bgr = img_t1[:, :, 1:4]
    img_t1_rgb = img_t1_bgr[:, :, ::-1]# .astype(np.uint8)
    img_t2_bgr = img_t2[:, :, 1:4]
    img_t2_rgb = img_t2_bgr[:, :, ::-1]# .astype(np.uint8)

    print('images shape')
    print(img_t1_rgb.shape)
    print(img_t2_rgb.shape)


    # seg = label[:, 0:nclasses, :, :]
    print('label')
    print(label.shape)
    seg = mx.nd.squeeze(label).asnumpy().transpose((1, 2, 0))
    print(seg.shape)

    fig2, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    ax1.set_title('Img T1')
    ax1.imshow(img_t1_rgb)
    ax2.set_title('Img T2')
    ax2.imshow(img_t2_rgb)
    ax3.set_title('Label')
    ax3.imshow(seg)

    plt.show()
    plt.close()
