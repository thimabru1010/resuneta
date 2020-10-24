"""
DataSet reader for the ISPRS data competition. It assumes the structure under the root directory
where the data are saved
/root/
    /training/
            /imgs/
            /masks/
    /validation/
            /imgs/
            /masks/

"""

import os
import numpy as np

from mxnet.gluon.data import dataset
import cv2
import mxnet as mx

class ISPRSDataset(dataset.Dataset):
    def __init__(self, root, mode='train', mtsk=True, color=True, transform=None, norm=None):

        self._mode = mode
        self.mtsk = mtsk
        self.color = color
        if color:
            self.colornorm = np.array([1./179, 1./255, 1./255])

        self._transform = transform
        self._norm = norm # Normalization of img

        # if (root[-1] == '/'):
        #     self._root_train = root + 'training/'
        #     self._root_val = root + 'validation/'
        # else:
        #     self._root_train = root + '/training/'
        #     self._root_val = root + '/validation/'

        self._root_train = os.path.join(root, 'train')
        self._root_val = os.path.join(root, 'val')

        if mode == 'train':
            self._root_img = os.path.join(self._root_train, 'imgs')
            self._root_mask_seg = os.path.join(self._root_train, 'masks', 'seg')
            self._root_mask_bound = os.path.join(self._root_train, 'masks', 'bound')
            self._root_mask_dist = os.path.join(self._root_train, 'masks', 'dist')
            self._root_mask_color = os.path.join(self._root_train, 'masks', 'color')
        elif mode == 'val':
            self._root_img = os.path.join(self._root_val, 'imgs')
            self._root_mask_seg = os.path.join(self._root_val, 'masks', 'seg')
            self._root_mask_bound = os.path.join(self._root_val, 'masks', 'bound')
            self._root_mask_dist = os.path.join(self._root_val, 'masks', 'dist')
            self._root_mask_color = os.path.join(self._root_val, 'masks', 'color')
        else:
            raise Exception ('I was given inconcistent mode, available choices: {train, val}, aborting ...')

        self._img_list = sorted(os.listdir(self._root_img))
        self._mask_list_seg = sorted(os.listdir(self._root_mask_seg))
        self._mask_list_bound = sorted(os.listdir(self._root_mask_bound))
        self._mask_list_dist = sorted(os.listdir(self._root_mask_dist))
        self._mask_list_color = sorted(os.listdir(self._root_mask_color))

        assert len(self._img_list) == len(self._mask_list_seg), "Seg masks and labels do not have same numbers, error"
        assert len(self._img_list) == len(self._mask_list_bound), "Bound masks and labels do not have same numbers, error"
        assert len(self._img_list) == len(self._mask_list_dist), "Dist masks and labels do not have same numbers, error"
        assert len(self._img_list) == len(self._mask_list_color), "Color masks and labels do not have same numbers, error"

        self.img_names = list(zip(self._img_list, self._mask_list_seg,
                                  self._mask_list_bound, self._mask_list_dist,
                                  self._mask_list_color))


    def __getitem__(self, idx):

        base_filepath = os.path.join(self._root_img, self.img_names[idx][0])
        mask_seg_filepath = os.path.join(self._root_mask_seg, self.img_names[idx][1])
        mask_bound_filepath = os.path.join(self._root_mask_bound, self.img_names[idx][2])
        mask_dist_filepath = os.path.join(self._root_mask_dist, self.img_names[idx][3])
        mask_color_filepath = os.path.join(self._root_mask_color, self.img_names[idx][4])

        # load in float32
        base = np.load(base_filepath)
        # if self.color:
        #     timg = base.transpose([1, 2, 0])[:, :, :3].astype(np.uint8)
        #     base_hsv = cv2.cvtColor(timg,cv2.COLOR_RGB2HSV)
        #     base_hsv = base_hsv *self.colornorm
        #     base_hsv = base_hsv.transpose([2,0,1]).astype(np.float32)

        base = base.astype(np.float32)#.transpose([1, 2, 0])
        print(base.shape)

        # Maybe the masks shouldn't be float 32
        mask_seg = np.load(mask_seg_filepath).astype(np.float32)
        if self.mtsk:
            mask_bound = np.load(mask_bound_filepath).astype(np.float32)
            mask_dist = np.load(mask_dist_filepath).astype(np.float32)
            mask_color = np.load(mask_color_filepath).astype(np.float32)
            # Maybe mask_color will fucked up
            # masks = np.concatenate([mask_seg, mask_bound, mask_dist, mask_color], axis=-1)
            masks = np.stack([mask_seg, mask_bound, mask_dist], axis=0)
            print(masks.shape)

        # mask_seg = mask_seg.astype(np.float32)
        # mask_bound = mask_bound.astype(np.float32)
        # mask_dist = mask_dist.astype(np.float32)
        # mask_color = mask_color.astype(np.float32)


        # if self.color:
        #     mask = np.concatenate([mask,base_hsv],axis=0)

        # Maybe there is an error here
        # if self.mtsk == False:
        #     mask = mask[:6,:,:]

        # Maybe there is an error here
        if self._transform is not None:
            # base, masks = self._transform(base, masks)
            # base = self._transform(base)
            # masks = self._transform(masks)
            if self._norm is not None:
                base = self._norm(base.astype(np.float32))
                mask_color = mask_color * self.colornorm
                # mask_color = (mask_color.transpose([1, 2, 0]) * self.colornorm).transpose([2,0,1])
        else:
            if self._norm is not None:
                base = self._norm(base.astype(np.float32))
                mask_color = mask_color * self.colornorm
                # mask_color = (mask_color.transpose([1, 2, 0]) * self.colornorm).transpose([2,0,1])

        if self.mtsk:
            base = mx.nd.array(base)
            masks = mx.nd.array(masks)
            return {'img': self._transform(base.astype(np.float32)), 'seg': self._transform(masks[0].astype(np.float32)), 'bound': self._transform(masks[1].astype(np.float32)),
                    'dist': self._transform(masks[2].astype(np.float32)), 'color': self._transform(mask_color.astype(np.float32))}
        else:
            return base.astype(np.float32), mask_seg.astype(np.float32)

    def __len__(self):
        return len(self.img_names)
