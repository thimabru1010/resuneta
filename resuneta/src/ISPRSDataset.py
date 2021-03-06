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


class ISPRSDataset(dataset.Dataset):
    def __init__(self, root, mode='train', mtsk=True, transform=None,
                 norm=None, color_channels=3):

        self._mode = mode
        self.mtsk = mtsk
        # self.color = color
        #if color:
        self.colornorm = np.array([1./179, 1./255, 1./255])
        self.color_channels = color_channels

        self._transform = transform
        self._norm = norm  # Normalization of img

        self._root_train = os.path.join(root, 'train')
        self._root_val = os.path.join(root, 'val')

        if mode == 'train':
            self._root_img = os.path.join(self._root_train, 'imgs')
            self._root_mask_seg = os.path.join(self._root_train, 'masks', 'seg')
            self._root_mask_bound = os.path.join(self._root_train, 'masks', 'bound')
            self._root_mask_dist = os.path.join(self._root_train, 'masks', 'dist')
            self._root_mask_color = os.path.join(self._root_train, 'masks', 'color')
            self._root_mask_cva = os.path.join(self._root_train, 'masks', 'cva')
        elif mode == 'val':
            self._root_img = os.path.join(self._root_val, 'imgs')
            self._root_mask_seg = os.path.join(self._root_val, 'masks', 'seg')
            self._root_mask_bound = os.path.join(self._root_val, 'masks', 'bound')
            self._root_mask_dist = os.path.join(self._root_val, 'masks', 'dist')
            self._root_mask_color = os.path.join(self._root_val, 'masks', 'color')
            self._root_mask_cva = os.path.join(self._root_val, 'masks', 'cva')
        else:
            raise Exception ('I was given inconcistent mode, available choices: {train, val}, aborting ...')

        self._img_list = sorted(os.listdir(self._root_img))
        self._mask_list_seg = sorted(os.listdir(self._root_mask_seg))
        self._mask_list_bound = sorted(os.listdir(self._root_mask_bound))
        self._mask_list_dist = sorted(os.listdir(self._root_mask_dist))
        self._mask_list_color = sorted(os.listdir(self._root_mask_color))
        self._mask_list_cva = sorted(os.listdir(self._root_mask_cva))

        assert len(self._img_list) == len(self._mask_list_seg), "Seg masks and labels do not have same numbers, error"
        assert len(self._img_list) == len(self._mask_list_bound), "Bound masks and labels do not have same numbers, error"
        assert len(self._img_list) == len(self._mask_list_dist), "Dist masks and labels do not have same numbers, error"
        assert len(self._img_list) == len(self._mask_list_color), "Color masks and labels do not have same numbers, error"
        assert len(self._img_list) == len(self._mask_list_cva), "CVA masks and labels do not have same numbers, error"

        self.img_names = list(zip(self._img_list, self._mask_list_seg,
                                  self._mask_list_bound, self._mask_list_dist,
                                  self._mask_list_color, self._mask_list_cva))


    def __getitem__(self, idx):

        base_filepath = os.path.join(self._root_img, self.img_names[idx][0])
        mask_seg_filepath = os.path.join(self._root_mask_seg, self.img_names[idx][1])
        mask_bound_filepath = os.path.join(self._root_mask_bound, self.img_names[idx][2])
        mask_dist_filepath = os.path.join(self._root_mask_dist, self.img_names[idx][3])
        mask_color_filepath = os.path.join(self._root_mask_color, self.img_names[idx][4])
        mask_cva_filepath = os.path.join(self._root_mask_cva, self.img_names[idx][4])

        # load in float32
        base = np.load(base_filepath)
        base = base.astype(np.float32)

        mask_seg = np.load(mask_seg_filepath).astype(np.float32)
        if self.mtsk:
            mask_bound = np.load(mask_bound_filepath).astype(np.float32)
            mask_dist = np.load(mask_dist_filepath).astype(np.float32)
            mask_color = np.load(mask_color_filepath).astype(np.float32)
            mask_cva = np.load(mask_cva_filepath).astype(np.float32)
            # CVA Change the channels. Be careful
            # H x W x 18
            masks = np.concatenate([mask_seg, mask_bound, mask_dist, mask_cva, mask_color], axis=-1)
        else:
            masks = mask_seg


        if self._transform is not None:
            augmented = self._transform(image=base, mask=masks)
            base = augmented['image']
            masks = augmented['mask']
            if self._norm is not None:
                base = self._norm(base.astype(np.float32))
                if self.mtsk:
                    masks[:, :, -self.color_channels:] = masks[:, :, -self.color_channels:] * self.colornorm
        else:
            if self._norm is not None:
                base = self._norm(base.astype(np.float32))
                if self.mtsk:
                    masks[:, :, -self.color_channels:] = masks[:, :, -self.color_channels:] * self.colornorm

        return base.astype(np.float32).transpose((2, 0, 1)), masks.astype(np.float32).transpose((2, 0, 1))

    def __len__(self):
        return len(self.img_names)
