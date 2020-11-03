import mxnet as mx
import mxnet.gluon.nn as nn


class UNet(nn.HybridBlock):
    def __init__(self, num_classes, first_nfilter=64, **kwargs):
        nn.HybridBlock.__init__(self, **kwargs)
        # super(UNet, self).__init__(**kwargs)
        with self.name_scope():
            # Applying padding=1 to have 'same' padding
            # Remeber formula to know output of a convolution:
            # o = (width - k + 2p)/s + 1 --> p = (k - 1)/2
            # Check this: https://www.quora.com/How-can-I-calculate-the-size-of-output-of-convolutional-layer
            self.first_conv = nn.Conv2D(first_nfilter, kernel_size=3, padding=1)
            self.conv1 = nn.Conv2D(first_nfilter, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2D(2 * first_nfilter, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2D(4 * first_nfilter, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2D(8 * first_nfilter, kernel_size=3, padding=1)

            self.conv_middle = nn.Conv2D(16*first_nfilter, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2D()
            self.conv_pred = nn.Conv2D(num_classes, kernel_size=1, padding=1)

    def hybrid_forward(self, F, x):
        # Encoder
        # conv block 1
        print(x.shape)
        conv1_1 = self.first_conv(x)
        print(conv1_1.shape)
        conv1_1 = F.relu(conv1_1)
        conv1_2 = self.conv1(conv1_1)
        conv1_2 = F.relu(conv1_2)
        pool1 = self.pool(conv1_2)
        # conv block 2
        conv2_1 = self.conv2(pool1)
        conv2_1 = F.relu(conv2_1)
        conv2_2 = self.conv2(conv2_1)
        conv2_2 = F.relu(conv2_2)
        pool2 = self.pool(conv2_2)
        # conv block 3
        conv3_1 = self.conv3(pool2)
        conv3_1 = F.relu(conv3_1)
        conv3_2 = self.conv3(conv3_1)
        conv3_2 = F.relu(conv3_2)
        pool3 = self.pool(conv3_2)
        # conv block 4
        conv4_1 = self.conv4(pool3)
        conv4_1 = F.relu(conv4_1)
        conv4_2 = self.conv4(conv4_1)
        conv4_2 = F.relu(conv4_2)
        pool4 = self.pool(conv4_2)

        # Middle
        # conv block 5 n_f=1024
        conv_middle = self.conv_middle(pool4)
        conv_middle = F.relu(conv_middle)
        conv_middle = self.conv_middle(conv_middle)
        conv_middle = F.relu(conv_middle)

        # Decoder
        # Upsampling conv block 1 --  n_f=512
        up1 = F.UpSampling(conv_middle, scale=2, sample_type='nearest')
        up1 = self.conv4(up1)
        # Concatenate along channel's dimension
        merge1 = F.concatenate(up1, conv4_2, axis=1)
        conv6_1 = self.conv4(merge1)
        conv6_1 = F.relu(conv6_1)
        conv6_2 = self.conv4(conv6_1)
        conv6_2 = F.relu(conv6_2)
        # Upsampling conv block 2 --  n_f=256
        up2 = F.UpSampling(conv6_2, scale=2, sample_type='nearest')
        up2 = self.conv3(up2)
        # Concatenate along channel's dimension
        merge2 = F.concatenate(up2, conv3_2, axis=1)
        conv7_1 = self.conv3(merge2)
        conv7_1 = F.relu(conv7_1)
        conv7_2 = self.conv3(conv7_1)
        conv7_2 = F.relu(conv7_2)
        # Upsampling conv block 3 --  n_f=128
        up3 = F.UpSampling(conv7_2, scale=2, sample_type='nearest')
        up3 = self.conv3(up3)
        # Concatenate along channel's dimension
        merge3 = F.concatenate(up3, conv2_2, axis=1)
        conv8_1 = self.conv2(merge3)
        conv8_1 = F.relu(conv8_1)
        conv8_2 = self.conv3(conv8_1)
        conv8_2 = F.relu(conv8_2)
        # Upsampling conv block 4 --  n_f=64
        up4 = F.UpSampling(conv8_2, scale=2, sample_type='nearest')
        up4 = self.conv1(up4)
        # Concatenate along channel's dimension
        merge4 = F.concatenate(up4, conv1_2, axis=1)
        conv9_1 = self.conv1(merge4)
        conv9_1 = F.relu(conv9_1)
        conv9_2 = self.conv1(conv9_1)
        conv9_2 = F.relu(conv9_2)

        out = self.conv_pred(conv9_2)

        return out
