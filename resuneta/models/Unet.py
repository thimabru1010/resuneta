import mxnet as mx
import mxnet.gluon.nn as nn


class UNet(nn.HybridBlock):
    def __init__(self, num_classes, nfilter=64, groups=1, weights=None,
                 from_logits=False, **kwargs):
        # nn.HybridBlock.__init__(self, **kwargs)
        super(UNet, self).__init__(**kwargs)
        with self.name_scope():
            # Applying padding=1 to have 'same' padding
            # Remeber formula to know output of a convolution:
            # o = (width - k + 2p)/s + 1 --> p = (k - 1)/2
            # Check this: https://www.quora.com/How-can-I-calculate-the-size-of-output-of-convolutional-layer
            # Encoder
            # self.encoder = nn.HybridSequential()
            self.conv1_1 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            self.conv1_2 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            nfilter *= 2 #  128
            self.conv2_1 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            self.conv2_2 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            nfilter *= 2 #  256
            self.conv3_1 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            self.conv3_2 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            nfilter *= 2 #  512
            self.conv4_1 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            self.conv4_2 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)

            nfilter *= 2 #  1024
            self.conv5_1 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            self.conv5_2 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)

            # Decoder
            nfilter //= 2 #  512
            self.upconv6 = nn.Conv2D(nfilter, kernel_size=1, padding=0,
                                     use_bias=False, groups=groups)
            self.conv6_1 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            self.conv6_2 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            nfilter //= 2 #  256
            self.upconv7 = nn.Conv2D(nfilter, kernel_size=1, padding=0,
                                     use_bias=False, groups=groups)
            self.conv7_1 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            self.conv7_2 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            nfilter //= 2 #  128
            self.upconv8 = nn.Conv2D(nfilter, kernel_size=1, padding=0,
                                     use_bias=False, groups=groups)
            self.conv8_1 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            self.conv8_2 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            nfilter //= 2 #  64
            self.upconv9 = nn.Conv2D(nfilter, kernel_size=1, padding=0,
                                     use_bias=False, groups=groups)
            self.conv9_1 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)
            self.conv9_2 = nn.Conv2D(nfilter, kernel_size=3, padding=1,
                                     groups=groups)

            # self.pool = nn.MaxPool2D()
            self.pool1 = nn.MaxPool2D()
            self.pool2 = nn.MaxPool2D()
            self.pool3 = nn.MaxPool2D()
            self.pool4 = nn.MaxPool2D()
            self.conv_pred = nn.Conv2D(num_classes, kernel_size=1)
            # Using Hybrid Sequential avoids this error:
            # UserWarning: Gradient of Parameter `unet0_conv0_bias` on context gpu(0) has not been updated by backward since last `step`.
            # Don't know why
            # self.conv_pred = nn.HybridSequential()
            # self.conv_pred.add(nn.Conv2D(num_classes, kernel_size=1))

            self.weights = weights

            self.from_logits = from_logits


    def hybrid_forward(self, F, x):
        # Encoder
        # conv block 1
        conv1_1 = self.conv1_1(x)
        conv1_1 = F.relu(conv1_1)
        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = F.relu(conv1_2)
        pool1 = self.pool1(conv1_2)
        # conv block 2
        conv2_1 = self.conv2_1(pool1)
        conv2_1 = F.relu(conv2_1)
        conv2_2 = self.conv2_2(conv2_1)
        conv2_2 = F.relu(conv2_2)
        pool2 = self.pool2(conv2_2)
        # conv block 3
        conv3_1 = self.conv3_1(pool2)
        conv3_1 = F.relu(conv3_1)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_2 = F.relu(conv3_2)
        pool3 = self.pool3(conv3_2)
        # conv block 4
        conv4_1 = self.conv4_1(pool3)
        conv4_1 = F.relu(conv4_1)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_2 = F.relu(conv4_2)
        pool4 = self.pool4(conv4_2)

        # Middle
        # conv block 5 n_f=1024
        conv_middle = self.conv5_1(pool4)
        conv_middle = F.relu(conv_middle)
        conv_middle = self.conv5_2(conv_middle)
        conv_middle = F.relu(conv_middle)

        # Decoder
        # [TODO] All convolutions after upsample needs a relu
        # Upsampling conv block 6 --  n_f=512
        up6 = F.UpSampling(conv_middle, scale=2, sample_type='nearest')
        up6 = self.upconv6(up6)
        up6 = F.relu(up6)
        # Concatenate along channel's dimension
        merge6 = F.concat(up6, conv4_2, dim=1)
        conv6_1 = self.conv6_1(merge6)
        conv6_1 = F.relu(conv6_1)
        conv6_2 = self.conv6_2(conv6_1)
        conv6_2 = F.relu(conv6_2)
        # Upsampling conv block 7 --  n_f=256
        up7 = F.UpSampling(conv6_2, scale=2, sample_type='nearest')
        up7 = self.upconv7(up7)
        up7 = F.relu(up7)
        # Concatenate along channel's dimension
        merge7 = F.concat(up7, conv3_2, dim=1)
        conv7_1 = self.conv7_1(merge7)
        conv7_1 = F.relu(conv7_1)
        conv7_2 = self.conv7_2(conv7_1)
        conv7_2 = F.relu(conv7_2)
        # Upsampling conv block 8 --  n_f=128
        up8 = F.UpSampling(conv7_2, scale=2, sample_type='nearest')
        up8 = self.upconv8(up8)
        up8 = F.relu(up8)
        # Concatenate along channel's dimension
        merge8 = F.concat(up8, conv2_2, dim=1)
        conv8_1 = self.conv8_1(merge8)
        conv8_1 = F.relu(conv8_1)
        conv8_2 = self.conv8_2(conv8_1)
        conv8_2 = F.relu(conv8_2)
        # Upsampling conv block 9 --  n_f=64
        up9 = F.UpSampling(conv8_2, scale=2, sample_type='nearest')
        up9 = self.upconv9(up9)
        up9 = F.relu(up9)
        # Concatenate along channel's dimension
        merge9 = F.concat(up9, conv1_2, dim=1)
        conv9_1 = self.conv9_1(merge9)
        conv9_1 = F.relu(conv9_1)
        conv9_2 = self.conv9_2(conv9_1)
        conv9_2 = F.relu(conv9_2)

        out = self.conv_pred(conv9_2)
        # out = F.log_softmax(out, axis=1)
        out_logits = F.softmax(out, axis=1)
        # print(out)
        if not self.from_logits:
            return out
        else:
            return out_logits
