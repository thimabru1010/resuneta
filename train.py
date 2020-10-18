from resuneta.models.resunet_d6_causal_mtskcolor_ddist import *
import mxnet as mx
# from mxnet import nd, gpu, gluon, autograd
from mxnet.gluon.data.vision import datasets, transforms
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_mcc(tp, tn, fp, fn):
    mcc = (tp*tn - fp*fn) / tf.math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn+fn))
    return mcc


    # End functions definition -----------------------------------------------------


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--resunet_a", help="choose resunet-a model or not",
    #                     type=str2bool, default=False)
    # parser.add_argument("--multitasking", help="choose resunet-a multitasking \
    #                     or not", type=str2bool, default=False)
    # parser.add_argument("--gpu_parallel",
    #                     help="choose 1 to train one multiple gpu",
    #                     type=str2bool, default=False)
    # parser.add_argument("-rp", "--results_path", help="Path where to save logs and model checkpoint. \
    #                     Logs and checkpoint will be saved inside this folder.",
    #                     type=str, default='./results/results_run1')
    # parser.add_argument("-cp", "--checkpoint_path", help="Path where to load \
    #                     model checkpoint to continue training",
    #                     type=str, default=None)
    # parser.add_argument("-dp", "--dataset_path", help="Path where to load dataset",
    #                     type=str, default='./DATASETS/patch_size=256_stride=32')
    # parser.add_argument("-bs", "--batch_size", help="Batch size on training",
    #                     type=int, default=4)
    # parser.add_argument("-lr", "--learning_rate",
    #                     help="Learning rate on training",
    #                     type=float, default=1e-3)
    # parser.add_argument("--loss", help="choose which loss you want to use",
    #                     type=str, default='weighted_cross_entropy',
    #                     choices=['weighted_cross_entropy', 'cross_entropy',
    #                              'tanimoto'])
    # parser.add_argument("-optm", "--optimizer",
    #                     help="Choose which optmizer to use",
    #                     type=str, choices=['adam', 'sgd'], default='adam')
    # parser.add_argument("--num_classes", help="Number of classes",
    #                     type=int, default=5)
    # parser.add_argument("--epochs", help="Number of epochs",
    #                     type=int, default=500)
    # parser.add_argument("-ps", "--patch_size", help="Size of patches extracted",
    #                     type=int, default=256)
    # parser.add_argument("--bound_weight", help="Boundary loss weight",
    #                     type=float, default=1.0)
    # parser.add_argument("--dist_weight", help="Distance transform loss weight",
    #                     type=float, default=1.0)
    # parser.add_argument("--color_weight", help="HSV transform loss weight",
    #                     type=float, default=1.0)
    # args = parser.parse_args()

    n_gpus = mx.context.num_gpus()
    devices = []
    for i in range(n_gpus):
        devices.append(mx.gpu(i))

    print(devices)
