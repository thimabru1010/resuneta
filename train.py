from resuneta.models.resunet_d6_causal_mtskcolor_ddist import ResUNet_d6
from resuneta.src.NormalizeDataset import Normalize
from resuneta.src.ISPRSDataset import ISPRSDataset
from resuneta.nn.loss.loss import Tanimoto_with_dual
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms
import argparse
import logging
import os
from prettytable import PrettyTable
from tqdm import tqdm

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


def train_model(net, dataloader, batch_size, devices, epochs):
    # softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    tanimoto = Tanimoto_with_dual()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.1})
    min_loss = float('inf')

    for epoch in range(epochs):
        epoch_seg_loss = {'train': 0.0, 'val': 0.0}
        epoch_bound_loss = {'train': 0.0, 'val': 0.0}
        epoch_dist_loss = {'train': 0.0, 'val': 0.0}
        epoch_color_loss = {'train': 0.0, 'val': 0.0}
        # epoch_cva_loss = {'Train': 0.0, 'Val': 0.0}
        epoch_total_loss = {'train': 0.0, 'val': 0.0}

        epoch_seg_acc = {'train': 0.0, 'val': 0.0}
        # MCC is calculated for validation only
        epoch_seg_mcc = 0.0

        # Train loop
        for data, label in tqdm(dataloader['train'], desc="Train"):
            # print(data.shape)
            # print(label.shape)
            # Diff 3: split batch and load into corresponding devices (GPU)
            data_list = gluon.utils.split_and_load(data, devices)
            seg_label_list = gluon.utils.split_and_load(label[:, 0:5, :, :], devices)
            bound_label_list = gluon.utils.split_and_load(label[:, 5:10, :, :], devices)
            dist_label_list = gluon.utils.split_and_load(label[:, 10:15, :, :], devices)
            color_label_list = gluon.utils.split_and_load(label[:, 15:18, :, :], devices)
            # Diff 4: run forward and backward on each devices.
            # MXNet will automatically run them in parallel
            seg_losses = []
            bound_losses = []
            dist_losses = []
            color_losses = []
            total_losses = []
            with autograd.record():
                for i, data in enumerate(zip(data_list, seg_label_list, bound_label_list, dist_label_list, color_label_list)):
                    X, y_seg, y_bound, y_dist, y_color = data
                    seg_logits, bound_logits, dist_logits, color_logits = net(X)
                    seg_losses.append(tanimoto(seg_logits, y_seg))
                    bound_losses.append(tanimoto(bound_logits, y_bound))
                    dist_losses.append(tanimoto(dist_logits, y_dist))
                    color_losses.append(tanimoto(color_logits, y_color))

                    total_losses.append(seg_losses[i] + bound_losses[i] + dist_losses[i] + color_losses[i])
            for loss in total_losses:
                loss.backward()
            trainer.step(batch_size)
            # Diff 5: sum losses over all devices
            seg_loss = []
            bound_loss = []
            dist_loss = []
            color_loss = []
            total_loss = []
            for l_total, l_seg, l_bound, l_dist, l_color in zip(total_losses, seg_losses, bound_losses, dist_losses, color_losses):
                seg_loss.append(l_seg.sum().asscalar())
                bound_loss.append(l_bound.sum().asscalar())
                dist_loss.append(l_dist.sum().asscalar())
                color_loss.append(l_color.sum().asscalar())
                total_loss.append(l_total.sum().asscalar())
            # Sum loss from batch
            epoch_seg_loss['train'] += sum(seg_loss)
            epoch_bound_loss['train'] += sum(bound_loss)
            epoch_dist_loss['train'] += sum(dist_loss)
            epoch_color_loss['train'] += sum(color_loss)
            epoch_total_loss['train'] += sum(total_loss)

        # After batch loop take the mean of batches losses
        epoch_seg_loss['train'] /= len(dataloader['train'])/batch_size
        epoch_bound_loss['train'] /= len(dataloader['train'])/batch_size
        epoch_dist_loss['train'] /= len(dataloader['train'])/batch_size
        epoch_color_loss['train'] /= len(dataloader['train'])/batch_size
        epoch_total_loss['train'] /= len(dataloader['train'])/batch_size

        metrics_table = PrettyTable()
        metrics_table.title = f'Epoch: {epoch}'
        metrics_table.field_names = ['Task', 'Loss', 'Val Loss',
                                     'Acc %', 'Val Acc %']

        metrics_table.add_row(['Seg', round(epoch_seg_loss['train'], 5),
                               0,
                               0,
                               0])

        metrics_table.add_row(['Bound', round(epoch_bound_loss['train'], 5),
                              0,
                              0,
                              0])

        metrics_table.add_row(['Dist', round(epoch_dist_loss['train'], 5),
                             0,
                             0,
                             0])

        metrics_table.add_row(['Color', round(epoch_color_loss['train'], 5),
                            0,
                            0,
                            0])

        metrics_table.add_row(['Total', round(epoch_total_loss['train'], 5),
                               0,
                               0,
                               0])




# End functions definition -----------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--resunet_a", help="choose resunet-a model or not",
    #                     type=str2bool, default=False)
    # parser.add_argument("--multitasking", help="choose resunet-a multitasking \
    #                     or not", type=str2bool, default=False)
    parser.add_argument("--debug", help="choose if you want to shoe debug logs",
                        action='store_true', default=False)
    parser.add_argument("--norm_path", help="Load a txt with normalization you want to apply.",
                        type=str, default=None)
    # parser.add_argument("--gpu_parallel",
    #                     help="choose 1 to train one multiple gpu",
    #                     type=str2bool, default=False)
    parser.add_argument("-rp", "--results_path", help="Path where to save logs and model checkpoint. \
                        Logs and checkpoint will be saved inside this folder.",
                        type=str, default='./results/results_run1')
    parser.add_argument("-cp", "--checkpoint_path", help="Path where to load \
                        model checkpoint to continue training",
                        type=str, default=None)
    parser.add_argument("-dp", "--dataset_path", help="Path where to load dataset",
                        type=str, default='./DATASETS/patch_size=256_stride=256_norm_type=1_data_aug=False')
    parser.add_argument("-bs", "--batch_size", help="Batch size on training",
                        type=int, default=4)
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
    parser.add_argument("--num_classes", help="Number of classes",
                         type=int, default=5)
    parser.add_argument("--epochs", help="Number of epochs",
                        type=int, default=500)
    # parser.add_argument("-ps", "--patch_size", help="Size of patches extracted",
    #                     type=int, default=256)
    # parser.add_argument("--bound_weight", help="Boundary loss weight",
    #                     type=float, default=1.0)
    # parser.add_argument("--dist_weight", help="Distance transform loss weight",
    #                     type=float, default=1.0)
    # parser.add_argument("--color_weight", help="HSV transform loss weight",
    #                     type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.results_path)):
        os.makedirs(args.results_path)

    if args.debug:
        file_path = os.path.join(args.results_path, 'train_debug.log')
        # logging.basicConfig(filename=file_path, level=logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
    else:
        file_path = os.path.join(args.results_path, 'train_info.log')
        # logging.basicConfig(filename=file_path, level=logging.INFO)
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger('__main__')
    # stream_handler = logging.StreamHandler()
    # logger.addHandler(stream_handler)
    # logger.setLevel(logging.INFO)

    n_gpus = mx.context.num_gpus()
    devices = []
    for i in range(n_gpus):
        devices.append(mx.gpu(i))

    Nfilters_init = 32
    net = ResUNet_d6(Nfilters_init, args.num_classes)
    net.initialize()
    # [TODO] Change this to receive right input size
    net.summary(mx.nd.random.uniform(shape=(args.batch_size, 3, 256, 256)))

    if args.checkpoint_path is None:
        net.collect_params().initialize(force_reinit=True, ctx=devices)
        # net.initialize(init=mx.init.Xavier(), ctx=devices)
    else:
        # [TODO] Load on CPU enable summary and then put on GPU
        net.load_parameters(args.checkpoint_path, ctx=devices)

    net.hybridize()

    logger.info(f'Devices found: {devices}')

    transformer = transforms.Compose([
        transforms.ToTensor()])

    if args.norm_path is not None:
        with open(args.norm_path, 'r') as f:
            mean, std = f.readlines()

        tnorm = Normalize(mean=mean, std=std)
    else:
        tnorm = Normalize()

    train_dataset = ISPRSDataset(root=args.dataset_path,
                                 mode='train', color=True, transform=transformer, mtsk=True, norm=tnorm)
    val_dataset = ISPRSDataset(root=args.dataset_path,
                               mode='val', color=True, mtsk=True, transform=transformer, norm=tnorm)

    dataloader = {}
    dataloader['train'] = gluon.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dataloader['val'] = gluon.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    print('='*40)
    logger.info(f'Training on {len(train_dataset)} images')
    logger.info(f'Validating on {len(val_dataset)} images')
    print('='*40)

    train_model(net, dataloader, args.batch_size, devices, args.epochs)
