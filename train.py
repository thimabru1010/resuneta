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
from mxboard import SummaryWriter

def compute_mcc(tp, tn, fp, fn):
    mcc = (tp*tn - fp*fn) / tf.math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn+fn))
    return mcc


def add_tensorboard_scalars(result_path, epoch, task, loss, acc=None, val_mcc=None):
    log_path = os.path.join(result_path, 'logs')
    # [TODO] Maybe 'Loss' need to be at logdir
    with SummaryWriter(logdir=os.path.join(log_path, task)) as sw:
        sw.add_scalar(tag='Loss', value=loss, global_step=epoch)

    if acc is not None:
        with SummaryWriter(logdir=os.path.join(log_path, task)) as sw:
            sw.add_scalar(tag='Accuracy', value=acc, global_step=epoch)

    if val_mcc is not None:
        with SummaryWriter(logdir=os.path.join(log_path, task)) as sw:
            sw.add_scalar(tag='MCC', value=val_mcc, global_step=epoch)

def train_model(args, net, dataloader, devices, patience=10, delta=0.001):
    # [TODO] substitute args parsers for variables
    # softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    if args.loss == 'tanimoto':
        loss_f = Tanimoto_with_dual()
    elif args.loss == 'cross_entropy':
        loss_f = gluon.loss.SoftmaxCrossEntropyLoss(axis=1, sparse_label=False)
    acc_metric = mx.metric.Accuracy()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-4})
    min_loss = float('inf')
    early_cont = 0

    for epoch in range(args.epochs):
        epoch_seg_loss = {'train': 0.0, 'val': 0.0}
        epoch_bound_loss = {'train': 0.0, 'val': 0.0}
        epoch_dist_loss = {'train': 0.0, 'val': 0.0}
        epoch_color_loss = {'train': 0.0, 'val': 0.0}
        # epoch_cva_loss = {'Train': 0.0, 'Val': 0.0}
        epoch_total_loss = {'train': 0.0, 'val': 0.0}

        epoch_seg_acc = {'train': 0.0, 'val': 0.0}
        acc_metric.reset()
        # MCC is calculated for validation only
        epoch_seg_mcc = 0.0

        # Train loop -----------------------------------------------------------
        for data, label in tqdm(dataloader['train'], desc="Train"):
            # Diff 3: split batch and load into corresponding devices (GPU)
            data_list = gluon.utils.split_and_load(data, devices)
            seg_label_list = gluon.utils.split_and_load(label[:, 0:5, :, :], devices)
            # if args.multitasking:
            bound_label_list = gluon.utils.split_and_load(label[:, 5:10, :, :], devices)
            dist_label_list = gluon.utils.split_and_load(label[:, 10:15, :, :], devices)
            color_label_list = gluon.utils.split_and_load(label[:, 15:18, :, :], devices)

            # Diff 4: run forward and backward on each devices.
            # MXNet will automatically run them in parallel
            seg_losses = []
            seg_corrects = []
            bound_losses = []
            dist_losses = []
            color_losses = []
            total_losses = []

            with autograd.record():
                # Gather results from all devices into a single list
                for i, data in enumerate(zip(data_list, seg_label_list, bound_label_list, dist_label_list, color_label_list)):
                    X, y_seg, y_bound, y_dist, y_color = data
                    seg_logits, bound_logits, dist_logits, color_logits = net(X)
                    seg_losses.append(loss_f(seg_logits, y_seg))
                    bound_losses.append(loss_f(bound_logits, y_bound))
                    dist_losses.append(loss_f(dist_logits, y_dist))
                    color_losses.append(loss_f(color_logits, y_color))
                    total_losses.append(seg_losses[i] + args.wbound*bound_losses[i] + args.wdist*dist_losses[i] + args.wcolor*color_losses[i])

                    acc_metric.update(mx.nd.argmax(seg_logits, axis=1), mx.nd.argmax(y_seg, axis=1))
            for loss in total_losses:
                loss.backward()
            trainer.step(args.batch_size)
            # Diff 5: sum losses over all devices
            seg_loss = []
            bound_loss = []
            dist_loss = []
            color_loss = []
            total_loss = []
            for l_seg, l_bound, l_dist, l_color, l_total in zip(seg_losses, bound_losses, dist_losses, color_losses, total_losses):
                # Sums for each device batch
                seg_loss.append(l_seg.sum().asscalar())
                bound_loss.append(l_bound.sum().asscalar())
                dist_loss.append(l_dist.sum().asscalar())
                color_loss.append(l_color.sum().asscalar())
                total_loss.append(l_total.sum().asscalar())
            # Sum loss from all inferences on the batch
            epoch_seg_loss['train'] += sum(seg_loss)
            epoch_bound_loss['train'] += sum(bound_loss)
            epoch_dist_loss['train'] += sum(dist_loss)
            epoch_color_loss['train'] += sum(color_loss)
            epoch_total_loss['train'] += sum(total_loss)

            seg_acc = []
            for seg_correct in seg_corrects:
                seg_acc.append(seg_correct.sum().asscalar())
            epoch_seg_acc['train'] += sum(seg_acc)

        # After batch loop take the mean of batches losses
        n_batches_tr = len(dataloader['train'])
        epoch_seg_loss['train'] /= n_batches_tr
        epoch_bound_loss['train'] /= n_batches_tr
        epoch_dist_loss['train'] /= n_batches_tr
        epoch_color_loss['train'] /= n_batches_tr
        epoch_total_loss['train'] = (epoch_total_loss['train'] / n_batches_tr) / 4

        _, epoch_seg_acc['train'] = acc_metric.get()
        acc_metric.reset()

        # Validation loop ------------------------------------------------------
        for data, label in tqdm(dataloader['val'], desc="Val"):
            data_list = gluon.utils.split_and_load(data, devices)
            seg_label_list = gluon.utils.split_and_load(label[:, 0:5, :, :], devices)
            bound_label_list = gluon.utils.split_and_load(label[:, 5:10, :, :], devices)
            dist_label_list = gluon.utils.split_and_load(label[:, 10:15, :, :], devices)
            color_label_list = gluon.utils.split_and_load(label[:, 15:18, :, :], devices)

            seg_losses = []
            seg_corrects = []
            bound_losses = []
            dist_losses = []
            color_losses = []
            total_losses = []

            for i, data in enumerate(zip(data_list, seg_label_list, bound_label_list, dist_label_list, color_label_list)):
                X, y_seg, y_bound, y_dist, y_color = data
                seg_logits, bound_logits, dist_logits, color_logits = net(X)
                seg_losses.append(loss_f(seg_logits, y_seg))
                bound_losses.append(loss_f(bound_logits, y_bound))
                dist_losses.append(loss_f(dist_logits, y_dist))
                color_losses.append(loss_f(color_logits, y_color))
                total_losses.append(seg_losses[i] + args.wbound*bound_losses[i] + args.wdist*dist_losses[i] + args.wcolor*color_losses[i])

                acc_metric.update(mx.nd.argmax(seg_logits, axis=1), mx.nd.argmax(y_seg, axis=1))

            seg_loss = []
            bound_loss = []
            dist_loss = []
            color_loss = []
            total_loss = []
            for l_seg, l_bound, l_dist, l_color, l_total in zip(seg_losses, bound_losses, dist_losses, color_losses, total_losses):
                # Sums for each device batch
                seg_loss.append(l_seg.sum().asscalar())
                bound_loss.append(l_bound.sum().asscalar())
                dist_loss.append(l_dist.sum().asscalar())
                color_loss.append(l_color.sum().asscalar())
                total_loss.append(l_total.sum().asscalar())
            # Sum loss from all inferences on the batch
            epoch_seg_loss['val'] += sum(seg_loss)
            epoch_bound_loss['val'] += sum(bound_loss)
            epoch_dist_loss['val'] += sum(dist_loss)
            epoch_color_loss['val'] += sum(color_loss)
            epoch_total_loss['val'] += sum(total_loss)

            seg_acc = []
            for seg_correct in seg_corrects:
                seg_acc.append(seg_correct.sum().asscalar())
            epoch_seg_acc['val'] += sum(seg_acc)

        # After batch loop take the mean of batches losses
        n_batches_val = len(dataloader['val'])
        epoch_seg_loss['val'] /= n_batches_val
        epoch_bound_loss['val'] /= n_batches_val
        epoch_dist_loss['val'] /= n_batches_val
        epoch_color_loss['val'] /= n_batches_val
        epoch_total_loss['val'] = (epoch_total_loss['val'] / n_batches_val) / 4

        _, epoch_seg_acc['val'] = acc_metric.get()

        # Show metrics ---------------------------------------------------------
        metrics_table = PrettyTable()
        metrics_table.title = f'Epoch: {epoch}'
        metrics_table.field_names = ['Task', 'Loss', 'Val Loss',
                                     'Acc %', 'Val Acc %']

        metrics_table.add_row(['Seg', round(epoch_seg_loss['train'], 5),
                               round(epoch_seg_loss['val'], 5),
                               round(100*epoch_seg_acc['train'], 5),
                               round(100*epoch_seg_acc['val'], 5)])

        metrics_table.add_row(['Bound', round(epoch_bound_loss['train'], 5),
                               round(epoch_bound_loss['val'], 5), 0, 0])

        metrics_table.add_row(['Dist', round(epoch_dist_loss['train'], 5),
                               round(epoch_dist_loss['val'], 5), 0, 0])

        metrics_table.add_row(['Color', round(epoch_color_loss['train'], 5),
                               round(epoch_color_loss['val'], 5), 0, 0])

        metrics_table.add_row(['Total', round(epoch_total_loss['train'], 5),
                               round(epoch_total_loss['val'], 5), 0, 0])

        print(metrics_table)

        # Add tensorboard scalars ----------------------------------------------

        add_tensorboard_scalars(args.result_path, args.epoch, 'Segmentation',
                                epoch_seg_loss, acc=epoch_seg_acc, val_mcc=None)

        add_tensorboard_scalars(args.result_path, args.epoch, 'Boundary',
                                epoch_bound_loss)

        add_tensorboard_scalars(args.result_path, args.epoch, 'Distance',
                                epoch_dist_loss)

        add_tensorboard_scalars(args.result_path, args.epoch, 'Color',
                                epoch_color_loss)

        # Early stopping -------------------------------------------------------
        if epoch_total_loss['val'] >= min_loss + delta:
            early_cont += 1
            print(f'EarlyStopping counter: {early_cont} out of {patience}')
            if early_cont >= patience:
                print("Early Stopping! \t Training Stopped")
                break
        else:
            early_cont = 0
            min_loss = epoch_total_loss['val']
            print("Saving best model...")
            net.save_parameters(os.path.join(args.results_path,
                                             'best_model.params'))


# End functions definition -----------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--resunet_a", help="choose resunet-a model or not",
    #                     type=str2bool, default=False)
    parser.add_argument("--multitasking", help="choose resunet-a multitasking \
                        or not", action='store_true', default=True)
    parser.add_argument("--debug", help="choose if you want to shoe debug logs",
                        action='store_true', default=False)
    parser.add_argument("--norm_path", help="Load a txt with normalization you want to apply.",
                        type=str, default=None)
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
    parser.add_argument("-lr", "--learning_rate",
                        help="Learning rate on training",
                        type=float, default=1e-4)
    parser.add_argument("--loss", help="choose which loss you want to use",
                        type=str, default='tanimoto',
                        choices=['weighted_cross_entropy', 'cross_entropy',
                                 'tanimoto'])
    parser.add_argument("-optm", "--optimizer",
                        help="Choose which optmizer to use",
                        type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument("--num_classes", help="Number of classes",
                         type=int, default=5)
    parser.add_argument("--epochs", help="Number of epochs",
                        type=int, default=500)
    parser.add_argument("-ps", "--patch_size", help="Size of patches extracted",
                        type=int, default=256)
    parser.add_argument("--wbound", help="Boundary loss weight",
                        type=float, default=1.0)
    parser.add_argument("--wdist", help="Distance transform loss weight",
                        type=float, default=1.0)
    parser.add_argument("--wcolor", help="HSV transform loss weight",
                        type=float, default=1.0)
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

    logger.info(f' {len(devices)} Devices found: {devices}')

    if args.norm_path is not None:
        with open(args.norm_path, 'r') as f:
            mean, std = f.readlines()

        tnorm = Normalize(mean=mean, std=std)
    else:
        tnorm = Normalize()

    train_dataset = ISPRSDataset(root=args.dataset_path,
                                 mode='train', color=True, mtsk=True, norm=tnorm)
    val_dataset = ISPRSDataset(root=args.dataset_path,
                               mode='val', color=True, mtsk=True, norm=tnorm)

    dataloader = {}
    dataloader['train'] = gluon.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dataloader['val'] = gluon.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    print('='*40)
    logger.info(f'Training on {len(train_dataset)} images')
    logger.info(f'Validating on {len(val_dataset)} images')
    print('='*40)

    train_model(args, net, dataloader, args.batch_size, devices, args.epochs)
