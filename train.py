from resuneta.models.resunet_d6_causal_mtskcolor_ddist import ResUNet_d6
from resuneta.models.Unet import UNet
from resuneta.models.Unet_small import UNet_small
from resuneta.src.NormalizeDataset import Normalize
from resuneta.src.ISPRSDataset import ISPRSDataset
from resuneta.nn.loss.loss import Tanimoto_with_dual
from weighted_cross_entropy import WeightedSoftmaxCrossEntropyLoss
import mxnet as mx
from mxnet import gluon, autograd
import argparse
import logging
import os
from prettytable import PrettyTable
from tqdm import tqdm
from mxboard import SummaryWriter
import numpy as np
# from gluoncv.loss import ICNetLoss
import gluoncv
import albumentations as A
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt


def add_tensorboard_scalars(summary_writer, result_path, epoch, task, loss, acc=None, val_mcc=None):
    # log_path = os.path.join(result_path, 'logs', task)
    # [TODO] Maybe 'Loss' need to be at logdir
    # with SummaryWriter(logdir=log_path, verbose=False):
    summary_writer.add_scalar(tag=task+'/Loss', value=loss, global_step=epoch)

    if acc is not None:
        # with SummaryWriter(logdir=log_path, verbose=False) as sw:
        summary_writer.add_scalar(tag=task+'/Accuracy', value=acc, global_step=epoch)

    if val_mcc is not None:
        # with SummaryWriter(logdir=log_path, verbose=False) as sw:
        summary_writer.add_scalar(tag=task+'/MCC', value=val_mcc, global_step=epoch)


def train_model(args, net, dataloader, devices, summary_writer, from_logits,
                patience=10, delta=0.001):

    print(f'Checking from logits: {from_logits}')
    wce_weights = None

    if args.loss == 'tanimoto':
        loss_seg = Tanimoto_with_dual()
        loss_bound = Tanimoto_with_dual()
        loss_dist = Tanimoto_with_dual()
        loss_color = Tanimoto_with_dual(no_past_def=False)
        loss_cva = Tanimoto_with_dual()
    elif args.loss == 'ce':
        loss_seg = gluon.loss.SoftmaxCrossEntropyLoss(axis=1,
                                                       from_logits=from_logits,
                                                       sparse_label=False)

        loss_bound = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

        loss_dist = gluon.loss.L2Loss()
        loss_color = gluon.loss.L2Loss()

        loss_cva = gluon.loss.SoftmaxCrossEntropyLoss(axis=1,
                                                      from_logits=from_logits,
                                                      sparse_label=False)
    elif args.loss == 'focal':
        loss_seg = gluoncv.loss.FocalLoss(axis=1, from_logits=from_logits,
                                           sparse_label=False, alpha=args.alpha,
                                           gamma=args.gamma)
        loss_bound = gluoncv.loss.FocalLoss(axis=1, from_logits=True,
                                          sparse_label=False, alpha=args.alpha,
                                          gamma=args.gamma)
        # L2Loss --> MSE
        loss_dist = gluon.loss.L2Loss() #  TODO: Maybe should put weights for distance transform
        loss_color = gluon.loss.L2Loss()

        loss_cva = gluoncv.loss.FocalLoss(axis=1, from_logits=from_logits,
                                          sparse_label=False, alpha=args.alpha,
                                          gamma=args.gamma)
    elif args.loss == 'wce':
        if args.wce_weights == 1:
            # wce_weights = mx.nd.array(np.array([1.1, 238, 0]))
            # wce_weights = mx.nd.array(np.array([1.6, 77, 0]))
            wce_weights = mx.nd.array(np.array([20, 50, 0]))
        elif args.wce_weights == 2:
            # wce_weights = mx.nd.array(np.array([1.1, 238, 11]))
            wce_weights = mx.nd.array(np.array([1.6, 77, 2.8]))
        elif args.wce_weights == 3:
            wce_weights = mx.nd.array(np.array([0.3, 0.7, 0]))
        elif args.wce_weights == 4:
            wce_weights = mx.nd.array(np.array([0.4, 0.6, 0]))
        elif args.wce_weights == 5:
            wce_weights = mx.nd.array(np.array([0.5, 0.5, 0]))
        print(f'WCE weights: {wce_weights}')
        loss_seg = WeightedSoftmaxCrossEntropyLoss(axis=1,
                                                    from_logits=from_logits,
                                                    sparse_label=False,
                                                    class_weights=wce_weights)

        loss_bound = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        # L2Loss --> MSE
        loss_dist = gluon.loss.L2Loss() #  TODO: Maybe should put weights for distance transform
        loss_color = gluon.loss.L2Loss()

        weights = mx.nd.array(np.array([1.0, 1.0]))
        loss_cva = WeightedSoftmaxCrossEntropyLoss(axis=1,
                                                    from_logits=from_logits,
                                                    sparse_label=False,
                                                    class_weights=weights)

    acc_metric = mx.metric.Accuracy()
    acc_metric_cva = mx.metric.Accuracy()
    acc_metric_def = mx.metric.Accuracy()
    mcc_metric = mx.metric.PCC()
    if args.optimizer == 'adam':
        optm = mx.optimizer.Adam(learning_rate=args.learning_rate,
                                 wd=args.weight_decay)
    elif args.optimizer == 'sgd':
        optm = mx.optimizer.SGD(momentum=args.momentum, learning_rate=args.learning_rate,
                                wd=args.weight_decay)
    trainer = gluon.Trainer(net.collect_params(), optm)
    min_loss = float('inf')
    early_cont = 0
    nclasses = args.num_classes

    for epoch in range(args.epochs):
        epoch_seg_loss = {'train': 0.0, 'val': 0.0}
        epoch_bound_loss = {'train': 0.0, 'val': 0.0}
        epoch_dist_loss = {'train': 0.0, 'val': 0.0}
        epoch_color_loss = {'train': 0.0, 'val': 0.0}
        epoch_cva_loss = {'train': 0.0, 'val': 0.0}
        epoch_total_loss = {'train': 0.0, 'val': 0.0}

        epoch_seg_acc = {'train': 0.0, 'val': 0.0}
        # epoch_seg_acc_def = {'train': 0.0, 'val': 0.0}
        epoch_cva_acc = {'train': 0.0, 'val': 0.0}
        acc_metric.reset()
        acc_metric_cva.reset()
        # MCC is calculated for validation only
        epoch_seg_mcc = 0.0
        epoch_seg_acc_def = 0.0

        # Train loop -----------------------------------------------------------
        for data, label in tqdm(dataloader['train'], desc="Train"):
            # Diff 3: split batch and load into corresponding devices (GPU)

            # print(data.shape)
            # print(label.shape)
            logger.debug(f'Train data shape: {data.shape}')
            data_list = gluon.utils.split_and_load(data, devices)
            logger.debug(f'Seg label shape: {label[:, 0:nclasses, :, :].shape}')
            seg_label_list = gluon.utils.split_and_load(label[:, 0:nclasses, :, :], devices)
            if args.multitasking:
                bound_label_list = gluon.utils.split_and_load(label[:, nclasses:2*nclasses, :, :], devices)
                dist_label_list = gluon.utils.split_and_load(label[:, 2*nclasses:3*nclasses, :, :], devices)
                cva_label_list = gluon.utils.split_and_load(label[:, 3*nclasses:(3*nclasses+2), :, :], devices)
                if args.dataset_type == 'amazon':
                    color_label_list = gluon.utils.split_and_load(label[:, -6:, :, :], devices)
                else:
                    color_label_list = gluon.utils.split_and_load(label[:, -3:, :, :], devices)
            else:
                bound_label_list = seg_label_list
                dist_label_list = seg_label_list
                color_label_list = seg_label_list
                cva_label_list = seg_label_list

            # Diff 4: run forward and backward on each devices.
            # MXNet will automatically run them in parallel
            seg_losses = []
            bound_losses = []
            dist_losses = []
            color_losses = []
            cva_losses = []
            total_losses = []

            with autograd.record():
                # Gather results from all devices into a single list
                for i, data in enumerate(zip(data_list, seg_label_list, bound_label_list, dist_label_list, color_label_list, cva_label_list)):
                    X, y_seg, y_bound, y_dist, y_color, y_cva = data
                    if args.multitasking:
                        seg_logits, bound_logits, dist_logits, cva_logits = net(X)
                        # No CVA
                        # seg_logits, bound_logits, dist_logits = net(X)
                        # cva_logits = y_cva
                        # No Dist
                        # seg_logits, bound_logits, cva_logits = net(X)
                        # dist_logits = y_dist
                        # seg_logits, dist_logits, cva_logits = net(X)
                        # bound_logits = y_bound
                    # logger.debug(f'Seg logits: {seg_logits}')
                    else:
                        seg_logits = net(X)
                        cva_logits = seg_logits
                        # print(seg_logits)
                    seg_losses.append(loss_seg(seg_logits, y_seg))
                    # logger.debug(f'Seg CE value: {seg_losses[i]}')
                    acc_metric.update(mx.nd.argmax(seg_logits, axis=1), mx.nd.argmax(y_seg, axis=1))
                    # y_seg_def_only = mx.nd.argmax(y_seg, axis=1)
                    # y_seg_def_only[y_seg_def_only == 2] == 0
                    # acc_metric_def.update(mx.nd.argmax(seg_logits, axis=1), y_seg_def_only)
                    acc_metric_cva.update(mx.nd.argmax(cva_logits, axis=1), mx.nd.argmax(y_cva, axis=1))

                    if args.multitasking:
                        bound_losses.append(args.wbound*loss_bound(bound_logits, y_bound))
                        dist_losses.append(args.wdist*loss_dist(dist_logits, y_dist))
                        # color_losses.append(args.wcolor*loss_color(color_logits, y_color))
                        color_losses.append(0.0)
                        cva_losses.append(args.wcva*loss_cva(cva_logits, y_cva))

                        total_losses.append(seg_losses[i] + bound_losses[i] + dist_losses[i] + cva_losses[i])
                        # total_losses.append(seg_losses[i] + bound_losses[i] + dist_losses[i])
                    else:
                        bound_losses.append(0.0)
                        dist_losses.append(0.0)
                        color_losses.append(0.0)
                        cva_losses.append(0.0)
                        total_losses.append(seg_losses[i])

            for loss in total_losses:
                loss.backward()
            trainer.step(args.batch_size)
            # Diff 5: sum losses over all devices
            seg_loss = []
            bound_loss = []
            dist_loss = []
            color_loss = []
            cva_loss = []
            total_loss = []
            for l_losses in zip(seg_losses, bound_losses, dist_losses, color_losses, cva_losses, total_losses):
                l_seg, l_bound, l_dist, l_color, l_cva, l_total = l_losses
                # Sums for each device batch
                seg_loss.append(l_seg.sum().asscalar())
                if args.multitasking:
                    bound_loss.append(l_bound.sum().asscalar())
                    dist_loss.append(l_dist.sum().asscalar())
                    # color_loss.append(l_color.sum().asscalar())
                    color_loss.append(0.0)
                    cva_loss.append(l_cva.sum().asscalar())
                total_loss.append(l_total.sum().asscalar())
            # Sum loss from all inferences on the batch
            epoch_seg_loss['train'] += sum(seg_loss)
            epoch_bound_loss['train'] += sum(bound_loss)
            epoch_dist_loss['train'] += sum(dist_loss)
            epoch_color_loss['train'] += sum(color_loss)
            epoch_cva_loss['train'] += sum(cva_loss)
            epoch_total_loss['train'] += sum(total_loss)

        # After batch loop take the mean of batches losses
        n_batches_tr = len(dataloader['train'])
        epoch_seg_loss['train'] /= n_batches_tr
        epoch_bound_loss['train'] /= n_batches_tr
        epoch_dist_loss['train'] /= n_batches_tr
        epoch_color_loss['train'] /= n_batches_tr
        epoch_cva_loss['train'] /= n_batches_tr
        if args.multitasking:
            # tasks_weights = [1.0, args.wbound, args.wdist, args.wcolor, args.wcva]
            tasks_weights = [1.0, args.wbound, args.wdist, args.wcva]
            epoch_total_loss['train'] = (epoch_total_loss['train'] / n_batches_tr) / sum(tasks_weights)
        else:
            epoch_total_loss['train'] = (epoch_total_loss['train'] / n_batches_tr)

        _, epoch_seg_acc['train'] = acc_metric.get()
        _, epoch_cva_acc['train'] = acc_metric_cva.get()
        acc_metric.reset()
        acc_metric_cva.reset()
        acc_metric_def.reset()
        mcc_metric.reset()

        # Validation loop ------------------------------------------------------
        for data, label in tqdm(dataloader['val'], desc="Val"):
            logger.debug(f'Val data shape: {data.shape}')
            data_list = gluon.utils.split_and_load(data, devices)
            seg_label_list = gluon.utils.split_and_load(label[:, 0:nclasses, :, :], devices)
            if args.multitasking:
                bound_label_list = gluon.utils.split_and_load(label[:, nclasses:2*nclasses, :, :], devices)
                dist_label_list = gluon.utils.split_and_load(label[:, 2*nclasses:3*nclasses, :, :], devices)
                cva_label_list = gluon.utils.split_and_load(label[:, 3*nclasses:(3*nclasses+2), :, :], devices)
                if args.dataset_type == 'amazon':
                    color_label_list = gluon.utils.split_and_load(label[:, -6:, :, :], devices)
                else:
                    color_label_list = gluon.utils.split_and_load(label[:, -3:, :, :], devices)
            else:
                bound_label_list = seg_label_list
                dist_label_list = seg_label_list
                color_label_list = seg_label_list
                cva_label_list = seg_label_list

            seg_losses = []
            bound_losses = []
            dist_losses = []
            color_losses = []
            cva_losses = []
            total_losses = []

            for i, data in enumerate(zip(data_list, seg_label_list, bound_label_list, dist_label_list, color_label_list, cva_label_list)):
                X, y_seg, y_bound, y_dist, y_color, y_cva = data
                if args.multitasking:
                    seg_logits, bound_logits, dist_logits, cva_logits = net(X)
                    # No CVA
                    # seg_logits, bound_logits, dist_logits = net(X)
                    # cva_logits = y_cva
                    # No Dist
                    # No Dist
                    # seg_logits, bound_logits, cva_logits = net(X)
                    # dist_logits = y_dist
                    # seg_logits, dist_logits, cva_logits = net(X)
                    # bound_logits = y_bound
                else:
                    seg_logits = net(X)
                    cva_logits = seg_logits

                seg_losses.append(loss_seg(seg_logits, y_seg))
                acc_metric.update(mx.nd.argmax(seg_logits, axis=1), mx.nd.argmax(y_seg, axis=1))
                # y_seg_def_only = mx.nd.argmax(y_seg, axis=1)
                # y_seg_def_only = y_seg_def_only.as_in_context(mx.cpu())
                # y_seg_def_only[y_seg_def_only == 2.0] == 0.0
                # acc_metric_def.update(mx.nd.argmax(seg_logits.as_in_context(mx.cpu()), axis=1), y_seg_def_only)
                acc_metric_cva.update(mx.nd.argmax(cva_logits, axis=1), mx.nd.argmax(y_cva, axis=1))
                # mcc_metric.update(mx.nd.argmax(seg_logits, axis=1), mx.nd.argmax(y_seg, axis=1))
                if args.multitasking:
                    bound_losses.append(args.wbound*loss_bound(bound_logits, y_bound))
                    dist_losses.append(args.wdist*loss_dist(dist_logits, y_dist))
                    # color_losses.append(args.wcolor*loss_color(color_logits, y_color))
                    color_losses.append(0.0)
                    cva_losses.append(args.wcva*loss_cva(cva_logits, y_cva))
                    total_losses.append(seg_losses[i] + bound_losses[i] + dist_losses[i] + cva_losses[i])
                    # total_losses.append(seg_losses[i] + bound_losses[i] + dist_losses[i])
                else:
                    bound_losses.append(0.0)
                    dist_losses.append(0.0)
                    color_losses.append(0.0)
                    cva_losses.append(0.0)
                    total_losses.append(seg_losses[i])


            seg_loss = []
            bound_loss = []
            dist_loss = []
            color_loss = []
            cva_loss = []
            total_loss = []
            for l_losses in zip(seg_losses, bound_losses, dist_losses, color_losses, cva_losses, total_losses):
                l_seg, l_bound, l_dist, l_color, l_cva, l_total = l_losses
                # Sums for each device batch
                seg_loss.append(l_seg.sum().asscalar())
                if args.multitasking:
                    bound_loss.append(l_bound.sum().asscalar())
                    dist_loss.append(l_dist.sum().asscalar())
                    # color_loss.append(l_color.sum().asscalar())
                    color_loss.append(0.0)
                    cva_loss.append(l_cva.sum().asscalar())
                total_loss.append(l_total.sum().asscalar())
            # Sum loss from all inferences on the batch
            epoch_seg_loss['val'] += sum(seg_loss)
            epoch_bound_loss['val'] += sum(bound_loss)
            epoch_dist_loss['val'] += sum(dist_loss)
            epoch_color_loss['val'] += sum(color_loss)
            epoch_cva_loss['val'] += sum(cva_loss)
            epoch_total_loss['val'] += sum(total_loss)

        # After batch loop take the mean of batches losses
        n_batches_val = len(dataloader['val'])
        epoch_seg_loss['val'] /= n_batches_val
        epoch_bound_loss['val'] /= n_batches_val
        epoch_dist_loss['val'] /= n_batches_val
        epoch_color_loss['val'] /= n_batches_val
        epoch_cva_loss['val'] /= n_batches_val
        if args.multitasking:
            # tasks_weights = [1.0, args.wbound, args.wdist, args.wcolor, args.wcva]
            tasks_weights = [1.0, args.wbound, args.wdist, args.wcva]
            epoch_total_loss['val'] = (epoch_total_loss['val'] / n_batches_val) / sum(tasks_weights)
        else:
            epoch_total_loss['val'] = (epoch_total_loss['val'] / n_batches_val)

        _, epoch_seg_acc['val'] = acc_metric.get()
        _, epoch_seg_acc_def = acc_metric_def.get()
        _, epoch_cva_acc['val'] = acc_metric_cva.get()
        _, epoch_seg_mcc = mcc_metric.get()

        # Show metrics ---------------------------------------------------------
        metrics_table = PrettyTable()
        metrics_table.title = f'Epoch: {epoch}'
        metrics_table.field_names = ['Task', 'Loss', 'Val Loss',
                                     'Acc %', 'Val Acc %', 'Val Def Acc %']

        metrics_table.add_row(['Seg', round(epoch_seg_loss['train'], 5),
                               round(epoch_seg_loss['val'], 5),
                               round(100*epoch_seg_acc['train'], 5),
                               round(100*epoch_seg_acc['val'], 5),
                               round(epoch_seg_mcc, 5)])

        metrics_table.add_row(['Bound', round(epoch_bound_loss['train'], 5),
                               round(epoch_bound_loss['val'], 5), 0, 0, 0])

        metrics_table.add_row(['Dist', round(epoch_dist_loss['train'], 5),
                               round(epoch_dist_loss['val'], 5), 0, 0, 0])

        metrics_table.add_row(['Color', round(epoch_color_loss['train'], 5),
                               round(epoch_color_loss['val'], 5), 0, 0, 0])

        metrics_table.add_row(['CVA', round(epoch_cva_loss['train'], 5),
                              round(epoch_cva_loss['val'], 5),
                              round(100*epoch_cva_acc['train'], 5),
                              round(100*epoch_cva_acc['val'], 5), 0])

        metrics_table.add_row(['Total', round(epoch_total_loss['train'], 5),
                               round(epoch_total_loss['val'], 5), 0, 0, 0])

        print(metrics_table)

        # Add tensorboard scalars ----------------------------------------------

        add_tensorboard_scalars(summary_writer, args.results_path, epoch,
                                'Segmentation', epoch_seg_loss,
                                acc=epoch_seg_acc, val_mcc=epoch_seg_mcc)

        if args.multitasking:
            add_tensorboard_scalars(summary_writer, args.results_path, epoch,
                                    'Boundary', epoch_bound_loss)

            add_tensorboard_scalars(summary_writer, args.results_path, epoch,
                                    'Distance', epoch_dist_loss)

            add_tensorboard_scalars(summary_writer, args.results_path, epoch,
                                    'Color', epoch_color_loss)

            add_tensorboard_scalars(summary_writer, args.results_path, epoch,
                                    'CVA', epoch_cva_loss, acc=epoch_cva_acc)

        add_tensorboard_scalars(summary_writer, args.results_path, epoch,
                                'Total', epoch_total_loss)

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
    summary_writer.close()

    # Save hyperparamters ------------------------------------------------------
    if wce_weights is not None:
        wce_weights = wce_weights.asnumpy()

    data = {'Model': args.model,
            'Dataset': args.dataset_type,
            'Epochs': epoch, 'Optimizer': args.optimizer,
            'Learning Rate': args.learning_rate,
            'Weight decay': args.weight_decay, 'Patch size': args.patch_size,
            'Batch size': args.batch_size, 'Loss': args.loss,
            'Data aug': args.data_aug, 'Strong aug': args.strong_aug,
            'WCE weights': wce_weights}

    if 'resuneta' in args.model:
        data['Bound weight'] = args.wbound
        data['Dist weight'] = args.wdist
        data['Color weight'] = args.wcolor
        data['CVA weight'] = args.wcva

    data_main = {'Hyperparameter': data.keys(), 'Value': data.values()}

    df = pd.DataFrame(data_main, columns=data_main.keys())

    ax = plt.subplot(111, frame_on=False)  # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis

    table(ax, df, loc='center')  # where df is your data frame

    dataset_path = args.dataset_path.split('/')[1]
    plt.savefig(os.path.join(args.results_path, f'hyperparamters_{dataset_path}.jpg'), dpi=200,
                bbox_inches='tight')



# End functions definition -----------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="choose resunet-a model or not",
                        type=str, choices=['resuneta', 'resuneta_small', 'unet', 'unet_small'], default='resuneta')
    parser.add_argument("--multitasking", help="choose resunet-a multitasking \
                        or not", action='store_true')

    parser.add_argument("--dataset_type", help="choose which dataset to use",
                        type=str, choices=['amazon', 'ISPRS'], default='ISPRS')
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
    parser.add_argument("--loss", help="choose which loss you want to use | \
                        [weighted_categorical_cross_entropy, cross_entropy, \
                        tanimoto with dual loss, focal loss]",
                        type=str, default='tanimoto',
                        choices=['wce', 'ce', 'tanimoto', 'focal'])
    parser.add_argument("--alpha", help="Choose Alpha value for focal \
                        loss if you use it.",
                        type=float, default=0.25)
    parser.add_argument("--gamma", help="Choose Gamma value for focal \
                        loss if you use it.",
                        type=int, default=2)

    parser.add_argument("-lr", "--learning_rate",
                        help="Learning rate on training",
                        type=float, default=1e-4)
    parser.add_argument("-optm", "--optimizer",
                        help="Choose which optmizer to use",
                        type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument("--momentum", help="SGD momemtum's. \
                        Should be used along with SGD optmizer",
                        type=float, default=0.0)
    parser.add_argument("-wd", "--weight_decay", help="Amount of weight decay",
                        type=float, default=0.0)

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
    parser.add_argument("--wcva", help="CVA loss weight",
                        type=float, default=1.0)
    parser.add_argument("--no_color", help="Don't use HSV transform task",
                        action='store_true')

    parser.add_argument("--groups", help="Groups to be used in convolutions",
                        type=int, default=1)

    parser.add_argument("--wce_weights", type=int, default=1,
                        help="Weights to be used on Weighted Cross Entropy")

    parser.add_argument("--data_aug",
                        help="Use data augmentation or not",
                        action='store_true')
    parser.add_argument("--strong_aug",
                        help="Use Strong data augmentation along with usual ones",
                        action='store_true')
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

    if args.loss == 'tanimoto':
        from_logits = True
    else:
        from_logits = False

    Nfilters_init = 32
    if args.model == 'resuneta':
        args.multitasking = True
        net = ResUNet_d6(args.dataset_type, Nfilters_init, args.num_classes,
                         patch_size=args.patch_size, verbose=args.debug,
                         from_logits=from_logits)
    if args.model == 'resuneta_small':
     args.multitasking = True
     net = ResUNet_d6(args.dataset_type, Nfilters_init, args.num_classes,
                      patch_size=args.patch_size, verbose=args.debug,
                      from_logits=from_logits, small=True)
    elif args.model == 'unet':
        # Changed from 64 to 32
        net = UNet(args.num_classes, groups=args.groups, nfilter=64)
        args.multitasking = False
        # net = UNet(input_channels=14, output_channels=args.num_classes)
    elif args.model == 'unet_small':
        # Changed from 64 to 32
        net = UNet_small(args.num_classes, groups=args.groups, nfilter=32)
        args.multitasking = False
    net.initialize()

    if args.dataset_type == 'ISPRS':
        net.summary(mx.nd.random.uniform(shape=(args.batch_size, 3, args.patch_size, args.patch_size)))
    else:
        net.summary(mx.nd.random.uniform(shape=(args.batch_size, 14, args.patch_size, args.patch_size)))

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

    if args.dataset_type == 'ISPRS':
        color_channels = 3
        tnorm = Normalize()
    else:
        mean = np.array([6884.0415,  6188.7725,  5670.7944,  4999.9517,
                         11815.252, 7389.2964, 5174.3335, 6844.906, 6138.854,
                         5629.7964,  4927.634,  12209.005, 7420.211, 5139.2495])
        std = np.array([4014.3228, 3610.9395, 3317.272,  2939.187,  7004.3374,
                        4432.908,  3082.9048, 3990.8325, 3581.0767, 3293.6165,
                        2893.691, 7257.484, 4430.521, 3046.775])
        # tnorm = Normalize(mean=mean, std=std)
        tnorm = None
        color_channels = 6

    if args.strong_aug:
        p_strong = 0.8
    else:
        p_strong = 0.0

    if args.data_aug:
        prob = 0.7
        aug = A.Compose([
            A.OneOf([A.HorizontalFlip(p=prob), A.VerticalFlip(p=prob)], p=1),
            A.RandomRotate90(p=prob),
            A.RandomSizedCrop(min_max_height=(60, 100),
                              height=args.patch_size, width=args.patch_size,
                              p=prob),
            A.OneOf([
                A.ElasticTransform(p=prob, alpha=120, sigma=120 * 0.05,
                                   alpha_affine=120 * 0.03),
                A.GridDistortion(p=prob),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=prob)],
                    p=p_strong),
            ])
    else:
        aug = None

    train_dataset = ISPRSDataset(root=args.dataset_path,
                                 mode='train',
                                 mtsk=args.multitasking, norm=tnorm,
                                 transform=aug, color_channels=color_channels)
    val_dataset = ISPRSDataset(root=args.dataset_path,
                               mode='val',
                               mtsk=args.multitasking, norm=tnorm,
                               transform=None, color_channels=color_channels)

    dataloader = {}
    dataloader['train'] = gluon.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                last_batch='rollover',
                                                num_workers=8, pin_memory=True)
    dataloader['val'] = gluon.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              last_batch='rollover',
                                              num_workers=8, pin_memory=True)

    # DEBUG
    # dataloader['train'] = gluon.data.DataLoader(train_dataset,
    #                                           batch_size=args.batch_size,
    #                                           shuffle=True,
    #                                           last_batch='rollover',
    #                                           pin_memory=True)
    # dataloader['val'] = gluon.data.DataLoader(val_dataset,
    #                                         batch_size=args.batch_size,
    #                                         shuffle=True,
    #                                         last_batch='rollover',
    #                                         pin_memory=True)

    print('='*40)
    logger.info(f'Training on {len(train_dataset)} images')
    logger.info(f'Validating on {len(val_dataset)} images')
    print('='*40)

    log_path = os.path.join(args.results_path, 'logs')
    summary_writer = SummaryWriter(logdir=log_path, verbose=False)

    train_model(args, net, dataloader, devices, summary_writer, from_logits)
