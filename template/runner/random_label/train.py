# Utils
import logging
import time

# Torch related stuff
import numpy as np
import torch
from tqdm import tqdm

from util.evaluation.metrics import accuracy
# DeepDIVA
from util.misc import AverageMeter


def train(train_loader, model, criterion, optimizer, observer, observer_criterion, observer_optimizer,
          writer, epoch, no_cuda=False, log_interval=25, **kwargs):
    """
    Training routine

    Parameters
    ----------
    :param train_loader : torch.utils.data.DataLoader
        The dataloader of the train set.

    :param model : torch.nn.module
        The network model being used.

    :param criterion : torch.nn.loss
        The loss function used to compute the loss of the model.

    :param optimizer : torch.optim
        The optimizer used to perform the weight update.

    :param writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    :param epoch : int
        Number of the epoch (for logging purposes).

    :param no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.

    :param log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    :return:
        None
    """
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    observer_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    observer_acc_meter = AverageMeter()
    data_time = AverageMeter()

    # Switch to train mode (turn on dropout & stuff)
    model.train()

    # Create a random object
    random_seed = 42
    random1 = np.random.RandomState(random_seed)
    num_classes = observer.module.output_channels

    # Iterate over whole training set
    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target) in pbar:

        # Measure data loading time
        data_time.update(time.time() - end)

        # Generate the shuffled labels
        # random1 = np.random.RandomState(random_seed)
        random_target = torch.LongTensor(random1.randint(0, num_classes, len(input)))

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
            random_target = random_target.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input)
        random_target_var = torch.autograd.Variable(random_target)

        acc, loss = train_one_mini_batch(model, criterion, optimizer, input_var, random_target_var, loss_meter, acc_meter)

        # Update random if necessary
        # if acc[0] > 80:
        #     logging.info('Random seed updated!')
        #     random_seed = random1.randint(0)

        input_features_var =  torch.autograd.Variable(model.module.features.data)
        target_var = torch.autograd.Variable(target)

        observer_acc, observer_loss = train_one_mini_batch(observer, observer_criterion, observer_optimizer, input_features_var, target_var, observer_loss_meter, observer_acc_meter)

        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar('train/mb_loss', loss.data[0], epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_accuracy', acc.cpu().numpy(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/obs_mb_loss', observer_loss.data[0], epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/obs_mb_accuracy', observer_acc.cpu().numpy(), epoch * len(train_loader) + batch_idx)
        else:
            writer.add_scalar('train/mb_loss_{}'.format(multi_run), loss.data[0],
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/mb_accuracy_{}'.format(multi_run), acc.cpu().numpy(),
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/obs_mb_loss_{}'.format(multi_run), observer_loss.data[0],
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('train/obs_mb_accuracy_{}'.format(multi_run), observer_acc.cpu().numpy(),
                              epoch * len(train_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log to console
        if batch_idx % log_interval == 0:
            pbar.set_description('train epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(train_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=loss_meter),
                             Acc1='{acc_meter.avg:.3f}\t'.format(acc_meter=acc_meter),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar('train/accuracy', acc_meter.avg, epoch)
        writer.add_scalar('train/obs_accuracy', observer_acc_meter.avg, epoch)
    else:
        writer.add_scalar('train/accuracy_{}'.format(multi_run), acc_meter.avg, epoch)
        writer.add_scalar('train/obs_accuracy_{}'.format(multi_run), observer_acc_meter.avg, epoch)

    logging.debug('Train epoch[{}]: '
                  'Acc@1={acc_meter.avg:.3f}\t'
                  'Loss={loss.avg:.4f}\t'
                  'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                  .format(epoch, batch_time=batch_time, data_time=data_time, loss=loss_meter, acc_meter=acc_meter))

    return acc_meter.avg


def train_one_mini_batch(model, criterion, optimizer, input_var, target_var, loss_meter, acc_meter):
    # Compute output
    output = model(input_var)

    # Compute and record the loss
    loss = criterion(output, target_var)
    loss_meter.update(loss.data[0], len(input_var))

    # Compute and record the accuracy
    acc1 = accuracy(output.data, target_var.data, topk=(1,))[0]
    acc_meter.update(acc1[0], len(input_var))

    # Reset gradient
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Perform a step by updating the weights
    optimizer.step()

    return acc1, loss
