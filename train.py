import os, sys
import argparse
import math

import jittor as jt
import jittor.nn as nn

from utils.misc import *
from net.CMG_Net import Network
from dataset import PointCloudDataset, PatchDataset, RandomPointcloudPatchSampler

jt.flags.use_cuda = jt.has_cuda

def parse_arguments():
    parser = argparse.ArgumentParser()
    ### Training
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--nepoch', type=int, default=900)
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--max_grad_norm', type=float, default=float('inf'))
    ### Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='../')
    parser.add_argument('--data_set', type=str, default='PCPNet', choices=['PCPNet'])
    parser.add_argument('--trainset_list', type=str, default='trainingset_whitenoise.txt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=700)
    parser.add_argument('--knn_l1', type=int, default=16)
    parser.add_argument('--knn_l2', type=int, default=32)
    parser.add_argument('--knn_h1', type=int, default=32)
    parser.add_argument('--knn_h2', type=int, default=16)
    parser.add_argument('--knn_d', type=int, default=16)
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='The number of patches sampled from each shape in an epoch')
    args = parser.parse_args()
    return args

def get_data_loaders(args):
    def worker_init_fn(worker_id):
        random.seed(args.seed)
        np.random.seed(args.seed)


    train_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='train',
            data_set=args.data_set,
            data_list=args.trainset_list,
        )
    train_datasampler = RandomPointcloudPatchSampler(train_dset, patches_per_shape=args.patches_per_shape, seed=args.seed)
    train_dataloader = PatchDataset(
            datasets=train_dset,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            sampler=train_datasampler,
        )

    return train_dataloader, train_datasampler

### Arguments
args = parse_arguments()
seed_all(args.seed)

PID = os.getpid()

### Datasets and loaders
print('Loading datasets ...')
train_dataloader, train_datasampler = get_data_loaders(args)
train_num_batch = (len(train_datasampler) - 1) // args.batch_size + 1

### Model
print('Building model ...')
model = Network(num_in=args.patch_size,
                knn_l1=args.knn_l1,
                knn_l2=args.knn_l2,
                knn_h1=args.knn_h1,
                knn_h2=args.knn_h2,
                knn_d=args.knn_d,
            )

### Optimizer and Scheduler
optimizer = nn.Adam(model.parameters(), lr=args.lr)
lambda1 = lambda epoch: (0.99*epoch/100 + 0.01) if epoch < 100 else  2e-3  if (0.5 * (1+math.cos(math.pi*(epoch-100)/(args.nepoch-200)))<2e-3 or epoch > 800) else 0.5 * (1+math.cos(math.pi*(epoch-100)/(args.nepoch-200)))
scheduler = nn.LambdaLR(optimizer, lr_lambda=lambda1)

# ### Logging
if args.logging:
    log_path, log_dir_name = get_new_log_dir(args.log_root, prefix='',
                                            postfix='_' + args.tag if args.tag is not None else '')
    sub_log_dir = os.path.join(log_path, 'log')
    os.makedirs(sub_log_dir)
    logger = get_logger(name='train(%d)(%s)' % (PID, log_dir_name), log_dir=sub_log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()

refine_epoch = -1
if args.resume != '':
    assert os.path.exists(args.resume), 'ERROR path: %s' % args.resume
    logger.info('Resume from: %s' % args.resume)

    ckpt = jt.load(args.resume)
    model.load_state_dict(ckpt['state_dict'])
    refine_epoch = ckpt['others']['epoch']

    logger.info('Load pretrained mode: %s' % args.resume)

if args.logging:
    code_dir = os.path.join(log_path, 'code')
    os.makedirs(code_dir, exist_ok=True)
    os.system('cp %s %s' % ('*.py', code_dir))
    os.system('cp -r %s %s' % ('net', code_dir))
    os.system('cp -r %s %s' % ('utils', code_dir))


### Arguments
logger.info('Command: {}'.format(' '.join(sys.argv)))
arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
logger.info('Arguments:\n' + arg_str)
logger.info(repr(model))
logger.info('training set: %d patches (in %d batches)' %
                (len(train_datasampler), (len(train_datasampler) - 1) // args.batch_size + 1))


def train(epoch):
    jt.sync_all(True)
    for train_batchind, batch in enumerate(train_dataloader):
        pcl_pat = batch['pcl_pat']
        center_normal = batch['center_normal']  # (B, 3)

        ### Reset grad and model state
        model.train()
        optimizer.zero_grad()

        ### Forward
        pred_nor, weights, trans = model(pcl_pat)
        loss, loss_tuple = model.get_loss(q_target=center_normal, q_pred=pred_nor, pred_weights=weights, pcl_in=pcl_pat, trans=trans)
  
        ### Backward and optimize
        optimizer.backward(loss)
        optimizer.clip_grad_norm(args.max_grad_norm)
        optimizer.step()
        
        pred_nor.sync()
        weights.sync()
        trans.sync()

        ### Logging
        s = ''
        for l in loss_tuple:
            s += '%.5f+' % l.item()
        logger.info('[Train] [%03d: %03d/%03d] | Loss: %.6f(%s)' % (
                    epoch, train_batchind, train_num_batch-1, loss.item(), s[:-1])
                )
        jt.sync_all(True)

if __name__ == '__main__':
    logger.info('Start training ...')
    try:
        for epoch in range(1, args.nepoch+1):
            logger.info('### Epoch %d ###' % epoch)
            if epoch <= refine_epoch:
                scheduler.step()
                continue

            start_time = time.time()
            train(epoch)
            end_time = time.time()
            logger.info('Time cost: %.1f s \n' % (end_time-start_time))

            scheduler.step()

            if epoch % args.interval == 0 or epoch == args.nepoch-1:
                opt_states = {
                    'epoch': epoch,
                }

                if args.logging:
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                    }
                    jt.save(checkpoint, os.path.join(log_path, 'ckpts', 'ckpt_{:d}.pkl'.format(epoch)))

    except KeyboardInterrupt:
        logger.info('Terminating ...')
