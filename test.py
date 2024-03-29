import os, sys
import shutil
import time
import argparse
import jittor as jt
import jittor.nn as nn
import numpy as np

from net.CMG_Net import Network
from utils.misc import get_logger, seed_all
from dataset import PointCloudDataset, PatchDataset, SequentialPointcloudPatchSampler, load_data
import scipy.spatial as spatial

jt.flags.use_cuda = jt.has_cuda

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='../')
    parser.add_argument('--data_set', type=str, default='PCPNet')
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--ckpt_dirs', type=str, default='001', help="can be multiple directories, separated by ',' ")
    parser.add_argument('--ckpt_iter', type=str, default='900')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--testset_list', type=str, default='testset_all.txt')
    parser.add_argument('--eval_list', type=str,
                        default=['testset_no_noise.txt', 'testset_low_noise.txt', 'testset_med_noise.txt', 'testset_high_noise.txt',
                                 'testset_vardensity_striped.txt', 'testset_vardensity_gradient.txt'],
                        nargs='*', help='list of .txt files containing sets of point cloud names for evaluation')
    parser.add_argument('--patch_size', type=int, default=800)
    parser.add_argument('--knn_l1', type=int, default=16)
    parser.add_argument('--knn_l2', type=int, default=32)
    parser.add_argument('--knn_h1', type=int, default=32)
    parser.add_argument('--knn_h2', type=int, default=16)
    parser.add_argument('--knn_d', type=int, default=16)
    parser.add_argument('--sparse_patches', type=eval, default=True, choices=[True, False],
                        help='test on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--save_pn', type=eval, default=False, choices=[True, False])
    parser.add_argument('--matric', type=str, default='CND', choices=['CND', 'RMSE'])
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    test_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='test',
            data_set=args.data_set,
            data_list=args.testset_list,
            sparse_patches=args.sparse_patches,
        )
    test_datasampler = SequentialPointcloudPatchSampler(test_dset)
    test_dataloader = PatchDataset(
            datasets=test_dset,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            sampler=test_datasampler
        )
    return test_dset, test_dataloader, test_datasampler

### Arguments
args = parse_arguments()
arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
print('Arguments:\n %s\n' % arg_str)

seed_all(args.seed)
PID = os.getpid()

### Datasets and loaders
test_dset, test_dataloader, test_datasampler = get_data_loaders(args)


def normal_error(normal_gts, normal_preds, eval_file='log.txt', matric='CND'):
    """
        Compute normal root-mean-square error (RMSE)
    """
    def l2_norm(v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    log_file = open(eval_file, 'w')
    def log_string(out_str):
        log_file.write(out_str+'\n')
        log_file.flush()

    errors   = []
    errors_o = []
    pgp30 = []
    pgp25 = []
    pgp20 = []
    pgp15 = []
    pgp10 = []
    pgp5  = []
    pgp_alpha = []

    for i in range(len(normal_gts)):
        normal_gt = normal_gts[i]
        normal_pred = normal_preds[i]

        normal_gt_norm = l2_norm(normal_gt)
        normal_results_norm = l2_norm(normal_pred)
        normal_pred = np.divide(normal_pred, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normal_gt = np.divide(normal_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        ### Unoriented rms
        nn = np.sum(np.multiply(normal_gt, normal_pred), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        ang = np.rad2deg(np.arccos(np.abs(nn)))

        ### Error metric
        errors.append(np.sqrt(np.mean(np.square(ang))))
        ### Portion of good points
        pgp30_shape = sum([j < 30.0 for j in ang]) / float(len(ang))
        pgp25_shape = sum([j < 25.0 for j in ang]) / float(len(ang))
        pgp20_shape = sum([j < 20.0 for j in ang]) / float(len(ang))
        pgp15_shape = sum([j < 15.0 for j in ang]) / float(len(ang))
        pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))
        pgp5_shape  = sum([j < 5.0 for j in ang])  / float(len(ang))
        pgp30.append(pgp30_shape)
        pgp25.append(pgp25_shape)
        pgp20.append(pgp20_shape)
        pgp15.append(pgp15_shape)
        pgp10.append(pgp10_shape)
        pgp5.append(pgp5_shape)

        pgp_alpha_shape = []
        for alpha in range(30):
            pgp_alpha_shape.append(sum([j < alpha for j in ang]) / float(len(ang)))

        pgp_alpha.append(pgp_alpha_shape)

        # Oriented rms
        errors_o.append(np.sqrt(np.mean(np.square(np.rad2deg(np.arccos(nn))))))

    avg_errors   = np.mean(errors)
    avg_errors_o = np.mean(errors_o)
    avg_pgp30 = np.mean(pgp30)
    avg_pgp25 = np.mean(pgp25)
    avg_pgp20 = np.mean(pgp20)
    avg_pgp15 = np.mean(pgp15)
    avg_pgp10 = np.mean(pgp10)
    avg_pgp5  = np.mean(pgp5)
    avg_pgp_alpha = np.mean(np.array(pgp_alpha), axis=0)

    log_string('%s per shape: ' % matric + str(errors))
    log_string('%s not oriented (shape average): ' % matric + str(avg_errors))
    log_string('%s oriented (shape average): ' % matric + str(avg_errors_o))
    log_string('PGP30 per shape: ' + str(pgp30))
    log_string('PGP25 per shape: ' + str(pgp25))
    log_string('PGP20 per shape: ' + str(pgp20))
    log_string('PGP15 per shape: ' + str(pgp15))
    log_string('PGP10 per shape: ' + str(pgp10))
    log_string('PGP5 per shape: ' + str(pgp5))
    log_string('PGP30 average: ' + str(avg_pgp30))
    log_string('PGP25 average: ' + str(avg_pgp25))
    log_string('PGP20 average: ' + str(avg_pgp20))
    log_string('PGP15 average: ' + str(avg_pgp15))
    log_string('PGP10 average: ' + str(avg_pgp10))
    log_string('PGP5 average: ' + str(avg_pgp5))
    log_string('PGP alpha average: ' + str(avg_pgp_alpha))
    log_file.close()

    return avg_errors


def test(ckpt_dir, ckpt_iter):
    ### Input/Output
    ckpt_path = os.path.join(args.log_root, ckpt_dir, 'ckpts/ckpt_%s.pkl' % ckpt_iter)
    output_dir = os.path.join(args.log_root, ckpt_dir, 'results_%s/ckpt_%s' % (args.data_set, ckpt_iter))
    if args.tag is not None and len(args.tag) != 0:
        output_dir += '_' + args.tag
    if not os.path.exists(ckpt_path):
        print('ERROR path: %s' % ckpt_path)
        return False, False

    file_save_dir = os.path.join(output_dir, 'pred_normal')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(file_save_dir, exist_ok=True)

    logger = get_logger('test(%d)(%s-%s)' % (PID, ckpt_dir, ckpt_iter), output_dir)
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    ### Model
    logger.info('Loading model: %s' % ckpt_path)
    ckpt = jt.load(ckpt_path)
    model = Network(num_in=args.patch_size,
                    knn_l1=args.knn_l1,
                    knn_l2=args.knn_l2,
                    knn_h1=args.knn_h1,
                    knn_h2=args.knn_h2,
                    knn_d=args.knn_d,
                    )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Num_params: %d' % num_params)
    logger.info('Num_params_trainable: %d' % trainable_num)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    shape_ind = 0
    shape_patch_offset = 0
    shape_num = len(test_dset.shape_names)
    shape_patch_count = test_dset.shape_patch_count[shape_ind]

    num_batch = len(test_dataloader)
    normal_prop = jt.array(np.zeros((shape_patch_count, 3), dtype='float32'))

    jt.sync_all(True)
    total_time = 0
    for batchind, data in enumerate(test_dataloader):
        pcl_pat = data['pcl_pat']        # (B, N, 3)
        data_trans = data['pca_trans']

        start_time = time.time()
        with jt.no_grad():
            n_est, weights, trans = model(pcl_pat)
        end_time = time.time()
        n_est.sync()
        weights.sync()
        trans.sync()
        elapsed_time = 1000 * (end_time - start_time)  # ms
        total_time += elapsed_time

        if batchind % 5 == 0:
            batchSize = pcl_pat.size()[0]
            logger.info('[%d/%d] %s: elapsed_time per point/patch: %.3f ms' % (
                        batchind, num_batch-1, test_dset.shape_names[shape_ind], elapsed_time / batchSize))

        n_est[:, :] = nn.bmm(n_est.unsqueeze(1), trans.transpose(2, 1)).squeeze(dim=1)
        if data_trans is not None:
            ### transform predictions with inverse PCA rotation (back to world space)
            n_est[:, :] = nn.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)
            
        ### Save the estimated normals to file
        batch_offset = 0
        while batch_offset < n_est.shape[0] and shape_ind + 1 <= shape_num:
            shape_patches_remaining = shape_patch_count - shape_patch_offset
            batch_patches_remaining = n_est.shape[0] - batch_offset

            ### append estimated patch properties batch to properties for the current shape on the CPU
            normal_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining), :] = \
                n_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]

            batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
            shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

            if shape_patches_remaining <= batch_patches_remaining:
                normals_to_write = normal_prop.numpy()

                ### for faster reading speed in the evaluation
                save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '_normal.npy')
                np.save(save_path, normals_to_write)
                if args.save_pn:
                    save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '.normals')
                    np.savetxt(save_path, normals_to_write)
                logger.info('saved normal: {} \n'.format(save_path))

                sys.stdout.flush()
                shape_patch_offset = 0
                shape_ind += 1
                if shape_ind < shape_num:
                    shape_patch_count = test_dset.shape_patch_count[shape_ind]
                    normal_prop = jt.zeros([shape_patch_count, 3])
    
    jt.sync_all(True)

    logger.info('Total Time: %.2f s, Shape Num: %d' % (total_time/1000, shape_num))
    return output_dir, file_save_dir


def eval(normal_gt_path, normal_pred_path, output_dir):
    print('\n  Evaluation ...')
    eval_summary_dir = os.path.join(output_dir, 'test_summary')
    os.makedirs(eval_summary_dir, exist_ok=True)

    all_avg_errors = []
    for cur_list in args.eval_list:
        print("\n***************** " + cur_list + " *****************")
        print("Result path: " + normal_pred_path)

        ### get all shape names in the list
        shape_names = []
        normal_gt_filenames = os.path.join(normal_gt_path, 'list', cur_list)
        with open(normal_gt_filenames) as f:
            shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))

        ### load all shapes
        normal_gts = []
        normal_preds = []
        for shape in shape_names:
            print(shape)
            shape_gt = shape.split('_noise_white_')[0]
            xyz_ori = load_data(filedir=normal_gt_path, filename=shape + '.xyz', dtype=np.float32)
            xyz_gt = load_data(filedir=normal_gt_path, filename=shape_gt + '.xyz', dtype=np.float32)
            normal_gt = load_data(filedir=normal_gt_path, filename=shape_gt + '.normals', dtype=np.float32)  # (N, 3)
            normal_pred = np.load(os.path.join(normal_pred_path, shape + '_normal.npy'))                  # (n, 3)
            ### eval with sparse point sets
            points_idx = load_data(filedir=normal_gt_path, filename=shape + '.pidx', dtype=np.int32)      # (n,)
            sys.setrecursionlimit(int(max(1000, round(xyz_gt.shape[0] / 10))))
            kdtree = spatial.cKDTree(xyz_gt, 10)
            qurey_points = xyz_ori[points_idx, :]
            _, nor_idx = kdtree.query(qurey_points)
            if args.matric == 'CND':
                normal_gt = normal_gt[nor_idx, :]
            elif args.matric == 'RMSE':
                normal_gt = normal_gt[points_idx, :]
            if normal_pred.shape[0] > normal_gt.shape[0]:
                normal_pred = normal_pred[points_idx, :]

            normal_gts.append(normal_gt)
            normal_preds.append(normal_pred)

        ### compute CND per-list
        avg_errors = normal_error(normal_gts=normal_gts,
                              normal_preds=normal_preds,
                              eval_file=os.path.join(eval_summary_dir, cur_list[:-4] + '_evaluation_results.txt'),
                              matric=args.matric)
        all_avg_errors.append(avg_errors)
        print('%s: %f' % (args.matric, avg_errors))

    s = ('\n {} \n All %s not oriented (shape average): {} | Mean: {}\n' % args.matric).format(
                normal_pred_path, str(all_avg_errors), np.mean(all_avg_errors))
    print(s)

    ### delete the output point normals
    if not args.save_pn:
        shutil.rmtree(normal_pred_path)
    return all_avg_errors



if __name__ == '__main__':
    ckpt_dirs = args.ckpt_dirs.split(',')

    for ckpt_dir in ckpt_dirs:
        eval_dict = ''
        sum_file = 'eval_' + args.data_set + ('_'+args.tag if len(args.tag) != 0 else '')
        log_file_sum = open(os.path.join(args.log_root, ckpt_dir, sum_file+'.txt'), 'a')
        log_file_sum.write('\n====== %s ======\n' % args.eval_list)

        output_dir, file_save_dir = test(ckpt_dir=ckpt_dir, ckpt_iter=args.ckpt_iter)
        if not output_dir or args.data_set == 'Semantic3D':
            continue
        all_avg_errors = eval(normal_gt_path=os.path.join(args.dataset_root, args.data_set),
                              normal_pred_path=file_save_dir,
                              output_dir=output_dir)

        s = '%s: %s | Mean: %f\n' % (args.ckpt_iter, str(all_avg_errors), np.mean(all_avg_errors))
        log_file_sum.write(s)
        log_file_sum.flush()
        eval_dict += s

        log_file_sum.close()
        s = ('\n All %s not oriented (shape average): \n{}\n' % args.matric).format(eval_dict)
        print(s)


