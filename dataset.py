import os, sys
import numpy as np
from tqdm.auto import tqdm
import scipy.spatial as spatial
import jittor as jt

# All shapes of PCPNet dataset
all_train_sets = ['fandisk100k', 'bunny100k', 'armadillo100k', 'dragon_xyzrgb100k', 'boxunion_uniform100k',
                  'tortuga100k', 'flower100k', 'Cup33100k']
all_test_sets = ['galera100k', 'icosahedron100k', 'netsuke100k', 'Cup34100k', 'sphere100k',
                 'cylinder100k', 'star_smooth100k', 'star_halfsmooth100k', 'star_sharp100k', 'Liberty100k',
                 'boxunion2100k', 'pipe100k', 'pipe_curve100k', 'column100k', 'column_head100k',
                 'Boxy_smooth100k', 'sphere_analytic100k', 'cylinder_analytic100k', 'sheet_analytic100k']
all_val_sets = ['cylinder100k', 'galera100k', 'netsuke100k']

def load_data(filedir, filename, dtype=np.float32, wo=False):
    d = None
    filepath = os.path.join(filedir, 'npy', filename + '.npy')
    os.makedirs(os.path.join(filedir, 'npy'), exist_ok=True)
    if os.path.exists(filepath):
        if wo:
            return True
        d = np.load(filepath)
    else:
        d = np.loadtxt(os.path.join(filedir, filename), dtype=dtype)
        np.save(filepath, d)
    return d


class PCATrans(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # compute PCA of points in the patch, center the patch around the mean
        pts = data['pcl_pat']
        pts_mean = pts.mean(0)
        pts = pts - pts_mean

        trans, _, _ = jt.linalg.svd(jt.t(pts))  # (3, 3)
        pts = jt.matmul(pts, trans)

        # since the patch was originally centered, the original cp was at (0,0,0)
        cp_new = -pts_mean
        cp_new = jt.matmul(cp_new, trans)

        # re-center on original center point
        data['pcl_pat'] = pts - cp_new
        data['pca_trans'] = trans

        if 'center_normal' in data:
            data['center_normal'] = jt.matmul(data['center_normal'], trans)
        return data


class SequentialPointcloudPatchSampler(jt.dataset.Sampler):
    def __init__(self, datasets):
        self.datasets = datasets
        self.total_patch_count = sum(datasets.shape_patch_count)

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class RandomPointcloudPatchSampler(jt.dataset.Sampler):
    # randomly get subset data from the whole dataset
    def __init__(self, datasets, patches_per_shape, seed=None, identical_epochs=False):
        self.datasets = datasets
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self._shuffle_rng = np.random.default_rng(self.seed) 

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(datasets.shape_names):
            self.total_patch_count += min(self.patches_per_shape, datasets.shape_patch_count[shape_ind])

    def __iter__(self):
        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self._shuffle_rng.integers(low=0, high=sum(self.datasets.shape_patch_count), size=(self.total_patch_count,), dtype=np.int64).tolist())

    def __len__(self):
        return self.total_patch_count


class PointCloudDataset(jt.dataset.Dataset):
    def __init__(self, root, mode=None, data_set='', data_list='', sparse_patches=False):
        super().__init__()
        self.mode = mode
        self.data_set = data_set
        self.sparse_patches = sparse_patches
        self.data_dir = os.path.join(root, data_set)

        self.pointclouds = []
        self.pointclouds_clean = []
        self.shape_names = []
        self.normals = []
        self.pidxs = []
        self.kdtrees = []
        self.kdtrees_clean = []
        self.shape_patch_count = []   # point number of each shape
        assert self.mode in ['train', 'val', 'test']

        if len(data_list) > 0:
            # get all shape names
            cur_sets = []
            with open(os.path.join(root, data_set, 'list', data_list)) as f:
                cur_sets = f.readlines()
            cur_sets = [x.strip() for x in cur_sets]
            cur_sets = list(filter(None, cur_sets))

        print('Current %s dataset:' % self.mode)
        for s in cur_sets:
            print('   ', s)

        self.load_data(cur_sets)

    def load_data(self, cur_sets):
        for s in tqdm(cur_sets, desc='Loading data'):
            pcl = load_data(filedir=self.data_dir, filename='%s.xyz' % s, dtype=np.float32)[:, :3]

            if s.find('_noise_white_') == -1:
                s_clean = s
            else:
                s_clean = s.split('_noise_white_')[0]
            pcl_clean = load_data(filedir=self.data_dir, filename='%s.xyz' % s_clean, dtype=np.float32)[:, :3]

            nor = load_data(filedir=self.data_dir, filename=s_clean + '.normals', dtype=np.float32)

            self.pointclouds.append(pcl)
            self.pointclouds_clean.append(pcl_clean)
            self.normals.append(nor)
            self.shape_names.append(s)

            # KDTree construction may run out of recursions
            sys.setrecursionlimit(int(max(1000, round(pcl.shape[0]/10))))
            kdtree = spatial.cKDTree(pcl, 10)
            self.kdtrees.append(kdtree)

            kdtree_clean = spatial.cKDTree(pcl_clean, 10)
            self.kdtrees_clean.append(kdtree_clean)

            if self.sparse_patches:
                pidx = load_data(filedir=self.data_dir, filename='%s.pidx' % s, dtype=np.int32)
                self.pidxs.append(pidx)
                self.shape_patch_count.append(len(pidx))
            else:
                self.shape_patch_count.append(pcl.shape[0])

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        # KDTree uses a reference, not a copy of these points,
        # so modifying the points would make the kdtree give incorrect results!
        data = {
            'pcl': self.pointclouds[idx].copy(),
            'pcl_clean': self.pointclouds_clean[idx].copy(),
            'kdtree': self.kdtrees[idx],
            'kdtree_clean': self.kdtrees_clean[idx],
            'normal': self.normals[idx],
            'pidx': self.pidxs[idx] if len(self.pidxs) > 0 else None,
            'name': self.shape_names[idx],
        }
        return data


class PatchDataset(jt.dataset.Dataset):
    def __init__(self, datasets, patch_size=1, with_trans=True, batch_size=32, sampler=None, num_workers=0):
        super().__init__()
        self.datasets = datasets
        self.patch_size = patch_size
        self.trans = None
        if with_trans:
            self.trans = PCATrans()

        self.sampler = sampler
        self.set_attrs(batch_size = batch_size, total_len=sum(datasets.shape_patch_count), num_workers=num_workers)
        
    def shape_index(self, index):
        """
            Translate global (dataset-wide) point index to shape index & local (shape-wide) point index
        """
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.datasets.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset  # index in shape with ID shape_ind
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count
        return shape_ind, shape_patch_ind

    def make_patch(self, pcl, kdtree=None, kdtree_clean=None, seed_idx=None, patch_size=1):
        """
        Args:
            pcl: (N, 3)
            kdtree:
            nor: (N, 3)
            seed_idx: (P,)
            patch_size: K
        Returns:
            pcl_pat, nor_pat: (P, K, 3)
        """
        seed_pnts = pcl[seed_idx, :]
        dists, pat_idx = kdtree.query(seed_pnts, k=patch_size)  # sorted by distance (nearest first)
        dist_max = max(dists)

        pcl_pat = pcl[pat_idx, :]        # (K, 3)
        pcl_pat = pcl_pat - seed_pnts    # center
        pcl_pat = pcl_pat / dist_max     # normlize

        _, nor_idx = kdtree_clean.query(seed_pnts)

        return pcl_pat, nor_idx

    def __getitem__(self, idx):
        """
            Returns a patch centered at the point with the given global index
            and the ground truth normal of the patch center
        """
        # find shape that contains the point with given global index
        shape_idx, patch_idx = self.shape_index(idx)
        shape_data = self.datasets[shape_idx]

        # get the query point
        if shape_data['pidx'] is None:
            center_point_idx = patch_idx
        else:
            center_point_idx = shape_data['pidx'][patch_idx]

        pcl_pat, normal_idx = self.make_patch(pcl=shape_data['pcl'],
                                              kdtree=shape_data['kdtree'],
                                              kdtree_clean=shape_data['kdtree_clean'],
                                              seed_idx=center_point_idx,
                                              patch_size=self.patch_size,
                                             )
        data = {
            'pcl_pat': jt.array(pcl_pat),
            'center_normal': jt.array(shape_data['normal'][normal_idx, :]),
            'name': shape_data['name'],
        }

        if self.trans is not None:
            data = self.trans(data)
        return data


if __name__ == '__main__':
    root = './dataset/'
    data_set = 'PCPNet'
    data_list = 'testset_%s.txt' % data_set

    test_dset = PointCloudDataset(
            root=root,
            mode='test',
            data_set=data_set,
            data_list=data_list,
        )
    test_set = PatchDataset(
            datasets=test_dset,
            patch_size=700,
        )
    test_dataloader = jt.dataset.DataLoader(
            test_set,
            sampler=SequentialPointcloudPatchSampler(test_set),
            batch_size=10,
            num_workers=1,
        )

    for batchind, data in enumerate(test_dataloader, 0):
        print(data['pcl_pat'].size())




