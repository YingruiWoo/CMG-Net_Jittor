import jittor as jt
import jittor.nn as nn

def my_squeeze(x, dim=None):
    shape = list(x.shape)
    if dim is None:
        new_shape = [s for s in shape if s > 1]
        return x.reshape(new_shape)
    else:
        if dim < 0: dim += len(shape)
        assert dim < len(shape) and dim >= 0
        assert shape[dim] == 1
        return x.reshape(shape[:dim] + shape[dim+1:])

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, M, C]
        dst: target points, [B, N, C]
    Output:
        dist: per-point square distance, [B, M, N]
    """
    return jt.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def knn_group_0(x, idx):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    x = x.float()
    idx = idx.long()

    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return x.gather(2, idx)

def knn_group_1(x, idx):
    """
    :param  x:      (B, F, N)
    :param  idx:    (B, M, k)
    :return (B, F, M, k)
    """
    x = x.float()
    idx = idx.long()

    B, F, N = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(2).expand(B, F, M, N)
    idx = idx.unsqueeze(1).expand(B, F, M, k)

    return x.gather(3, idx)

def get_knn_idx(pos, query, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    dists = square_distance(query, pos)
    idx, dists = jt.argsort(dists, dim=-1)
    dists, idx = dists[:, :, offset:k+offset], idx[:, :, offset:k+offset]
    return idx[:, :, offset:]


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, with_bn=1, activation='relu'):
        super().__init__()
        assert with_bn in [0, 1, 2]
        self.with_bn = with_bn > 0 and activation is not None

        self.linear = nn.Linear(in_features, out_features)

        if self.with_bn:
            if with_bn == 2:
                self.bn = nn.BatchNorm2d(out_features)
            else:
                self.bn = nn.BatchNorm1d(out_features)

        if activation is None:
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def execute(self, x):
        """
        x: (*, C)
        y: (*, C)
        """
        y = self.linear(x)
        if self.with_bn:
            if x.ndim == 2:    # (B, C)
                y = self.activation(self.bn(y))
            elif x.ndim == 3:  # (B, N, C)
                y = self.activation(self.bn(y.transpose(1, 2))).transpose(1, 2)
            elif x.ndim == 4:  # (B, H, W, C)
                y = self.activation(self.bn(y.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        else:
            y = self.activation(y)
        return y


class Conv1D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv1D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv1d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm1d(output_dim)
        if with_relu:
            self.activation = nn.ReLU()

    def execute(self, x):
        """
        x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = self.activation(x)
        return x

class Conv2D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv2D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv2d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm2d(output_dim)
        if with_relu:
            self.activation = nn.ReLU(
            )

    def execute(self, x):
        """
        x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = self.activation(x)
        return x

class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.ReLU()

    def execute(self, x):
        """
        x: (B, C)
        """
        x = self.activation(self.bn(self.fc(x)))
        return x

class GraphConv_H(nn.Module):
    def __init__(self, in_channels, output_scale, neighbor_feature):
        super().__init__()
        self.in_channels = in_channels
        self.output_scale = output_scale
        self.neighbor_feature = neighbor_feature

        if self.neighbor_feature == 1:
            self.conv1 = Conv2D(3, 64, with_bn=True, with_relu=True)
            self.conv2 = Conv2D(64, 64, with_bn=True, with_relu=True)
        if self.neighbor_feature == 1:
            self.graph_conv = Conv2D(3 * 2 + 64, 256, with_bn=True, with_relu=True)
        if self.neighbor_feature == 2:
            self.graph_conv = Conv2D(3 * 2 + 256, 256, with_bn=True, with_relu=True)

    def get_edge_feature(self, x, pos, knn_idx):
        """
        :param        x: (B, C, N)
        :param  knn_idx: (B, N, K)
        :return edge_feat: (B, C, N, K)
        """
        
        knn_pos = knn_group_1(pos, knn_idx)   # (B, C, N, K)
        pos_tiled = pos[:, :, :self.output_scale].unsqueeze(-1).expand_as(knn_pos)
        
        knn_pos = knn_pos - pos_tiled
        knn_dist = jt.sum(knn_pos ** 2, dim=1, keepdims=True)
        knn_r = jt.sqrt(knn_dist.max(dim=3, keepdims=True))
        knn_pos = knn_pos / knn_r.expand_as(knn_pos)
        
        if self.neighbor_feature == 1:
            knn_x = self.conv1(knn_pos)
            knn_x = self.conv2(knn_x ) + knn_x
        if self.neighbor_feature == 2:
            knn_x = knn_group_1(x, knn_idx)
            x_tiled = x[:, :, :self.output_scale].unsqueeze(-1).expand_as(knn_x)
        
            knn_x = knn_x - x_tiled
            knn_xdist = jt.sum(knn_x ** 2, dim=1, keepdims=True)
            knn_xr = jt.sqrt(knn_xdist.max(dim=3, keepdims=True))
            knn_x = knn_x / knn_xr.expand_as(knn_x)
        
        edge_pos = jt.concat([pos_tiled, knn_pos, knn_x], dim=1)
        return edge_pos

    def execute(self, x, pos, knn_idx):
        """
        :param  x: (B, N, x)
              pos: (B, N, y)
        :return y: (B, N, z)
          knn_idx: (B, N, K)
        """
        
        edge_pos = self.get_edge_feature(x, pos, knn_idx=knn_idx)

        y = self.graph_conv(edge_pos)
    
        y_global = y.max(dim=3, keepdims=False)

        return y_global

class HierarchicalLayer(nn.Module):
    def __init__(self, output_scale, input_dim, output_dim, last_dim=0, with_last=False, with_fc=True, neighbor_feature=False):
        super(HierarchicalLayer, self).__init__()
        self.output_scale = output_scale
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.with_last = with_last
        self.with_fc = with_fc
        self.neighbor_feature = neighbor_feature

        self.conv_in = Conv1D(input_dim, input_dim*2, with_bn=True, with_relu=with_fc)

        if with_fc:
            self.fc = FC(input_dim*2, input_dim//2)
            if with_last:
                self.conv_out = Conv1D(input_dim + input_dim//2 + last_dim, output_dim, with_bn=True)
            else:
                self.conv_out = Conv1D(input_dim + input_dim//2, output_dim, with_bn=True)
        else:
            if with_last:
                self.conv_out = Conv1D(input_dim + input_dim*2 + last_dim, output_dim, with_bn=True)
            else:
                self.conv_out = Conv1D(input_dim + input_dim*2, output_dim, with_bn=True)
        
        if self.neighbor_feature:
            self.GraphConv = GraphConv_H(256, self.output_scale, self.neighbor_feature)

    def execute(self, x, x_last=None, knn_idx=None, pos=None):
        """
        x: (B, C, N)
        x_last: (B, C)
        """
        BS, _, _ = x.shape

        ### Global information
        ori_x = x
        y = self.conv_in(x)
        x_global = jt.max(y, dim=2, keepdims=False)
        if self.with_fc:
            x_global = self.fc(x_global)

        ### Neighbor information
        if self.neighbor_feature:
            x = self.GraphConv(x, pos, knn_idx)
            x = ori_x[:, :, :self.output_scale] + x
        else:
            x = ori_x[:, :, :self.output_scale]
    
        ### Feature fusion for shifting
        if self.with_last:
            x = jt.concat([x_global.view(BS, -1, 1).repeat(1, 1, self.output_scale),
                        x_last.view(BS, -1, 1).repeat(1, 1, self.output_scale), x], dim=1)
        else:
            x = jt.concat([x_global.view(BS, -1, 1).repeat(1, 1, self.output_scale), x], dim=1)

        x = self.conv_out(x) 
        x = x + ori_x[:, :, :self.output_scale]
        
        return x, x_global

def batch_quat_to_rotmat(q, out=None):

    batchsize = q.size(0)

    if out is None:
        out = q.new_empty([batchsize, 3, 3])

    # 2 / squared quaternion 2-norm
    s = 2/jt.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = nn.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - jt.multiply((h[:, 2, 2] + h[:, 3, 3]), s)
    out[:, 0, 1] = jt.multiply((h[:, 1, 2] - h[:, 3, 0]), s)
    out[:, 0, 2] = jt.multiply((h[:, 1, 3] + h[:, 2, 0]), s)

    out[:, 1, 0] = jt.multiply((h[:, 1, 2] + h[:, 3, 0]), s)
    out[:, 1, 1] = 1 - jt.multiply((h[:, 1, 1] + h[:, 3, 3]), s)
    out[:, 1, 2] = jt.multiply((h[:, 2, 3] - h[:, 1, 0]), s)

    out[:, 2, 0] = jt.multiply((h[:, 1, 3] - h[:, 2, 0]), s)
    out[:, 2, 1] = jt.multiply((h[:, 2, 3] + h[:, 1, 0]), s)
    out[:, 2, 2] = 1 - jt.multiply((h[:, 1, 1] + h[:, 2, 2]), s)

    return out


