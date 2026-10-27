import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class TemporalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MultiScale_TemporalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=(1, 2, 3, 4), residual=True, residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0), nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True), TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation)) for ks, dilation in zip(kernel_size, dilations)])
        self.branches.append(nn.Sequential(nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0), nn.BatchNorm2d(branch_channels), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)), nn.BatchNorm2d(branch_channels)))
        self.branches.append(nn.Sequential(nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)), nn.BatchNorm2d(branch_channels)))
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class TDGC(nn.Module):

    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(TDGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1, beta=1, gamma=0.1):
        x1, x3 = (self.conv1(x).mean(-2), self.conv3(x))
        x1 = self.tanh(x1.unsqueeze(-1) - x1.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        x4 = self.tanh(x3.mean(-3).unsqueeze(-1) - x3.mean(-3).unsqueeze(-2))
        x3 = x3.permute(0, 2, 1, 3)
        x5 = torch.einsum('btmn,btcn->bctm', x4, x3)
        x1 = x1 * beta + x5 * gamma
        return x1

class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class unit_gcn(nn.Module):

    def __init__(self, in_channels, out_channels, A, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(TDGC(in_channels, out_channels))
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.register_buffer("A", torch.from_numpy(A.astype(np.float32)))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-06)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha, self.beta, self.gamma)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y

class TCN_GCN_unit(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=(1, 2)):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

class Model(nn.Module):
    def __init__(self, adjacency, n_classes, n_joints, num_person, seq_len,
                 adaptive=True, drop_out=0.0):
        super().__init__()
        self.n_classes = int(n_classes)
        self.n_joints = int(n_joints)
        self.num_person = int(num_person)
        self.seq_len = int(seq_len)
        self.feature_dim = 256
        if min(self.n_classes, self.n_joints, self.num_person, self.seq_len) <= 0:
            raise ValueError("Class, joint, person and frame counts must be positive.")
        adjacency = np.asarray(adjacency, dtype=np.float32)
        if adjacency.shape != (3, self.n_joints, self.n_joints):
            raise ValueError("Graph adjacency does not match the configured joint count.")
        self.data_bn = nn.BatchNorm1d(self.num_person * 3 * self.n_joints)
        channels = (3, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256)
        self.layers = nn.ModuleList([
            TCN_GCN_unit(channels[i], channels[i + 1], adjacency,
                         stride=2 if i in (4, 7) else 1,
                         residual=i != 0, adaptive=adaptive)
            for i in range(10)
        ])
        self.drop_out = nn.Dropout(float(drop_out)) if drop_out else nn.Identity()
        bn_init(self.data_bn, 1)

    def get_input_shape(self):
        return (3, self.seq_len, self.n_joints, self.num_person)

    def forward_feature(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected N,C,T,V,M skeletons, got {tuple(x.shape)}.")
        n, c, t, v, m = x.shape
        if (c, v, m) != (3, self.n_joints, self.num_person) or t <= 0:
            raise ValueError(f"Input shape {tuple(x.shape)} does not match {self.get_input_shape()}.")
        x = x.permute(0, 4, 3, 1, 2).contiguous().reshape(n, m * v * c, t)
        x = self.data_bn(x)
        x = x.reshape(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous()
        x = x.reshape(n * m, c, t, v)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(n, m, self.feature_dim, -1).mean(-1).mean(1)
        return self.drop_out(x)

    def forward(self, x):
        return self.forward_feature(x)

    def forward_contrastive(self, x):
        return F.normalize(self.forward_feature(x), dim=1)
