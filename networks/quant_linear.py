import torch as t

class Quantizer(t.nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, 'The bit-width of identity quantizer must be None'

    def forward(self, x):
        return x

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)   # -64
                self.thd_pos = 2 ** (bit - 1) - 1   # 63

        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)
        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x


class QuanLinear(t.nn.Linear):
    def __init__(self, m: t.nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        # quantized_act = self.quan_a_fn(x)
        quantized_act = x
        return t.nn.functional.linear(quantized_act, quantized_weight, self.bias)


class QuanConv1d(t.nn.Conv1d):
    def __init__(self, m: t.nn.Conv1d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == t.nn.Conv1d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = t.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight)
        if m.bias is not None:
            self.bias = t.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        # quantized_act = self.quan_a_fn(x)
        quantized_act = x
        return self._conv_forward(quantized_act, quantized_weight,self.bias)


class Q_MSA(t.nn.Module):
    def __init__(self, in_channels, kernel_sizes=[3, 5, 7]):
        super(Q_MSA, self).__init__()
        self.in_channels = in_channels
        self.kernel_sizes = [k for k in kernel_sizes if k % 2 == 1]  # 只保留奇数核大小
        assert self.kernel_sizes, "至少需要一个有效的卷积核大小"
        self.convs = t.nn.ModuleList()
        orig_convs = [t.nn.Conv1d(in_channels, in_channels // 16, k, padding=k // 2, bias=False) for k in
                      self.kernel_sizes]
        # for k, conv in (self.kernel_sizes, orig_convs):
        for k, conv in zip(self.kernel_sizes, orig_convs):

            conv_block = t.nn.Sequential(
                QuanConv1d(m=conv,
                           quan_w_fn=LsqQuan(bit=7, all_positive=False, symmetric=False, per_channel=True),
                           quan_a_fn=IdentityQuan()),
                t.nn.GELU()
            )
            self.convs.append(conv_block)

        fus_org_conv = t.nn.Conv1d(len(self.kernel_sizes) * (in_channels // 16), in_channels, 1)
        self.fusion = QuanConv1d(m=fus_org_conv,
                                 quan_w_fn=LsqQuan(bit=7, all_positive=False, symmetric=False, per_channel=True),
                                 quan_a_fn=IdentityQuan())
        self.sigmoid = t.nn.Sigmoid()

    def forward(self, x):
        # 输入检查
        if x is None:
            raise ValueError("输入张量不能为None")

        # 输入x形状: (batch_size, in_channels) = (32, 512)
        shortcut = x

        # 为1D卷积添加序列维度: (32, 512, 1)
        x_unsqueezed = x.unsqueeze(2)

        # 多尺度通道特征提取
        feats = []
        for conv in self.convs:
            if conv is None:
                continue  # 跳过未正确初始化的卷积层
            feat = conv(x_unsqueezed)
            feats.append(feat)

        # 确保有特征可以处理
        if not feats:
            return x  # 如果没有有效特征，直接返回输入

        # 拼接多尺度特征
        feats_concat = t.cat(feats, dim=1)

        # 特征融合并生成通道注意力权重
        weights = self.fusion(feats_concat)
        attn_weights = self.sigmoid(weights).squeeze(2)

        # 应用通道注意力
        channel_out = attn_weights * x

        # 残差连接
        return channel_out + shortcut