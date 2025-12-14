import gc

import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
from networks.quant_linear import QuanLinear, IdentityQuan, LsqQuan
from networks.quant_linear import Q_MSA
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
import numpy as np
import random
from torch.autograd import Variable
from resnet import *
import pickle
from torchvision import transforms, models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):
        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, n_blocks, channels, output_dim):
        super().__init__()

        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block=BasicBlock, n_blocks=[2, 2, 2, 2], channels=[64, 128, 256, 512], stride=1):
        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return x, h


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class multi_scale_channel_attention(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[3, 5, 7]):
    # def __init__(self, in_channels, kernel_sizes=[1, 3, 5]):
    # def __init__(self, in_channels, kernel_sizes=[5, 7, 11]):
    # def __init__(self, in_channels, kernel_sizes=[5]):
    # def __init__(self, in_channels, kernel_sizes=[7]):
    # def __init__(self, in_channels, kernel_sizes=[3]):
    # def __init__(self, in_channels, kernel_sizes=[1]):
    # def __init__(self, in_channels, kernel_sizes=[11]):
        super(multi_scale_channel_attention, self).__init__()
        self.in_channels = in_channels

        # 确保卷积核大小有效
        self.kernel_sizes = [k for k in kernel_sizes if k % 2 == 1]  # 只保留奇数核大小
        assert self.kernel_sizes, "至少需要一个有效的卷积核大小"

        # 初始化多尺度卷积
        self.convs = nn.ModuleList()
        for k in self.kernel_sizes:
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, in_channels // 16, k, padding=k // 2, bias=False),
                nn.GELU()
            )
            self.convs.append(conv_block)

        # 初始化融合层
        self.fusion = nn.Conv1d(
            len(self.kernel_sizes) * (in_channels // 16),
            in_channels,
            1
        )
        self.sigmoid = nn.Sigmoid()

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
        feats_concat = torch.cat(feats, dim=1)

        # 特征融合并生成通道注意力权重
        weights = self.fusion(feats_concat)
        attn_weights = self.sigmoid(weights).squeeze(2)

        # 应用通道注意力
        channel_out = attn_weights * x

        # 残差连接
        return channel_out + shortcut


class SE1D(nn.Module):
    def __init__(self, in_chnls, ratio=16):
        super(SE1D, self).__init__()
        # 压缩：聚合序列维度（若输入是二维，需先扩展长度维度为1）
        self.squeeze = nn.AdaptiveAvgPool1d(1)  # 对序列长度维度池化
        # 激励：1D卷积模拟全连接（比Linear更适配序列输入）
        self.compress = nn.Conv1d(in_chnls, in_chnls // ratio, kernel_size=1)
        self.excitation = nn.Conv1d(in_chnls // ratio, in_chnls, kernel_size=1)

    def forward(self, x):
        # 若输入是二维 (batch, channels)，扩展为 (batch, channels, 1)（增加长度维度1）
        if x.dim() == 2:
            x = x.unsqueeze(2)  # 变为 (32, 512, 1)

        # 压缩：(batch, channels, length) → (batch, channels, 1)
        out = self.squeeze(x)
        # 激励：学习通道权重
        out = self.compress(out)  # (batch, channels//ratio, 1)
        out = F.relu(out)
        out = self.excitation(out)  # (batch, channels, 1)
        attn = F.sigmoid(out)  # 通道注意力权重

        # 若原始输入是二维，乘权重后去掉长度维度
        if x.size(2) == 1:
            return (x * attn).squeeze(2)  # 回到 (32, 512)
        else:
            return x * attn  # 保留序列维度


class Model(nn.Module):
    def __init__(self, pretrained=True, num_classes=7, drop_rate=0, p=0.5, alpha=0.1, device=None):
        super(Model, self).__init__()
        self.device = device
        '''加载CLIP模型'''
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)  # 直接加载到指定设备
        self.classes = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]  # 示例：7个类别
        '''CLIP'''
        # self.clip_templates = [
        #     "a photo of a {}", "a picture of a {}", "an image of a {}",
        #     "a photo of one {}", "a picture of one {}", "an image of one {}",
        #     "a photo of the {}", "a picture of the {}", "an image of the {}",
        #     "a photo of some {}", "a picture of some {}", "an image of some {}",
        #     "a photo of the {}s", "a picture of the {}s", "an image of the {}s",
        #     "a photo of these {}", "a picture of these {}", "an image of these {}",
        #     "some photos of a {}", "some pictures of a {}", "some images of a {}",
        #     "photos of a {}", "pictures of a {}", "images of a {}",
        #     "a photo containing a {}", "a picture containing a {}", "an image containing a {}",
        #     "a photo showing a {}", "a picture showing a {}", "an image showing a {}",
        #     "a photo of  a {}", "a picture of  a {}", "an image of  a {}",
        #     "a {} in a photo", "a {} in a picture", "a {} in an image",
        #     "a {} in the photo", "a {} in the picture", "a {} in the image",
        #     "the {} in a photo", "the {} in a picture", "the {} in an image",
        #     "a photo with a {}", "a picture with a {}", "an image with a {}",
        #     "a photo has a {}", "a picture has a {}", "an image has a {}",
        #     "a photo with the {}", "a picture with the {}", "an image with the {}",
        #     "a {} photo", "a {} picture", "a {} image",
        #     "{} photo", "{} picture", "{} image",
        #     "a {} in a photograph", "a {} in a pic", "a {} in a snapshot",
        #     "a photograph of a {}", "a pic of a {}", "a snapshot of a {}",
        #     "a pic with a {}", "a snapshot with a {}", "a photograph with a {}",
        #     "showing a {}", "depicting a {}",
        #     "a photo depicting a {}", "a picture depicting a {}", "an image depicting a {}",
        #     "a photo showing a {}", "a picture showing a {}", "an image showing a {}",
        #     "a photo of a {} in the wild", "a picture of a {} in the wild", "an image of a {} in the wild",
        #     "a photo of a {} in nature", "a picture of a {} in nature", "an image of a {} in nature",
        # ]
        # self.prompts = [[template.format(cls) for template in self.clip_templates]
        #                 for cls in self.classes]  # 7组80个模板
        '''KE2NT'''
        self.clip_templates = ["This person is {}", "A photo of a {} face", "This is a {} facial expression",
                               "A person makes a {} facial expression",
                               "An ID photo of a {} facial expression", "A person is feeling {}",
                               "A person feeling {} on the face",
                               "A photo of a person with a {} expression on the face",
                               "A photo of a person making a {} facial expression"]
        # self.clip_templates = ["This person is {}", "A photo of a {} face", "This is a {} facial expression",
        #                        "A person makes a {} facial expression",
        #                        "An ID photo of a {} facial expression", "A person is feeling {}",
        #                        "A person feeling {} on the face",
        #                        "A photo of a person with a {} expression on the face",
        #                        "A photo of a person making a {} facial expression",
        #                        "a photo of a {}", "a picture of a {}", "an image of a {}",
        #                        "a photo of one {}", "a picture of one {}", "an image of one {}",
        #                        "a photo of the {}", "a picture of the {}", "an image of the {}",
        #                        "a photo of some {}", "a picture of some {}", "an image of some {}",
        #                        "a photo of the {}s", "a picture of the {}s", "an image of the {}s",
        #                        "a photo of these {}", "a picture of these {}", "an image of these {}",
        #                        "some photos of a {}", "some pictures of a {}", "some images of a {}",
        #                        "photos of a {}", "pictures of a {}", "images of a {}",
        #                        "a photo containing a {}", "a picture containing a {}", "an image containing a {}",
        #                        "a photo showing a {}", "a picture showing a {}", "an image showing a {}",
        #                        "a photo of  a {}", "a picture of  a {}", "an image of  a {}",
        #                        "a {} in a photo", "a {} in a picture", "a {} in an image",
        #                        "a {} in the photo", "a {} in the picture", "a {} in the image",
        #                        "the {} in a photo", "the {} in a picture", "the {} in an image",
        #                        "a photo with a {}", "a picture with a {}", "an image with a {}",
        #                        "a photo has a {}", "a picture has a {}", "an image has a {}",
        #                        "a photo with the {}", "a picture with the {}", "an image with the {}",
        #                        "a {} photo", "a {} picture", "a {} image",
        #                        "{} photo", "{} picture", "{} image",
        #                        "a {} in a photograph", "a {} in a pic", "a {} in a snapshot",
        #                        "a photograph of a {}", "a pic of a {}", "a snapshot of a {}",
        #                        "a pic with a {}", "a snapshot with a {}", "a photograph with a {}",
        #                        "showing a {}", "depicting a {}",
        #                        "a photo depicting a {}", "a picture depicting a {}", "an image depicting a {}",
        #                        "a photo showing a {}", "a picture showing a {}", "an image showing a {}",
        #                        "a photo of a {} in the wild", "a picture of a {} in the wild",
        #                        "an image of a {} in the wild",
        #                        "a photo of a {} in nature", "a picture of a {} in nature", "an image of a {} in nature"
        #                        ]
        self.prompts = [[template.format(cls) for template in self.clip_templates]
                        for cls in self.classes]  # 7组80个模板
        # self.class_descriptor = {
        #     "surprise": 'widened eyes, an open mouth, raised eyebrows, and a frozen expression.',
        #     "fear": 'raised eyebrows, parted lips, a furrowed brow, and a retracted chin.',
        #     "disgust": 'a wrinkled nose, lowered eyebrows, a tightened mouth, and narrow eyes.',
        #     "happiness": 'a smiling mouth, raised cheeks, wrinkled eyes, and arched eyebrows.',
        #     "sadness": 'tears, a downward turned mouth, drooping upper eyelids, and a wrinkled forehead.',
        #     "anger": 'furrowed eyebrows, narrow eyes, tightened lips, and flared nostrils.',
        #     "neutral": 'relaxed facial muscles, a straight mouth, a smooth forehead, and unremarkable eyebrows.'
        # }
        # self.prompts = [[template.format(cls)+', with {}'.format(self.class_descriptor[cls]) for template in self.clip_templates]
        #                 for cls in self.classes]  # 7组80个模板

        with torch.no_grad():
            self.clip_model.eval()  # 评估模式
            self.tokenized = clip.tokenize([p for b in self.prompts for p in b]).to(self.device)  # (80*7,77)
            self.tmp_promt_feature = self.clip_model.encode_text(self.tokenized).view(len(self.classes),
                                                                                      len(self.clip_templates),
                                                                                      -1).float().to(self.device)
            # 转移到目标设备    # (560,512)-->(7,80,512)
            # self.anchor_tokenized = clip.tokenize([p for b in self.anchor_prompts for p in b]).to(
            #     self.device)  # (9*7,77)
            # self.anchor_feature = self.clip_model.encode_text(self.anchor_tokenized).view(len(self.classes),
            #                                                                               len(self.anchor_templates),
            #                                                                               -1).float().to(
            #     self.device)  # (7, 9, 512)
        '''领域不变性表情的文本描述'''
        '''ChatGPT'''
        # self.class_descriptor = [
        #     'widened eyes, an open mouth, raised eyebrows, and a frozen expression.',
        #     'raised eyebrows, parted lips, a furrowed brow, and a retracted chin.',
        #     'a wrinkled nose, lowered eyebrows, a tightened mouth, and narrow eyes.',
        #     'a smiling mouth, raised cheeks, wrinkled eyes, and arched eyebrows.',
        #     'tears, a downward turned mouth, drooping upper eyelids, and a wrinkled forehead.',
        #     'furrowed eyebrows, narrow eyes, tightened lips, and flared nostrils.',
        #     'relaxed facial muscles, a straight mouth, a smooth forehead, and unremarkable eyebrows.']
        #
        # # '''What are useful facial muscular movements for the facial expression of neutral? please summarizing these information as a sentence'''
        # # self.class_descriptor=[
        # #     "surprise is the frontalis muscle raising the eyebrows and the levator palpebrae superioris widening the eyes, often accompanied by the masseter and pterygoid muscles dropping the jaw open.",
        # #     "fear is the frontalis muscle raising the eyebrows, the levator palpebrae superioris widening the eyes, and the platysma and associated muscles stretching the lips horizontally.",
        # #     "disgust is the levator labii superioris and alaque nasi muscles raising the upper lip and wrinkling the nose, as if to block an offensive odor.",
        # #     "happiness are the combined action of the zygomaticus major pulling the lips up and back and the orbicularis oculi tightening to raise the cheeks and create crow's feet around the eyes.",
        # #     "sadness is the corrugator supercilii and depressor glabellae pulling the inner eyebrows upward and together, combined with the depressor anguli oris pulling the corners of the mouth downward.",
        # #     "anger is the corrugator supercilii and procerus muscles lowering and furrowing the eyebrows, combined with the orbicularis oris and mentalis tightening the lips and pushing up the chin.",
        # #     "neutral facial expression is the minimal engagement of facial muscles, resulting in relaxed frontalis (smooth forehead), orbicularis oculi (soft eyes), and zygomaticus and depressor muscles (a relaxed, non-upturned or downturned mouth)."]
        '''Deepseek'''
        # self.class_descriptor =[
        #     "frontalis raising eyebrows and levator palpebrae widening eyes",
        #     "frontalis raising eyebrows and platysma stretching lips horizontally",
        #     "levator labii raising upper lip and wrinkling nose",
        #     "zygomaticus major lifting mouth corners and orbicularis oculi narrowing eyes",
        #     "depressor anguli oris pulling mouth corners down",
        #     "corrugator supercilii lowering brows and mentalis tightening lips",
        #     "minimal muscle engagement with relaxed eyelids and neutral lips"
        # ]
        # with torch.no_grad():
        #     self.clip_model.eval()  # 评估模式
        #     self.class_tokenized = clip.tokenize([p for p in self.class_descriptor]).to(self.device)  # (7,77)
        #     self.class_tmp_promt_feature = self.clip_model.encode_text(self.class_tokenized).view(len(self.classes),
        #                                                                               -1).float().to(self.device)
        '''加载ResNet50网络'''
        # resnet50 = EACResNet(Bottleneck, [3, 4, 6, 3])
        # with open(r'./pretrained/resnet50_ft_weight.pkl', 'rb') as f:
        #     obj = f.read()
        # weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        # resnet50.load_state_dict(weights)
        # self.features = nn.Sequential(*list(resnet50.children())[:-2])
        # self.features2 = nn.Sequential(*list(resnet50.children())[-2:-1])
        #
        # # self.sour_proj = nn.Linear(2048, 512)
        # # self.sour_proj = QuanLinear(m=nn.Linear(2048, 512),
        # #                           quan_w_fn=LsqQuan(bit=7, all_positive=False, symmetric=False, per_channel=True),
        # #                           quan_a_fn=IdentityQuan())
        # self.sour_proj = nn.AdaptiveAvgPool1d(512)
        # fc_in_dim = 512
        '''加载MobileNet网络'''
        # mobile = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # self.features = nn.Sequential(*list(mobile.features))  # Matches TF backbone
        # self.features2 = nn.AdaptiveAvgPool2d(1)
        # self.sour_proj = nn.AdaptiveAvgPool1d(512)
        # fc_in_dim = 512
        '''加载ResNet18网络'''
        res18 = ResNet(block=BasicBlock, n_blocks=[2, 2, 2, 2], channels=[64, 128, 256, 512], output_dim=1000)
        # msceleb_model = torch.load(r'./pretrained/resnet18_msceleb.pth', map_location=device)  # 加载到指定设备
        msceleb_model = torch.load(r'./pretrained/resnet18_msceleb.pth', map_location=self.device,
                                   weights_only=True)  # 加载到指定设备
        state_dict = msceleb_model['state_dict']
        res18.load_state_dict(state_dict, strict=False)

        self.drop_rate = drop_rate
        self.features = nn.Sequential(*list(res18.children())[:-2])
        self.features2 = nn.Sequential(*list(res18.children())[-2:-1])

        fc_in_dim = list(res18.children())[-1].in_features
        '''分类器'''
        self.fc = nn.Linear(fc_in_dim, num_classes)

        self.parm = {}
        for name, parameters in self.fc.named_parameters():
            self.parm[name] = parameters

        self.loss_cos = torch.nn.CosineEmbeddingLoss()
        self.loss_L1 = torch.nn.L1Loss()
        self.lamb = 0.4
        self.attn = multi_scale_channel_attention(in_channels=512)
        # self.attn = Q_MSA(in_channels=512)
        # self.attn = SE1D(in_chnls=512)
        self.Qlinear = QuanLinear(m=nn.Linear(768, 512),
                                  quan_w_fn=LsqQuan(bit=7, all_positive=False, symmetric=False, per_channel=True),
                                  quan_a_fn=IdentityQuan())
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')  # 替换elementwise_mean
        self.Vlinear = nn.Linear(768, 512)
        # self.linear = QuanLinear(m=nn.Linear(768, 512),
        #                           quan_w_fn=LsqQuan(bit=7, all_positive=False, symmetric=False, per_channel=True),
        #                           quan_a_fn=IdentityQuan())
        # self.vallina_linear = nn.Linear(768, 512)

    def supervisor(self, x, targets, cnum):
        branch = x
        branch = branch.reshape(branch.size(0), branch.size(1), 1, 1)
        branch = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum)).to(self.device)(branch)
        branch = branch.reshape(branch.size(0), branch.size(1), branch.size(2) * branch.size(3))
        loss_2 = 1.0 - 1.0 * torch.mean(torch.sum(branch, 2)) / cnum  # set margin = 3.0

        mask = Mask(x.size(0))
        branch_1 = x.reshape(x.size(0), x.size(1), 1, 1) * mask.to(self.device)
        branch_1 = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum)).to(self.device)(branch_1)
        branch_1 = branch_1.view(branch_1.size(0), -1)
        loss_1 = nn.CrossEntropyLoss()(branch_1, targets)
        return [loss_1, loss_2]
        # return loss_1

    def forward(self, x, targets, phase='train', ratio=0.1):
        # 将输入放到指定设备
        x = x.to(self.device)
        # img = x
        # 获取CLIP特征
        with torch.no_grad():
            # image_features = clip_model.encode_image(x).to(device).float()  # (32,512)
            feature_list, image_features = self.clip_model.encode_image(x)
            image_features = image_features.to(self.device).float()  # 转换为float32
            '''以下引入Text_encoder'''
            if phase == 'train':
                '''80个template表示多个领域，但可能存在一个差的模板'''
                text_feature = [self.tmp_promt_feature[i] for i in
                                (targets.tolist() if isinstance(targets, torch.Tensor) else targets)]  # (32,80,512)
                text_feature = torch.stack(text_feature)  # (32, 80, 512)

                # anchor_txt_feature = [self.anchor_feature[i] for i in
                #                 (targets.tolist() if isinstance(targets, torch.Tensor) else targets)]  # (32,80,512)
                # anchor_txt_feature = torch.stack(anchor_txt_feature)

                '''领域不变性特征'''
                # inv_text_feature = [self.class_tmp_promt_feature[i] for i in
                #                 (targets.tolist() if isinstance(targets, torch.Tensor) else targets)]  # (32,512)
                # inv_text_feature = torch.stack(inv_text_feature)
                # # inv_text_feature = inv_text_feature.unsqueeze(dim=1).repeat(1, text_feature.shape[1], 1)
                # # text_feature = 0.7 *text_feature + 0.3*inv_text_feature
                # text_feature = inv_text_feature.unsqueeze(dim=1)
            else:

                pass
        '''MSPA1.1'''
        processed_features = []
        Bs, n_token, dim = feature_list[0].shape
        for i, f in enumerate(feature_list):
            '''AllPthToken'''
            x_img = f[:, 1:, :].to(self.device).float()  # (B,49,768)
            # 先转置通道维度，再重塑为空间形状
            x_reshaped = x_img.permute(0, 2, 1).view(Bs, dim, 7, 7)  # (B,768,7,7)
            # 通过适配器处理并展平
            processed_features.append(torch.mean(x_reshaped, [2, 3]))
            '''AllclsToken'''
            # x_img = f[:, 0, :].to(self.device).float()  # (B,49,768)
            # x_reshaped = x_img.view(Bs, dim)  # (B,768,7,7)
            # processed_features.append(x_reshaped)
        feature_cat = torch.stack(processed_features).mean(0)
        # feature_cat = self.attn(feature_cat)
        # with torch.no_grad():
        #     feature_cat = feature_cat.type(self.clip_model.visual.conv1.weight.dtype) @ self.clip_model.visual.proj
        '''量化意识训练'''
        feature_cat = self.Qlinear(feature_cat)
        # feature_cat = self.Vlinear(feature_cat)
        image_features = feature_cat + image_features

        '''ResNet'''
        x = self.features(x)  # (32,512,7,7)
        feat = x
        x = self.features2(x)  # (32,512,1,1)
        x = x.view(x.size(0), -1)
        # x = self.sour_proj(x)   # only for ResNet50, MobileNet
        '''CLIP Adapter'''
        image_features = self.attn(image_features)
        x = image_features * torch.sigmoid(x)
        # 以ResNet输出的特征为加权值去修饰CLIP输出的特征，使CLIP的输出特征保留与表情相关的语义

        out = self.fc(x)
        # if phase == 'train':
        #     xf = x
        #
        #     def loss_fn(xf_single, text_single):
        #         """仅计算损失，不处理梯度"""
        #         dot_product = torch.dot(xf_single, text_single)
        #         xf_norm = torch.norm(xf_single)
        #         text_norm = torch.norm(text_single)
        #         cos_sim = dot_product / (xf_norm * text_norm + 1e-8)
        #         return cos_sim  # 返回损失值，不计算梯度
        #
        #     grad_fn = torch.func.grad(loss_fn)  # grad_fn(xf, text)会返回loss对xf的梯度
        #     per_sample_vmap = torch.vmap(grad_fn, in_dims=(None, 0))    # (512,) (9, 512)
        #     batch_vmap = torch.vmap(per_sample_vmap, in_dims=(0, 0))    # 按照批次的维度计算梯度
        #     xf_grad = batch_vmap(xf, text_feature)  #(32,512)+(32,9,512)-->(512,)+(9,512)-->(512,)+(512,)-->最后合并
        #
        #     '''ResNet18_reg2_Text_final_R2Others.pth'''
        #     # 先聚合所有模板的影响力, 再选择通道
        #     channel_impact = xf_grad.abs() * xf.abs().unsqueeze(1)  # (32,80,512) |grad|*|input|
        #     # # channel_impact = xf_grad.abs() * xf.unsqueeze(1)  # (32,80,512)  cvpr2025
        #     # # channel_impact = xf_grad.abs()  # (32,80,512)  |grad|  tpami2024
        #     channel_impact_aggregated = channel_impact.mean(dim=1)  # (32, 512)
        #     mu = channel_impact_aggregated.mean(dim=1, keepdim=True)  # (32, 1)
        #     std = channel_impact_aggregated.std(dim=1, keepdim=True)  # (32, 1)
        #     mask_aggregated = channel_impact_aggregated > (mu + 1.0 * std)  # (32, 512)
        #     # mask_aggregated = channel_impact_aggregated > (mu + 0.5 * std)  # (32, 512)
        #     # mask_aggregated = channel_impact_aggregated > (mu + 2.0 * std)  # (32, 512)
        #     # mask_aggregated = channel_impact_aggregated > mu  # (32, 512)
        #     # 阈值法筛选敏感性
        #     # maximum, _ = channel_impact_aggregated.median(dim=1)
        #     # mask_aggregated = channel_impact_aggregated > maximum.unsqueeze(dim=1).repeat(1, 512)
        #     # # maximum, _ = channel_impact_aggregated.max(dim=1)
        #     # # maximum = maximum / 2
        #     # # mask_aggregated = channel_impact_aggregated > maximum.unsqueeze(dim=1).repeat(1, 512)
        #     reg_input = xf * mask_aggregated.float()
        #     L1_reg = 0.01 * torch.abs(reg_input).sum()
        #     # L1_reg = 0.001 * torch.abs(reg_input).sum()  # For AffectNet BS=256; FERplus BS=256
        #
        #     '''余弦相似度'''
        #     # x_cos = x
        #     # x_text_feature = text_feature.mean(dim=1)
        #     # x_text_feature = x_text_feature / x_text_feature.norm(dim=-1, keepdim=True).float()  # (32,512)
        #     # cosine_sim_label = torch.ones(x_cos.shape[0]).to(self.device)
        #     # # I_text_cos = 2 * self.loss_cos(F.normalize(x_cos, dim=-1), x_text_feature, cosine_sim_label)
        #     # I_text_cos = self.loss_cos(F.normalize(x_cos, dim=-1), x_text_feature, cosine_sim_label)
        #     # # inv_text_feature = inv_text_feature / inv_text_feature.norm(dim=-1, keepdim=True).float()
        #     # # inv_sim_label = torch.ones(x_cos.shape[0]).to(self.device)
        #     # # inv_text_cos = 5 * self.loss_cos(F.normalize(x_cos, dim=-1), inv_text_feature, inv_sim_label)
        #     # # I_text_cos = I_text_cos + inv_text_cos
        #     '''OPL'''
        #     x_cos = x
        #     x_text_feature = text_feature.mean(dim=1)
        #     x_text_feature = x_text_feature / x_text_feature.norm(dim=-1, keepdim=True).float()  # (32,512)
        #     x_cos = F.normalize(x_cos, p=2, dim=-1)
        #     labels = targets[:, None]  # extend dim
        #     mask = torch.eq(labels, labels.t()).bool().to(x_cos.device)
        #     # eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(x_cos.device)
        #     # mask_pos = mask.masked_fill(eye, 0).float()
        #     mask_pos = mask.float()
        #     mask_neg = (~mask).float()
        #     dot_prod = torch.matmul(x_cos, x_text_feature.t())  # 6*6 样本之间计算余弦相似性
        #     pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)  # 同类样本相似性之和/同类样本数=单个同类样本间的相似性
        #     neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (
        #             mask_neg.sum() + 1e-6)  # 异类样本相似性之和/异类样本数=单个异类样本间的相似性
        #     I_text_cos = (1.0 - pos_pairs_mean) + (2 * neg_pairs_mean)
        #     # I_text_cos = 0.1*I_text_cos
        #     # I_text_cos = (1.0 - pos_pairs_mean)
        #     # inv_text_feature = inv_text_feature / inv_text_feature.norm(dim=-1, keepdim=True).float()
        #     # inv_sim_label = torch.ones(x_cos.shape[0]).to(self.device)
        #     # inv_text_cos = 5 * self.loss_cos(F.normalize(x_cos, dim=-1), inv_text_feature, inv_sim_label)
        #     # I_text_cos = I_text_cos + inv_text_cos
        #
        #     '''计算图像与文本间相互的KL散度，使图像的特征逼近领域不变性特征'''
        #     # # 1. 特征处理（不变）
        #     # prompt_feature = self.tmp_promt_feature / self.tmp_promt_feature.norm(dim=-1, keepdim=True).float()     # (7,9,512)
        #     # prompt_feature = prompt_feature.view(len(self.classes) * len(self.clip_templates), -1)      # (63,512)
        #     # cos_sim = torch.matmul(x_cos, prompt_feature.T).view(-1, len(self.classes), len(self.clip_templates))   # (32,512)*(512,63)-->(32,7,9)
        #     #
        #     # # 2. 预计算概率和对数概率
        #     # prob_per_template = torch.softmax(cos_sim, dim=1)  # (batch, classes, templates)
        #     # log_prob_per_template = F.log_softmax(cos_sim, dim=1)  # (batch, classes, templates)
        #     #
        #     # # 3. 向量化计算所有模板对的对称KL散度
        #     # batch_size, num_classes, num_templates = prob_per_template.shape
        #     #
        #     # # 扩展维度以匹配j和k
        #     # log_prob_j = log_prob_per_template.unsqueeze(-1)  # (batch, classes, templates, 1)
        #     # prob_j = prob_per_template.unsqueeze(-1)  # (batch, classes, templates, 1)
        #     # log_prob_k = log_prob_per_template.unsqueeze(2)  # (batch, classes, 1, templates)
        #     # prob_k = prob_per_template.unsqueeze(2)  # (batch, classes, 1, templates)
        #     #
        #     # # 计算KL(j||k)和KL(k||j)，并沿类别维度求和
        #     # kl_jk = torch.sum(prob_j * (log_prob_j - log_prob_k), dim=1)  # (batch, templates, templates)
        #     # kl_kj = torch.sum(prob_k * (log_prob_k - log_prob_j), dim=1)  # (batch, templates, templates)
        #     # symmetric_kl = (kl_jk + kl_kj) / 2  # (batch, templates, templates)
        #     #
        #     # # 4. 筛选j < k的有效对并求平均
        #     # mask = torch.triu(torch.ones(num_templates, num_templates, device=symmetric_kl.device), diagonal=1).bool()
        #     # L_kl = symmetric_kl[:, mask].mean()  # 最终标量损失
        #
        # else:
        #     pass
        if phase == 'train':
            # A = self.supervisor(x, targets, cnum=73)
            # # return out, I_text_cos
            # sep_loss = 1.5*A[0] + 5*A[1]
            # return out, [L1_reg, I_text_cos, sep_loss]
            # return out, [L1_reg, I_text_cos, domain_feature_relation_loss]
            return out, [L1_reg, I_text_cos]
            # return out, x
            # return out, x, text_feature.mean(1)

        else:
            # return out, out
            return out, x




class my_MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3,1)


        input = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        input = input.transpose(3,1).contiguous()

        return input

    def __repr__(self):
        kh, kw = _pair(self.kernel_size)
        dh, dw = _pair(self.stride)
        padh, padw = _pair(self.padding)
        dilh, dilw = _pair(self.dilation)
        padding_str = ', padding=(' + str(padh) + ', ' + str(padw) + ')' \
            if padh != 0 or padw != 0 else ''
        dilation_str = (', dilation=(' + str(dilh) + ', ' + str(dilw) + ')'
                        if dilh != 0 and dilw != 0 else '')
        ceil_str = ', ceil_mode=' + str(self.ceil_mode)
        return self.__class__.__name__ + '(' \
            + 'kernel_size=(' + str(kh) + ', ' + str(kw) + ')' \
            + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
            + padding_str + dilation_str + ceil_str + ')'


class my_AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(my_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        input = input.transpose(3,1)
        input = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)
        input = input.transpose(3,1).contiguous()
        return input


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'

def Mask(nb_batch):
    bar = []
    for i in range(7):
        foo = [1] * 63 + [0] *  10
        if i == 6:
            foo = [1] * 64 + [0] *  10
        random.shuffle(foo)  #### generate mask
        bar += foo  # bar = [1, 512]
    bar = [bar for i in range(nb_batch)]    #[N, 512],所有类同一个0/1处理，不利于学习各个类的表情属性
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, 512, 1, 1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar