import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import glob
import cv2

class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels, domains, transform=None):
        self.image_paths = image_paths
        self.labels = labels  # 表情标签
        self.domains = domains  # 域标签
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图片
        img = cv2.imread(self.image_paths[idx])
        # img = Image.open(self.image_paths[idx]).convert('RGB')
        # 应用预处理
        if self.transform:
            img = self.transform(img)
        # 返回图片、表情标签、域标签
        return img, self.labels[idx], self.domains[idx]


def create_dataset(domain_folders):
    """
    合并多个数据集并生成标签
    参数:
        domain_folders: 列表，每个元素为(数据集路径, 域标签)
    """
    image_paths = []
    labels = []
    domains = []

    # 遍历每个数据集
    for folder_path, domain_id in domain_folders:
        # 遍历每个表情类别文件夹
        # img_path = glob.glob(folder_path + '/*/*/*')
        img_path = glob.glob(folder_path + '/*/*')
        exp_label = [int(i.split('/')[-2]) for i in img_path]
        # exp_label = [int(i.split('\\')[-2]) for i in img_path]
        dom_label = [domain_id for i in img_path]

        image_paths.extend(img_path)
        labels.extend(exp_label)
        domains.extend(dom_label)

    return image_paths, labels, domains

# 6. 为训练集和测试集分别应用不同的transform
class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, domain = self.dataset[idx]
        return self.transform(img), label, domain


def calculate_domain_losses(dom_labels, exp_labels, exp_op, criterion):
    mask_0 = (dom_labels == 0)
    mask_1 = (dom_labels == 1)
    mask_2 = (dom_labels == 2)

    # 使用masked_select提取子张量（比直接索引更安全）
    exp_op_0 = torch.masked_select(exp_op, mask_0.unsqueeze(1)).view(-1, exp_op.size(1))
    exp_labels_0 = torch.masked_select(exp_labels, mask_0)

    exp_op_1 = torch.masked_select(exp_op, mask_1.unsqueeze(1)).view(-1, exp_op.size(1))
    exp_labels_1 = torch.masked_select(exp_labels, mask_1)

    exp_op_2 = torch.masked_select(exp_op, mask_2.unsqueeze(1)).view(-1, exp_op.size(1))
    exp_labels_2 = torch.masked_select(exp_labels, mask_2)

    # 强制保留梯度（仅在确认子张量梯度丢失时使用）
    if exp_op_0.requires_grad is False:
        exp_op_0.requires_grad_(True)
    if exp_op_1.requires_grad is False:
        exp_op_1.requires_grad_(True)
    if exp_op_2.requires_grad is False:
        exp_op_2.requires_grad_(True)

    # 计算损失（空样本处理同上）
    loss_0 = criterion(exp_op_0, exp_labels_0) if mask_0.any() else torch.tensor(0.0, device=exp_op.device,
                                                                                 requires_grad=True)
    loss_1 = criterion(exp_op_1, exp_labels_1) if mask_1.any() else torch.tensor(0.0, device=exp_op.device,
                                                                                 requires_grad=True)
    loss_2 = criterion(exp_op_2, exp_labels_2) if mask_2.any() else torch.tensor(0.0, device=exp_op.device,
                                                                                 requires_grad=True)

    return loss_0, loss_1, loss_2


