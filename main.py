import argparse
import torch
import torch.nn as nn
from Trainer import traniner
from Networks import Model
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset, DataLoader, random_split, RandomSampler
import numpy as np
import random
from dataset import create_dataset, EmotionDataset, TransformDataset, calculate_domain_losses
from tqdm import tqdm
import os
from collections import Counter

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 单GPU只需初始化一个
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_class_frequencies(dataset, num_classes):
    """
    从数据集中计算类别频率

    Args:
        dataset: 包含标签的数据集
        num_classes: 类别数量
    """
    # 收集所有标签
    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]  # 假设数据集返回 (data, label)
        all_labels.append(label)

    # 统计每个类别的样本数
    label_counter = Counter(all_labels)

    # 计算频率
    class_counts = torch.zeros(num_classes)
    for class_idx, count in label_counter.items():
        class_counts[class_idx] = count

    # 转换为频率（归一化到 [0,1]）
    class_frequencies = class_counts / class_counts.sum()

    return class_frequencies


# 使用示例
# class_freq = compute_class_frequencies(train_dataset, num_classes=10)

def main(args):
    solver = traniner(args)
    setup_seed(3407)
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    valid_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    '''单源域泛化'''
    train_dataset = datasets.ImageFolder(args.source_train_pathes, train_transform)
    valid_dataset = datasets.ImageFolder(args.source_test_pathes, valid_transform)
    '''多源域泛化'''
    # train_dataset1 = datasets.ImageFolder(args.source_train_pathes1, train_transform)
    # train_dataset2 = datasets.ImageFolder(args.source_train_pathes2, train_transform)
    # train_dataset = ConcatDataset([train_dataset1, train_dataset2])
    # valid_dataset1 = datasets.ImageFolder(args.source_test_pathes1, valid_transform)
    # valid_dataset2 = datasets.ImageFolder(args.source_test_pathes2, valid_transform)
    # valid_dataset = ConcatDataset([valid_dataset1, valid_dataset2])
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    '''多源域loader'''
    # sampler = RandomSampler(train_dataset, replacement=False)
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=args.batch_size,
    #                           sampler=sampler,
    #                           num_workers=args.workers)

    test_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    # 初始化模型（单GPU）
    model = Model(device=args.device)
    model.to(args.device)

    # 优化器和调度器
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.0005, amsgrad=True)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0005, amsgrad=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=0.0005, amsgrad=True)    # For AffectNet BS=256; FERplus BS=256
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # 确保保存目录存在
    os.makedirs("./checkpoints", exist_ok=True)

    best_acc = 0
    for i in range(1, args.epochs + 1):
        ratio = 2.0 * (i / (args.epochs + 1))
        train_acc, train_loss = solver.train(model, train_loader, optimizer, scheduler, args.device, ratio)
        test_acc, test_loss = solver.test(model, test_loader, args.device)

        print(f"epoch: {i}, tran_acc: {train_acc:.4f}, tran_loss: {train_loss:.4f}, "
              f"valid_acc: {test_acc:.4f}, valid_loss: {test_loss:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(), }, "checkpoints/ResNet50_best_R2Others.pth")

        torch.save({'model_state_dict': model.state_dict(), }, "checkpoints/ResNet50_final_R2Others.pth")

        with open('results.txt', 'a') as f:
            f.write(str(i) + '_' + str(test_acc) + '\n')


def run_test(args):

    '''原数据读取''' # For the RAF-DB source domain
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    image_paths, labels, domains = create_dataset([(r'./data/sfew/valid', 0)])
    tet_dataset = EmotionDataset(image_paths=image_paths, labels=labels, domains=domains, transform=None)
    print(f'Validation set size: {len(tet_dataset)}')
    tet_dataset = TransformDataset(tet_dataset, data_transforms_val)
    '''现在数据读取'''
    # data_transforms_test = transforms.Compose([
    #     # transforms.ToPILImage(),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # test_path = r'./data/sfew/valid'
    # tet_dataset = datasets.ImageFolder(test_path, data_transforms_test)
    # print(f'Validation set size: {len(tet_dataset)}')
    test_loader = torch.utils.data.DataLoader(
        tet_dataset,
        # batch_size=32,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    model = Model(device=args.device)
    model.to(args.device)
    checkpoint = torch.load("./checkpoints/1115_model_exp/Model_KE2NT_Q_l_full_best_R2Others.pth", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    # img_tensor = []
    # txt_tensor = []
    with torch.no_grad():
        model.eval()
        correct_sum = 0
        data_num = 0
        for imgs1, labels, _ in tqdm(test_loader, desc='Validating'):
        # for imgs1, labels in tqdm(test_loader, desc='Validating'):
            imgs1 = imgs1.to(args.device)
            labels = labels.to(args.device)

            # outputs, img, txt = model(imgs1, labels, phase='train')
            outputs, _ = model(imgs1, labels, phase='test')
            # img_tensor.append(img)
            # txt_tensor.append(txt)
            _, predicts = torch.max(outputs, 1)
            # predicts = torch.cat((ind_t, predicts.unsqueeze(1)), dim=1)
            # predicts, t = torch.mode(predicts, dim=1)
            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num
            data_num += outputs.size(0)

        test_acc = correct_sum.float() / float(data_num)
        print(f'Test acc: {test_acc}')
    # return torch.cat(img_tensor, dim=0), torch.cat(txt_tensor, dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_train_pathes', type=str, default=r'./data/raf/train', help='source dataset path.')
    parser.add_argument('--source_test_pathes', type=str, default=r'./data/raf/test', help='source dataset path.')
    parser.add_argument('--label_path', type=str, default='list_patition_label.txt', help='label_path')
    parser.add_argument('--workers', type=int, default=8, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--w', type=int, default=7, help='width of the attention map')
    parser.add_argument('--h', type=int, default=7, help='height of the attention map')
    parser.add_argument('--lam', type=float, default=5, help='kl_lambda')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--random_seed', type=int, default=3407, help='random seed.')
    parser.add_argument('--device', default="cuda:8", type=str, help='which GPUs for training')
    args = parser.parse_args()

    # main(args)    # train
    # run_test(args)  # test

    '''绘制图1的直方图统计'''
    import scipy.io as scio
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm  # 用于高斯分布拟合

    # 1. 基础配置（解决后端/中文显示）
    plt.switch_backend('Agg')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示


    # 2. 距离计算函数（可选：欧式/余弦/曼哈顿距离）
    def calculate_euclidean_distance(vec1, vec2):
        """计算逐样本欧式距离（最常用）"""
        return np.linalg.norm(vec1 - vec2, axis=1)


    def calculate_cosine_distance(vec1, vec2):
        """计算逐样本余弦距离（1-余弦相似度）"""
        norm1 = np.linalg.norm(vec1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(vec2, axis=1, keepdims=True)
        cos_sim = (vec1 * vec2).sum(axis=1) / (norm1 * norm2).squeeze()
        return 1 - cos_sim  # 距离越小，向量越相似


    # 3. 加载数据并计算两组真实距离
    target_data_tmp1 = scio.loadmat(r'CAFE_R2S.mat')
    img_tensor1, tet_tensor1 = target_data_tmp1['img'], target_data_tmp1['txt']

    # 第一组距离：img与txt的逐样本欧式距离（核心：真实距离，非标准差）
    # distance_group1 = calculate_euclidean_distance(img_tensor1, tet_tensor1)
    distance_group1 = calculate_cosine_distance(img_tensor1, tet_tensor1)

    target_data_tmp2 = scio.loadmat(r'DU_GFER_R2S.mat')
    img_tensor2, tet_tensor2 = target_data_tmp2['img'], target_data_tmp2['txt']

    # 第一组距离：img与txt的逐样本欧式距离（核心：真实距离，非标准差）
    # distance_group2 = calculate_euclidean_distance(img_tensor2, tet_tensor2)
    distance_group2 = calculate_cosine_distance(img_tensor2, tet_tensor2)

    # 4. 统一距离区间（保证两组对比公平）
    all_distances = np.concatenate([distance_group1, distance_group2])
    bins = np.linspace(all_distances.min(), all_distances.max(), 100)  # 15个等距区间
    x_range = np.linspace(all_distances.min(), all_distances.max(), 1000)  # 拟合曲线的x轴范围

    # 5. 高斯分布拟合（计算每组的均值、标准差，生成拟合曲线）
    # 第一组拟合
    mu1, sigma1 = norm.fit(distance_group1)  # 拟合高斯分布的均值和标准差
    gauss_curve1 = norm.pdf(x_range, mu1, sigma1) * len(distance_group1) * (bins[1] - bins[0])  # 缩放至直方图尺度
    # 第二组拟合
    mu2, sigma2 = norm.fit(distance_group2)
    gauss_curve2 = norm.pdf(x_range, mu2, sigma2) * len(distance_group2) * (bins[1] - bins[0])

    # 6. 绘制直方图+高斯拟合曲线
    plt.figure(figsize=(12, 7))

    # 绘制直方图（区间数量统计）
    plt.hist(distance_group1, bins=bins, alpha=0.6, color=(0.4, 0.7, 1.0),
             edgecolor='lightskyblue', label='CAFE')
    plt.hist(distance_group2, bins=bins, alpha=0.6, color='peachpuff',
             edgecolor='peachpuff', label='Ours')

    # 绘制高斯拟合曲线（红色/绿色实线，更醒目）
    plt.plot(x_range, gauss_curve1, color='royalblue', linewidth=2.5, linestyle='--',
             label='CAFE')
    plt.plot(x_range, gauss_curve2, color='orange', linewidth=2.5, linestyle='--',
             label='Ours')

    # 7. 图表标注（贴合“距离-数量+拟合分布”）
    plt.xlabel('1-cos(.) statistic', fontsize=18, fontweight='medium')
    plt.ylabel('Sample Count', fontsize=18, fontweight='medium')
    # plt.title('Distance statistics between SFEW 2.0 images and KE2NT text prompts.', fontsize=18, pad=10)
    plt.legend(loc='upper right', fontsize=18, framealpha=0.9)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # 8. 保存高分辨率图片
    plt.savefig('cosdistance_hist_with_gauss.jpg', dpi=800, bbox_inches='tight', facecolor='white')
    plt.close()


