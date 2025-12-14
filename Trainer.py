import torch
import torch.nn as nn
import torch.nn.functional as F
# from LibMTL.weighting.abstract_weighting import AbsWeighting
# from LibMTL.LibMTL.weighting import DB_MTL

def MultiClassGHMC(logits, targets):
    batch_size = logits.size(0)

    # 1. 计算多分类交叉熵损失（基础损失）
    ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [batch_size]

    # 2. 计算多分类梯度模长g_i = 1 - p_iy_i（p_iy_i是真实类别的预测概率）
    pred_probs = F.softmax(logits, dim=1)  # 所有类别的预测概率，[batch_size, num_classes]
    # 提取每个样本真实类别的预测概率
    true_class_probs = pred_probs[torch.arange(batch_size), targets]  # [batch_size]
    # 梯度模长：g_i越小，样本越简单
    g = 1.0 - true_class_probs  # [batch_size]

    # 3. 统计每个梯度区间的样本数量（计算梯度密度GD(g_i)）
    # 将g_i映射到对应的区间索引（0~bins-1）
    bin_indices = torch.floor(g / (1.0 / 7)).long()
    # 确保区间索引不超过bins-1（避免g=1.0时越界）
    bin_indices = torch.clamp(bin_indices, 0, 7 - 1)
    # 统计每个区间的样本数（delta_k）
    bin_counts = torch.bincount(bin_indices, minlength=7).float()  # [bins]

    # 4. 计算每个样本的协调权重beta_i = N / (GD(g_i))，GD(g_i) = bin_counts[bin_idx] / bin_width
    # 先获取每个样本所在区间的样本数
    bin_count_per_sample = bin_counts[bin_indices]  # [batch_size]
    # 梯度密度GD(g_i)
    grad_density = bin_count_per_sample / (1.0 / 7)  # [batch_size]
    # 协调权重（N为batch_size，避免分母为0加eps）
    beta = batch_size / (grad_density + 1e-6)  # [batch_size]

    # 5. 计算多分类GHM-C损失：加权平均交叉熵损失
    ghmc_loss = (beta * ce_loss).mean()

    return ghmc_loss

def hardsample(logits, targets):
    probs = F.softmax(logits, dim=1)
    target_probs = probs[torch.arange(len(targets)), targets]
    # weights = (1 - target_probs).detach()
    weights = 1. / target_probs.detach()
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    weighted_loss = weights * ce_loss
    return weighted_loss.mean()


def imb_cls(logits, targets):
    probs = F.softmax(logits, dim=1)
    p_i = probs[torch.arange(len(targets)), targets]
    s = 1 - p_i
    epsilon = 1e-8
    p_i = torch.clamp(p_i, epsilon, 1.0 - epsilon)
    s = torch.clamp(s, epsilon, 1.0 - epsilon)

    log_ratio = torch.log(p_i / s)

    num = 1. / (1.+torch.exp(-2.0 * log_ratio))
    loss = -torch.log(num)

    return loss.mean()


def imb_cls_2(logits, targets, lambda_param=0.5):
    probs = F.softmax(logits, dim=1)
    batch_size = len(targets)

    p_i = probs[torch.arange(batch_size), targets]
    S = 1.0 - p_i  # 因为 sum_{j≠i} p_j = 1 - p_i

    epsilon = 1e-8
    p_i = torch.clamp(p_i, epsilon, 1.0 - epsilon)
    S = torch.clamp(S, epsilon, 1.0 - epsilon)

    x = lambda_param * torch.log(p_i) - (1 - lambda_param) * torch.log(S)
    sigma_x = torch.sigmoid(x)
    loss = -torch.log(sigma_x)

    return loss.mean()


def label_smoothing_loss(logits, targets, epsilon=0.1):
    num_classes = logits.size(-1)

    # 标准交叉熵
    ce_loss = F.cross_entropy(logits, targets, reduction='none')

    # 均匀分布损失
    log_probs = F.log_softmax(logits, dim=-1)
    uniform_loss = -log_probs.mean(dim=-1)

    # 组合
    loss = (1 - epsilon) * ce_loss + epsilon * uniform_loss
    return loss.mean()


def label_smoothing_encouraging_loss(x, target, smoothing=0.1, log_end=0.75):
    assert smoothing < 1.0, "平滑系数必须小于1.0"

    # 计算log概率和概率值
    logprobs = F.log_softmax(x, dim=-1)
    probs = torch.exp(logprobs)

    # 计算基础奖励项 (likelihood bonus)
    bonus = torch.log(torch.clamp(1 - probs, min=1e-5))  # 避免log(0)

    # 分段函数调整：当概率超过log_end时使用线性近似
    if log_end != 1.0:
        y_log_end = torch.log(torch.ones_like(probs) - log_end)
        # 线性近似部分：在log_end处与原函数连续
        bonus_after_log_end = (probs - log_end) / (log_end - 1) + y_log_end
        # 选择适用的奖励项
        bonus = torch.where(probs > log_end, bonus_after_log_end, bonus)

    # 计算鼓励损失的"硬"部分（针对真实标签）
    el_loss = (bonus - logprobs).gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)

    # 计算鼓励损失的"平滑"部分（所有类别的平均）
    smooth_loss = (bonus - logprobs).mean(dim=-1)

    # 结合标签平滑：置信度部分 + 平滑部分
    confidence = 1.0 - smoothing
    loss = confidence * el_loss + smoothing * smooth_loss

    return loss.mean()

class traniner(object):
    def __init__(self, args):
        self.args = args
        # self.da = DB_MTL()
    def train(self, model, train_loader, optimizer, scheduler, device, ratio):
        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0

        model.to(device)
        model.train()

        # for imgs1, labels, _ in train_loader:
        for imgs1, labels in train_loader:
            imgs1 = imgs1.to(device)
            labels = labels.to(device)

            criterion = nn.CrossEntropyLoss(reduction='none')

            output, MC_loss = model(imgs1, labels, phase='train', ratio=ratio)

            loss1 = nn.CrossEntropyLoss()(output, labels)
            '''引入DFT'''
            # ce_loss = F.cross_entropy(output, labels, reduction='none')  # 形状: (32,)
            # probs = F.softmax(output, dim=-1)  # 对最后一个维度(类别维度)计算softmax
            # label_indices = labels.unsqueeze(-1)  # 形状: (32,) → (32, 1)
            # true_class_probs = probs.gather(1, label_indices)
            # true_class_probs = true_class_probs.squeeze(-1).detach()  # 形状: (32,)
            # weighted_losses = ce_loss * true_class_probs  # 形状: (32,)
            # loss1 = weighted_losses.mean()  # 形状: ()

            # loss = loss1 + 5 * MC_loss[1] + 1.5 * MC_loss[0]    # 特征散度函数的权值大于类特征损失，说明特征间的多样性较低，CLIP的特征很一般化，而不具有表情特异性
            # loss1 = MultiClassGHMC(output, labels)
            # loss1 = hardsample(output, labels)
            # loss1 = imb_cls(output, labels)
            # loss1 = label_smoothing_loss(output, labels)
            # loss1 = label_smoothing_encouraging_loss(output, labels)
            # loss = loss1 + MC_loss[0] + MC_loss[1] + MC_loss[2]
            loss = loss1 + MC_loss[0] + MC_loss[1]
            # loss = loss1 + MC_loss
            # loss = loss1
            # loss = self.da([loss1, MC_loss[0], MC_loss[1]])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_cnt += 1
            _, predicts = torch.max(output, 1)
            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num
            running_loss += loss

        scheduler.step()
        running_loss = running_loss / iter_cnt
        acc = correct_sum.float() / float(train_loader.dataset.__len__())
        return acc, running_loss


    def test(self, model, test_loader, device):
        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            iter_cnt = 0
            correct_sum = 0
            data_num = 0

            # for imgs1, labels, _ in test_loader:
            for imgs1, labels in test_loader:
                imgs1 = imgs1.to(device)
                labels = labels.to(device)

                outputs, _ = model(imgs1, labels, phase='test')
                # outputs, loss = model(imgs1, clip_model, labels, phase='test')

                loss = nn.CrossEntropyLoss()(outputs, labels)

                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)

                correct_num = torch.eq(predicts, labels).sum()
                correct_sum += correct_num

                running_loss += loss
                data_num += outputs.size(0)

            running_loss = running_loss / iter_cnt
            test_acc = correct_sum.float() / float(data_num)

        return test_acc, running_loss