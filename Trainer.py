import torch
import torch.nn as nn
import torch.nn.functional as F


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
            loss = loss1 + MC_loss[0] + MC_loss[1]

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
