import torch.optim
import numpy as np
from dataset import EEGDataset
from model.RLN import RLN
from model.utils import train, test
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

epochs = 1
batch_size = 64
# num_class = 6
num_class = 72

history = np.zeros(10)

if __name__ == '__main__':
    dataPath = 'H:\\EEG\\EEGDATA'

    for i in range(0, 10):
        dataset = EEGDataset(file_path=dataPath + '\\' + 'S' + str(i + 1) + '.mat', num_class=num_class)
        train_size = int(0.9 * len(dataset))
        test_size = int(len(dataset)) - train_size

        # 分割数据集
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=3, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)

        # 创建模型
        model = RLN(num_class=num_class, channel=124, batch_size=batch_size).cuda()

        # 检查参数数量
        if i == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params}")
            print(f"Trainable parameters: {trainable_params}")

        print('第{}位受试者:  train_num={}, test_num={}'.format(int(i + 1), train_size, test_size))

        # 设置网络参数
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(epochs):
            losses = []
            accuracy = []
            train_loop = tqdm(train_loader, total=len(train_loader))

            f = 0
            for (x, y) in train_loop:
                f += 1
                if batch_size == 64 and f == 73:
                    continue
                x = x.cuda()
                y = y.cuda()
                loss, y_ = train(model=model, optimizer=optimizer, x=x, y=y)
                corrects = (torch.argmax(y_, dim=1).data == y.data)
                acc = corrects.cpu().int().sum().numpy() / batch_size
                losses.append(loss)
                accuracy.append(acc)

                train_loop.set_description(f'Epoch [{epoch + 1}/{epochs}] - Train')
                train_loop.set_postfix(loss=loss.item(), acc=acc)

            test_loop = tqdm(test_loader, total=len(test_loader))
            sum_acc, flag = 0, 0
            max_acc = 0
            for (xx, yy) in test_loop:
                if batch_size == 64 and flag == 8:  # 跳过第81个step的原因是kfold分配的验证集在batich_size=64时，
                    continue  # 第81个step无法填满，导致除以精度异常甚至报错
                val_loss, val_acc = test(model=model, x=xx, y=yy)
                val_acc = val_acc / batch_size
                sum_acc += val_acc
                losses.append(val_loss)
                accuracy.append(val_acc)
                flag += 1

                test_loop.set_description(f'               Test ')
                test_loop.set_postfix(loss=val_loss.item(), acc=val_acc)
                # print('test step:{}/{} loss={:.5f} acc={:.3f}'.format(step, int(test_size / batch_size), val_loss,
                #                                                       val_acc))
            epoch_acc = sum_acc / flag
            print('本轮epoch平均准确率：{}'.format(epoch_acc))
            if epoch_acc > max_acc:
                history[i] = epoch_acc
        print('受试者{}训练完成，测试准确率：{}'.format(i+1, history[i]))
        print('---------------------------------------------------------')
    print(history)
    print('训练完成，平均准确率：{}'.format(np.mean(history)))


