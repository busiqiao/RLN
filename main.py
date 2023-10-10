import torch.optim
import numpy as np
from dataset import EEGDataset
from model.RLN import RLN
from model.utils import train, test
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
CUDA_LAUNCH_BLOCKING = 1

epochs = 35
batch_size = 64
num_class = 6

if __name__ == '__main__':
    dataPath = 'H:\\EEG\\EEGDATA'

    for i in range(0, 2):
        dataset = EEGDataset(dataPath + '\\' + 'S' + str(i + 1) + '.mat')
        train_size = int(0.9 * len(dataset))
        test_size = int(len(dataset)) - train_size

        # 分割数据集
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=3, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)

        print('第{}位受试者:  train_num={}, test_num={}'.format(int(i + 1), train_size, test_size))

        # 创建模型
        model = RLN(num_class=num_class, channel=124, batch_size=batch_size).cuda()

        # 检查参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

        # 设置网络参数
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        for epoch in range(epochs):
            losses = []
            accuracy = []
            loop = tqdm(train_loader, total=len(train_loader))

            for step, (x, y) in enumerate(loop):
                if batch_size == 64 and step == 72:
                    continue
                x = x.cuda()
                y = y.cuda()
                loss, y_ = train(model=model, optimizer=optimizer, x=x, y=y)
                corrects = (torch.argmax(y_, dim=1).data == y.data)
                acc = corrects.cpu().int().sum().numpy() / batch_size

                loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
                loop.set_postfix(loss=loss.item(), acc=acc)









