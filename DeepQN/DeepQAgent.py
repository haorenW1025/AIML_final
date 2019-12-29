import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# pylint: disable=no-member

""" setup GPU """
DEVICE = torch.device("cuda:0 " if torch.cuda.is_available() else "cpu")

class DeepQNet(nn.Module):

    def __init__(self):
        super(DeepQNet, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_net(x)

class DeepQAgent():

    def __init__(self, model, save_dir, learning_rate):
        self.save_dir = save_dir
        self.model = model.to(DEVICE)
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                                lr=learning_rate)

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def train(self, total_epoch):
        iters = 0
        for epoch in range(1, total_epoch + 1):
            self.model.train()
            for idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                iters += 1
                train_info = 'Epoch: [{0}][{1}/{2}]'.format(
                    epoch, idx+1, len(self.train_loader))
                output = self.model(data)

                ''' compute loss, backpropagation, update parameters '''
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())


            ''' save model '''
            self.save_model(os.path.join(self.save_dir,
                            'model_{}.pth.tar'.format(epoch)))

if __name__ == "__main__":
    from torchsummary import summary
    net = DeepQNet()
    summary(net, input_size=(3, 200, 200), batch_size=32)
