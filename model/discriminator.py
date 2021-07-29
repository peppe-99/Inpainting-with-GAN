import torch
from torch.nn import Module, Sequential, BatchNorm2d, LeakyReLU, Conv2d, Sigmoid

from utils.function import weights_init

from utils.parameters import device


class Discriminator(Module):

    # Struttura Discriminatore
    def __init__(self):
        super(Discriminator, self).__init__()

        self.t1 = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(64),
            LeakyReLU(0.2, inplace=True)
        )

        self.t2 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True)
        )

        self.t3 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True)
        )

        self.t4 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(512),
            LeakyReLU(0.2, inplace=True)
        )

        self.t5 = Sequential(
            Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=1, padding=0),
            Sigmoid()
        )

        self.apply(weights_init)
        self.cuda()
        self.to(device=device)

    def forward(self, x):
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        return x

    def summary(self):
        print(self)
