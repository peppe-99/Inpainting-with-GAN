import torch
from torch.nn import Module, Sequential, Conv2d, LeakyReLU, BatchNorm2d, ReLU, ConvTranspose2d, Tanh

from utils.function import weights_init
from utils.parameters import device


class Generator(Module):

    # Struttura Generatore
    def __init__(self):
        super(Generator, self).__init__()

        self.t1 = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            LeakyReLU(0.2, inplace=True)
        )

        self.t2 = Sequential(
            Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(64),
            LeakyReLU(0.2, inplace=True)
        )

        self.t3 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True)
        )

        self.t4 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2, inplace=True)
        )

        self.t5 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(512),
            LeakyReLU(0.2, inplace=True)

        )

        self.t6 = Sequential(
            Conv2d(512, 4000, kernel_size=(4, 4)),
            BatchNorm2d(4000),
            ReLU()
        )

        self.t7 = Sequential(
            ConvTranspose2d(in_channels=4000, out_channels=512, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(512),
            ReLU()
        )

        self.t8 = Sequential(
            ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(256),
            ReLU()
        )

        self.t9 = Sequential(
            ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(128),
            ReLU()
        )

        self.t10 = Sequential(
            ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(64),
            ReLU()
        )

        self.t11 = Sequential(
            ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            BatchNorm2d(64),
            ReLU()
        )

        self.t12 = Sequential(
            ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(4, 4), stride=2, padding=1),
            Tanh()
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
        x = self.t6(x)
        x = self.t7(x)
        x = self.t8(x)
        x = self.t9(x)
        x = self.t10(x)
        x = self.t11(x)
        x = self.t12(x)
        return x

    def summary(self):
        print(self)
