import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from matplotlib import pyplot as plt

from utils.parameters import img_size, batch_size


def create_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


# Pesi personalizzati
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def ritagliare_centro(input_real, input_cropped, real_cpu, real_center, real_center_cpu):
    with torch.no_grad():
        input_real.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.resize_(real_cpu.size()).copy_(real_cpu)
        real_center.resize_(real_center_cpu.size()).copy_(real_center_cpu)
        input_cropped[:, 0, int(img_size / 4):int(img_size / 4 + img_size / 2),
        int(img_size / 4):int(img_size / 4 + img_size / 2)] = 2 * 117.0 / 255.0 - 1.0
        input_cropped[:, 1, int(img_size / 4):int(img_size / 4 + img_size / 2),
        int(img_size / 4):int(img_size / 4 + img_size / 2)] = 2 * 104.0 / 255.0 - 1.0
        input_cropped[:, 2, int(img_size / 4):int(img_size / 4 + img_size / 2),
        int(img_size / 4):int(img_size / 4 + img_size / 2)] = 2 * 123.0 / 255.0 - 1.0

        return input_real, input_cropped, real_center


def prepare_data(dataset_path):
    transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.CenterCrop(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dset.ImageFolder(root=dataset_path, transform=transform)
    assert dataset
    if dataset_path.endswith('testing/'):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return dataloader


def create_graphic_loss_training(G_losses, D_losses):
    plt.figure(figsize=(20, 10))
    plt.title("Perdità di Generatore e Discriminatore Durante il Training")
    plt.plot(G_losses, label="Generatore")
    plt.plot(D_losses, label="Discriminatore")
    plt.xlabel("Batch")
    plt.ylabel("Accuratezza")
    plt.legend()
    fig = plt.gcf()
    fig.savefig('./log/losses_training.png')


def create_graphic_accuratezza(perdita_ricostruzione):
    plt.figure(figsize=(20, 10))
    plt.title("Perdità ricostruzione delle immagini")
    plt.plot(perdita_ricostruzione)
    plt.xlabel("Batch")
    plt.ylabel("Perdita")
    fig = plt.gcf()
    fig.savefig('./log/perdita_ricostruzione_testing.png')
