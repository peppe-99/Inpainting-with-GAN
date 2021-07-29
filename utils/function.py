import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

from utils.parameters import img_size, batch_size, TRAIN_RESULT
import torchvision.utils as vutils


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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return dataloader


# def save_batch(real_cpu, input_cropped, recon_image, batch, epoch):
