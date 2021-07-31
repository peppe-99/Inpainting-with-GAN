from model.generator import Generator
from model.discriminator import Discriminator
from utils.function import prepare_data, ritagliare_centro, create_dir
from utils.parameters import *

import torchvision.utils as vutils

from tqdm import tqdm
import torch

create_dir(TEST_RESULT)

generator = Generator()
generator.load_state_dict(torch.load("./log/generator.pt"))
generator.eval()

discriminator = Discriminator()
discriminator.load_state_dict(torch.load("./log/discriminator.pt"))
discriminator.eval()

dataloader = prepare_data("./dataset/testing/")

i = 1

for data in tqdm(dataloader, ncols=100, desc="Batch analizzati"):
    real_cpu, _ = data
    real_center_cpu = real_cpu[:, :, int(img_size / 4):int(img_size / 4) + int(img_size / 2),
                      int(img_size / 4):int(img_size / 4) + int(img_size / 2)]
    batch_size = real_cpu.size(0)
    real_cpu = real_cpu.cuda()
    real_center_cpu = real_center_cpu.cuda()
    real_cpu.to(device)
    real_center_cpu.to(device)

    # Individuiamo e ritagliamo il centro dell'immagine reale
    input_real, input_cropped, real_center = ritagliare_centro(input_real, input_cropped, real_cpu, real_center,
                                                               real_center_cpu)

    fake = generator(input_cropped)

    recon_image = input_cropped.clone()
    recon_image.data[:, :, int(img_size / 4):int(img_size / 4 + img_size / 2),
    int(img_size / 4):int(img_size / 4 + img_size / 2)] = fake.data

    vutils.save_image(recon_image, TEST_RESULT + f"/ricostruite_testing_{i}.png")

    i += 1
