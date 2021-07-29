from model.generator import Generator
from model.discriminator import Discriminator
from utils.parameters import *

import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt


from utils.function import create_dir, ritagliare_centro, prepare_data

if __name__ == '__main__':
    create_dir(TRAIN_RESULT)

    # Otteniamo il dataloader per l'allenamento
    dataloader = prepare_data("./dataset/resized/")

    # Instanziamo generatore, discriminatore, ottimizzatori e criterio per il calcolo della loss
    generator = Generator()
    discriminator = Discriminator()
    criterion = nn.BCELoss()
    criterion.cuda().to(device)
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training
    for epoch in range(0, epochs):
        create_dir(TRAIN_RESULT + "epoca_%03d" % (epoch + 1))

        for i, data in enumerate(dataloader, 0):
            create_dir(TRAIN_RESULT + "epoca_%03d/batch_%03d/reali" % (epoch + 1, i + 1))
            create_dir(TRAIN_RESULT + "epoca_%03d/batch_%03d/ritagliate" % (epoch + 1, i + 1))
            create_dir(TRAIN_RESULT + "epoca_%03d/batch_%03d/ricostruite" % (epoch + 1, i + 1))


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

            # Alleniamo il discriminatore con immagini reali
            discriminator.zero_grad()
            with torch.no_grad():
                label.resize_((batch_size, 1, 1, 1)).fill_(real_label)

            output = discriminator(real_center)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # Allenaimo il discriminatore con immagini generati del generatore
            fake = generator(input_cropped)
            fake.cuda()
            label.data.fill_(fake_label)
            output = discriminator(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Allenaimo il generatore
            # sarà più bravo quanto meglio il discriminatore sbaglierà nel riconoscere un immagine vera da una generata
            generator.zero_grad()
            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = discriminator(fake)
            errG_D = criterion(output, label)

            wtl2Matrix = real_center.clone()
            wtl2Matrix.data.fill_(0.999 * 10)
            wtl2Matrix.data[:, :, 0: int(img_size / 2), 0: int(img_size / 2)] = 0.999

            errG_l2 = (fake - real_center).pow(2)
            errG_l2 = errG_l2 * wtl2Matrix
            errG_l2 = errG_l2.mean()

            errG = (1 - 0.999) * errG_D + 0.999 * errG_l2

            errG.backward()

            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('Epoca: [%d / %d] batch: [%d / %d]\nPerdita discriminatore: %.4f Perdità generatore: %.4f\n'
                  % (epoch + 1, epochs, i + 1, len(dataloader), errD.data, errG))

            # Sostituiamo la parte mancante dell'immagine con quella generata del generatore e salviamo
            recon_image = input_cropped.clone()
            recon_image.data[:, :, int(img_size / 4):int(img_size / 4 + img_size / 2),
                            int(img_size / 4):int(img_size / 4 + img_size / 2)] = fake.data

            # Salviamo le immagini reali, ritagliate e generate
            vutils.save_image(real_cpu, TRAIN_RESULT + 'epoca_%03d/batch_%03d/reali.png' % (epoch + 1, i + 1))
            vutils.save_image(input_cropped, TRAIN_RESULT + 'epoca_%03d/batch_%03d/ritagliate.png' % (epoch + 1, i + 1))
            vutils.save_image(recon_image, TRAIN_RESULT + 'epoca_%03d/batch_%03d/ricostruite.png' % (epoch + 1, i + 1))
