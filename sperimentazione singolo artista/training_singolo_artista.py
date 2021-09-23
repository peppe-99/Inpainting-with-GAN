from matplotlib import pyplot as plt
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from model.generator import Generator
from model.discriminator import Discriminator
from utils.parameters import *

from utils.function import create_dir, ritagliare_centro, prepare_data, create_graphic_training, make_graphic_loss_G

if __name__ == '__main__':
    create_dir("log")
    create_dir("result/training")

    dataloader = prepare_data("./dataset/training")

    generator = Generator()
    discriminator = Discriminator()
    criterion = nn.BCELoss()
    criterion.cuda().to(device)
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)
    real_center = Variable(real_center)
    label = Variable(label)

    for epoch in range(0, 50):
        create_dir("result/training/epoca_%03d" % (epoch + 1))

        for i, data in enumerate(dataloader, 0):
            create_dir("result/training/epoca_%03d/batch_%03d" % (epoch + 1, i + 1))

            real_cpu, _ = data
            real_center_cpu = real_cpu[:, :, int(img_size / 4):int(img_size / 4) + int(img_size / 2),
                              int(img_size / 4):int(img_size / 4) + int(img_size / 2)]
            batch_size = real_cpu.size(0)
            real_cpu = real_cpu.cuda().to(device)
            real_center_cpu = real_center_cpu.cuda().to(device)

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
            D_x = output.mean().item()

            # Allenaimo il discriminatore con immagini generati del generatore
            fake = generator(input_cropped)
            fake.cuda().to(device)
            label.data.fill_(fake_label)
            output = discriminator(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Allenaimo il generatore
            # sarà più bravo quanto meglio il discriminatore sbaglierà nel riconoscere un immagine vera da una generata
            generator.zero_grad()
            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = discriminator(fake)
            loss_adv = criterion(output, label)

            # Calcoliamo la loss recostruction
            wtl2Matrix = real_center.clone()
            wtl2Matrix.data.fill_(0.999 * 10)
            wtl2Matrix.data[:, :, 4: int(img_size / 2 - 4), 4: int(img_size / 2 - 4)] = 0.999
            loss_rec = (fake - real_center).pow(2)
            loss_rec = loss_rec * wtl2Matrix
            loss_rec = loss_rec.mean()

            errG = (1 - 0.999) * loss_adv + 0.999 * loss_rec

            errG.backward()

            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Sostituiamo la parte mancante dell'immagine con quella generata del generatore e salviamo
            recon_image = input_cropped.clone()
            recon_image.data[:, :, int(img_size / 4):int(img_size / 4 + img_size / 2),
            int(img_size / 4):int(img_size / 4 + img_size / 2)] = fake.data

            # Salviamo le immagini reali, ritagliate e generate
            vutils.save_image(real_cpu, "result/training/epoca_%03d/batch_%03d/reali.png" % (epoch + 1, i + 1))
            vutils.save_image(input_cropped, 'result/training/epoca_%03d/batch_%03d/ritagliate.png' % (epoch + 1, i + 1))
            vutils.save_image(recon_image, 'result/training/epoca_%03d/batch_%03d/ricostruite.png' % (epoch + 1, i + 1))

            print(f'Epoca: [{epoch + 1}/{50}] Batch: [{i + 1}/{len(dataloader)}] '
                  f'Loss_D: %.4f Loss_G: %.4f [Loss_rec: %.4f | Loss_adv: %.4f]'
                  % (errD.mean().item(), errG.mean().item(), loss_rec.mean().item(), loss_adv.mean().item()))

            losses_reconstruction.append(round(loss_rec.mean().item(), 4))
            losses_adversarial.append(round(loss_adv.mean().item(), 4))
            losses_generatore.append(round(errG.mean().item(), 4))
            losses_discriminatore.append(round(errD.mean().item(), 4))

    # Salviamo il modello allenato
    torch.save(generator.state_dict(), "log/generatore_singolo_artista.pt")
    torch.save(discriminator.state_dict(), "./log/discriminatore_singolo_artista.pt")

    plt.figure()
    plt.title("Perdità di Generatore e Discriminatore Durante il Training (esperimento singolo artista)")
    plt.plot(losses_discriminatore, label="Discriminatore")
    plt.plot(losses_generatore, label="Generatore")
    plt.xlabel("Batch analizzati")
    plt.ylabel("Loss")
    plt.legend()
    fig = plt.gcf()
    fig.savefig('./log/losses_training_esperimento_singolo_artista.png')