import torchvision.utils as vutils

from matplotlib import pyplot as plt

from model.discriminator import Discriminator
from model.generator import Generator

from utils.function import prepare_data, ritagliare_centro
from utils.parameters import *

discriminator = Discriminator()
discriminator.load_state_dict(torch.load("./log/discriminatore_singolo_artista.pt"))
discriminator.eval()

generator = Generator()
generator.load_state_dict(torch.load("./log/generatore_singolo_artista.pt"))
generator.eval()

accuratezza = []
criterion = torch.nn.BCELoss()

accuracy = 0.0
total = 0.0
num_img = 1

dataloader = prepare_data("./dataset/testing/")

for data in dataloader:
    real_cpu, _ = data
    real_center_cpu = real_cpu[:, :, int(img_size / 4):int(img_size / 4) + int(img_size / 2),
                      int(img_size / 4):int(img_size / 4) + int(img_size / 2)]
    batch_size = real_cpu.size(0)
    real_cpu = real_cpu.cuda()
    real_center_cpu = real_center_cpu.cuda()
    real_cpu.to(device)
    real_center_cpu.to(device)

    input_real, input_cropped, real_center = ritagliare_centro(input_real, input_cropped, real_cpu, real_center,
                                                               real_center_cpu)

    output = discriminator(real_center)
    D_x = output.mean().item()
    print("label: 1 \t output: %f" % D_x)
    accuracy += D_x
    total += 1
    accuratezza.append(100 * accuracy / total)

    fake = generator(input_cropped)
    output = discriminator(fake.detach())
    D_G_z = output.mean().item()
    print("label: 0 \t output: %f" % D_G_z)
    accuracy += (1 - D_G_z)
    total += 1
    accuratezza.append(100 * accuracy / total)

    recon_image = input_cropped.clone()
    recon_image.data[:, :, int(img_size / 4):int(img_size / 4 + img_size / 2),
    int(img_size / 4):int(img_size / 4 + img_size / 2)] = fake.data

    for i in range(0, batch_size):
        vutils.save_image([input_real[i], input_cropped[i], recon_image[i]],
                          f"./result/testing/ricostruite_{num_img}.png")
        num_img += 1

print(f"Accuratezza: {round((100 * accuracy / total), 2)}%")

plt.figure()
plt.title("Accuratezza del discriminatore (singolo artista)")
plt.plot(accuratezza)
plt.xlabel("Batch analizzati")
plt.ylabel("Accuratezza")
fig = plt.gcf()
fig.savefig('./log/accuratezza_discriminatore_singolo_artista.png')
