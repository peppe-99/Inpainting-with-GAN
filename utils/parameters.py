import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_size = 128      # dimensione delle immagini
epochs = 100        # numero di epoche del training
batch_size = 64     # grandezza di un batch di campioni
lr = 0.0002         # tasso di apprendimento

input_real = torch.FloatTensor(batch_size, 3, 128, 128).cuda().to(device)     # Tensore immagine originale
input_cropped = torch.FloatTensor(batch_size, 3, 128, 128).cuda().to(device)  # Tensore immagine ritagliata
real_center = torch.FloatTensor(batch_size, 3, 64, 64).cuda().to(device)      # Tensore centro ritagliato
label = torch.FloatTensor(batch_size, 1, 1, 1).cuda().to(device)              # Etichetta
real_label = 1
fake_label = 0

TRAIN_RESULT = 'risultati/training/'    # Directory risultati training


