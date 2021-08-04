import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_size = 128      # dimensione delle immagini (64 x 64)
epochs = 100        # numero di epoche del training
batch_size = 64     # grandezza di un batch di campioni
lr = 0.0002         # tasso di apprendimento

input_real = torch.FloatTensor(batch_size, 3, 128, 128).cuda().to(device)     # Tensore immagine originale
input_cropped = torch.FloatTensor(batch_size, 3, 128, 128).cuda().to(device)  # Tensore immagine ritagliata
real_center = torch.FloatTensor(batch_size, 3, 64, 64).cuda().to(device)      # Tensore centro ritagliato
label = torch.FloatTensor(batch_size, 1, 1, 1).cuda().to(device)              # Etichetta
real_label = 1
fake_label = 0

losses_discriminatore = []
losses_generatore = []
losses_reconstruction = []
losses_adversarial = []

ricostruzione = []

TRAIN_RESULT = 'risultati/training/'    # Directory risultati training
TEST_RESULT = 'risultati/testing/'       # Directory risultati testing

"""
    x rappresenta un immagine di dimensione 3 x 64 x 64
    
    D(x) è il risultato del discriminatore sull'input x. Rappresenta la probabbilità che x sia 
    reale piuttosto che generata. D(x) dovrebbe essere alto quando x è reale e basso se è generato.
    Sostanzialmente D(x) e il discriminatore stesso sono un semplice classificatore binario.
    
    z è un vettore dello spazio latente campionato da una distribuzione normale standard.
    
    G(z) è la funzione generatrice che mappa il vettore latente z nello spazio dati. L'obiettivo della funzione G è
    quello di stimare la distribuzione da cui provengono i dati di allenamento (p_data) in modo da poter generare
    campioni falsi da quella disatribuzione stimata (p_g). Sostanzialmente l'obiettivo del generatore è quello di 
    generare nuove immagini cercando di confonderle tra quelle reali.
    
    D(G(z)) è la probabbilità che l'output del generatore G sia un immagine reale. 
    
    Sostanzialmente D e G giocano al gioco del minimax dove D cerca di massimizzare la probabbilità di classificare
    correttamente (accuratezza) le immagini reali e false [ log(D(x)) ]. Invece, G prova a minimizzare la probbabilità
    che D classifichi i suoi output come falsi [log(1 - D(G(z)))].
     
    Il gioco si risolve quando p_g = p_data ovvero immagini generate e reali combaciano ed il discriminatore sarà
    costretto a classificarle casualmente.
    
"""

