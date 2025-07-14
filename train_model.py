import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from model_unet import unet
from data_tools import scaled_in, scaled_ou
from torchinfo import summary                                                                             #  pour affichage du modèle

def training(path_save_spectrogram, weights_path, name_model, training_from_scratch, epochs, batch_size):
    """
    Fonction pour entraîner le modèle U-Net++ sur des spectrogrammes bruités et propres.
    Sauvegarde le meilleur modèle sur disque.
    """
    # Chargement des données .npy
    X_in = np.load(path_save_spectrogram + 'noisy_voice_amp_db.npy')
    X_ou = np.load(path_save_spectrogram + 'voice_amp_db.npy')

    # Calcul du bruit à prédire
    X_ou = X_in - X_ou

    # Stats avant normalisation
    print("Stats avant normalisation :")
    print("X_in :", stats.describe(X_in.reshape(-1, 1)))
    print("X_ou :", stats.describe(X_ou.reshape(-1, 1)))

    # Normalisation
    X_in = scaled_in(X_in)
    X_ou = scaled_ou(X_ou)

    # Stats après normalisation
    print("Shape après normalisation :", X_in.shape, X_ou.shape)
    print("Stats après normalisation :")
    print("X_in :", stats.describe(X_in.reshape(-1, 1)))
    print("X_ou :", stats.describe(X_ou.reshape(-1, 1)))

    # Reshape (batch, 1, height, width)
    X_in = X_in.reshape(X_in.shape[0], 1, X_in.shape[1], X_in.shape[2])
    X_ou = X_ou.reshape(X_ou.shape[0], 1, X_ou.shape[1], X_ou.shape[2])

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_in, X_ou, test_size=0.10, random_state=42
    )

    # Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Dataloaders
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Modèle
    model = unet()
    if not training_from_scratch:
        model.load_state_dict(torch.load(os.path.join(weights_path, name_model + '.pth')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Affichage du résumé du modèle (comme TensorFlow)
    summary(model, input_size=(batch_size, 1, 128, 128))

    # Optimiseur et fonction de perte
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            sd1, sd2, sd3, sd4 = model(X_batch)

            # Somme des 4 MSE
            loss = sum(criterion(out, y_batch) for out in [sd1, sd2, sd3, sd4])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                sd1, sd2, sd3, sd4 = model(X_val)
                loss = sum(criterion(out, y_val) for out in [sd1, sd2, sd3, sd4])
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        #  Sauvegarde du meilleur modèle
        if avg_val_loss < best_val_loss:
            print("Meilleur modèle sauvegardé.")
            torch.save(model.state_dict(), os.path.join(weights_path, 'model_unet.pth'))
            best_val_loss = avg_val_loss

    # Affichage de la courbe de perte
    plt.plot(range(1, epochs + 1), train_losses, label='Training loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
