import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from model_unet import unet
from data_tools import scaled_in, scaled_ou

def training(path_save_spectrogram, weights_path, name_model, training_from_scratch, epochs, batch_size):
    """
    Fonction pour entraîner le modèle U-Net++ sur des spectrogrammes bruités et propres.
    Sauvegarde le meilleur modèle sur disque.
    """
    # Chargement des donnees
    X_in = np.load(path_save_spectrogram + 'noisy_voice_amp_db.npy')
    X_ou = np.load(path_save_spectrogram + 'voice_amp_db.npy')

    # Calcul du bruit a predire (bruit = noisy - clean)
    X_ou = X_in - X_ou

    # Affichage statistique avant normalisation
    print("Stats avant normalisation :")
    print("X_in :", stats.describe(X_in.reshape(-1, 1)))
    print("X_ou :", stats.describe(X_ou.reshape(-1, 1)))

    # Normalisation entre -1 et 1 (une seule fois)
    X_in = scaled_in(X_in)                  # noisy input
    X_ou = scaled_ou(X_ou)                  # bruit cible

    # Affichage shape et stats après normalisation
    print("Shape après normalisation :", X_in.shape, X_ou.shape)
    print("Stats après normalisation :")
    print("X_in :", stats.describe(X_in.reshape(-1, 1)))
    print("X_ou :", stats.describe(X_ou.reshape(-1, 1)))

    # Reshape pour Conv2D : (batch, height, width, channels)
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
    X_ou = X_ou.reshape(X_ou.shape[0], X_ou.shape[1], X_ou.shape[2], 1)

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_in, X_ou, test_size=0.10, random_state=42)

    # Chargement ou création du modèle
    if training_from_scratch:
        model = unet()
    else:
        model = unet(pretrained_weights=weights_path + name_model + '.h5')

    # Création des cibles multi-sorties (une même cible pour chaque sortie)
    y_train_multi = [y_train] * 4
    y_test_multi = [y_test] * 4

    # Checkpoint pour sauvegarder le meilleur modèle
    checkpoint = ModelCheckpoint(
        weights_path + '/model_unet.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto'
    )

    # Affichage résumé modèle
    model.summary()

    # Entraînement
    history = model.fit(
        X_train, y_train_multi,
        validation_data=(X_test, y_test_multi),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint],
        shuffle=True,
        verbose=1
    )

    # Visualisation des pertes
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(range(1, len(loss) + 1), loss, label='Training loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()




