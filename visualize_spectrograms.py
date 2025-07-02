import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os

PATHS = {
    "noise": "data/Train/spectrogram/noise_amp_db.npy",
    "voice": "data/Train/spectrogram/voice_amp_db.npy",
    "noisy_voice": "data/Train/spectrogram/noisy_voice_amp_db.npy"
}

SAMPLE_RATE = 8000
HOP_LENGTH = 63
N_FFT = 255

def load_and_validate(key):
    """Charge et valide les données avant affichage"""
    try:
        data = np.load(PATHS[key])
        print(f"\n=== Traitement {key} ===")
        print(f"Forme initiale: {data.shape}")
        
        if len(data.shape) == 3:
            print("Moyenne des 5 spectrogrammes")
            data = np.mean(data, axis=0)  # Moyenne sur l'axe du batch
            
        print(f"Valeurs min/max: {np.min(data):.2f}/{np.max(data):.2f} dB")
        return data.T if data.shape[0] > data.shape[1] else data
        
    except Exception as e:
        print(f" Erreur avec {key}: {str(e)}")
        return None

def plot_combined_spectrograms():
    """Affiche les 3 spectrogrammes côte à côte"""
    plt.figure(figsize=(18, 6))
    
    # Noise
    plt.subplot(1, 3, 1)
    noise = load_and_validate("noise")
    if noise is not None:
        librosa.display.specshow(noise, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                               x_axis='time', y_axis='linear', vmin=-80, vmax=0)
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogramme Noise")

    # Voice
    plt.subplot(1, 3, 2)
    voice = load_and_validate("voice")
    if voice is not None:
        librosa.display.specshow(voice, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                               x_axis='time', y_axis='linear', vmin=-80, vmax=0)
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogramme Voice")

    # Noisy Voice
    plt.subplot(1, 3, 3)
    noisy = load_and_validate("noisy_voice")
    if noisy is not None:
        librosa.display.specshow(noisy, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                               x_axis='time', y_axis='linear', vmin=-80, vmax=0)
        plt.colorbar(format='%+2.0f dB')
        plt.title("Spectrogramme Noisy Voice")

    plt.tight_layout()
    plt.savefig("combined_spectrograms.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    print("=== Début ===")
    
    # Vérification des fichiers
    missing = [k for k,v in PATHS.items() if not os.path.exists(v)]
    if missing:
        print(f"Fichiers manquants: {missing}")
    else:
        plot_combined_spectrograms()
    
    print("=== Terminé ===")