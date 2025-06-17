import os
import librosa
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from data_tools import scaled_in, inv_scaled_ou
from data_tools import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio

from pesq import pesq  


def prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
               audio_output_prediction, sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft):
    """ 
    This function takes pretrained weights, noisy voice sound to denoise, predicts
    the denoised sound and saves it to disk.
    """
    try:
        # Charger directement le modèle complet
        loaded_model = tf.keras.models.load_model(weights_path + '/model_unet.h5')
        print("Loaded model directly from h5 file")
    except Exception as e:
        print(f"Erreur lors du chargement direct du modèle : {e}")
        # Si échec, reconstruire le modèle et charger les poids
        from model_unet import unet
        loaded_model = unet()
        loaded_model.load_weights(weights_path + '/model_unet.h5')
        print("Loaded weights into new model")

    # Extraire les audios bruyants et convertir en numpy
    audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, sample_rate,
                                 frame_length, hop_length_frame, min_duration)

    # Dimension du spectrogramme carré
    dim_square_spec = int(n_fft / 2) + 1
    print(f"dim_square_spec: {dim_square_spec}")

    # Calcul amplitude et phase du son
    m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(
        audio, dim_square_spec, n_fft, hop_length_fft)

    # Normalisation globale -1 à 1
    X_in = scaled_in(m_amp_db_audio)
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)

    # Prédiction
    X_pred = loaded_model.predict(X_in)

    # Désnormaliser la sortie
    inv_sca_X_pred = inv_scaled_ou(X_pred)

    # Soustraire le bruit estimé du signal bruyant
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0]

    print(f"X_denoise.shape: {X_denoise.shape}")
    print(f"m_pha_audio.shape: {m_pha_audio.shape}")
    print(f"frame_length: {frame_length}")
    print(f"hop_length_fft: {hop_length_fft}")

    # Reconstruction audio à partir du spectrogramme débruité et phase
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)

    # Nombre total d’échantillons
    nb_samples = audio_denoise_recons.shape[0]

    # Sauvegarder tout en un seul fichier audio
    denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length) * 10  # facteur d'amplification si nécessaire

    # Construire chemin complet de sauvegarde
    save_path = os.path.join(dir_save_prediction, audio_output_prediction)
    wavfile.write(save_path, sample_rate, denoise_long[0, :].astype(np.float32))
    print(f"Audio débruité sauvegardé dans : {save_path}")

    # === Évaluation PESQ ===

    # Charger les fichiers audio pour PESQ
    rate_clean, clean = wavfile.read(os.path.join(audio_dir_prediction, audio_input_prediction[0]))  # fichier bruité
    rate_denoised, denoised = wavfile.read(save_path)  # fichier débruité

    if rate_clean != sample_rate or rate_denoised != sample_rate:
        print("Erreur de fréquence d'échantillonnage pour le calcul PESQ.")
    else:
        min_len = min(len(clean), len(denoised))
        clean = clean[:min_len]
        denoised = denoised[:min_len]

        try:
            if sample_rate == 8000:
                pesq_score = pesq(sample_rate, clean, denoised, 'nb')  # narrowband
            else:
                pesq_score = pesq(sample_rate, clean, denoised, 'wb')  # wideband
            print(f"PESQ Score: {pesq_score:.3f}")
        except Exception as e:
            print("Erreur lors du calcul du PESQ :", e)
