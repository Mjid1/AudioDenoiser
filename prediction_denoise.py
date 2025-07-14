import os
import torch
import numpy as np
from scipy.io import wavfile
from data_tools import (
    scaled_in, inv_scaled_ou,
    audio_files_to_numpy,
    numpy_audio_to_matrix_spectrogram,
    matrix_spectrogram_to_numpy_audio
)
from model_unet import unet
from pesq import pesq

def prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction,
               audio_input_prediction, audio_output_prediction,
               sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft):
    """
    Prédiction sur un audio bruité avec le modèle U-Net++ (PyTorch),
    reconstruction audio, calcul PESQ et early exit si PESQ > 2.5.
    """

    # Chargement du modèle 
    model = unet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(os.path.join(weights_path, 'model_unet.pth'), map_location=device))
    model.eval()
    model.to(device)
    print("Modèle PyTorch chargé avec succès.")

    # Chargement de l audio bruite
    audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, sample_rate,
                                 frame_length, hop_length_frame, min_duration)

    #  Spectrogramme 
    dim_square_spec = int(n_fft / 2) + 1
    m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(
        audio, dim_square_spec, n_fft, hop_length_fft
    )

    # Pretraitement 
    X_in = scaled_in(m_amp_db_audio)
    X_in = X_in.reshape(X_in.shape[0], 1, X_in.shape[1], X_in.shape[2])  # (batch, 1, H, W)
    X_in_tensor = torch.tensor(X_in, dtype=torch.float32).to(device)

    # Predictions 
    with torch.no_grad():
        sd1, sd2, sd3, sd4 = model(X_in_tensor)

    # Convertir en numpy
    preds = [sd.cpu().numpy() for sd in [sd1, sd2, sd3, sd4]]

    #  Chargement de l audio propre 
    clean_path = os.path.join(audio_dir_prediction, audio_input_prediction[0])
    rate_clean, clean = wavfile.read(clean_path)

    final_audio = None
    for i, sd in enumerate(preds):
        # Inversion de la normalisation
        inv_pred = inv_scaled_ou(sd)
        X_denoise = m_amp_db_audio - inv_pred[:, :, :, 0]
        X_denoise = np.clip(X_denoise, a_min=-80.0, a_max=0.0)

        # Reconstruction du signal audio
        audio_denoise = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio,
                                                          frame_length, hop_length_fft)
        denoise_long = audio_denoise.reshape(1, -1)

        # Sauvegarde temporaire pour PESQ
        temp_path = os.path.join(dir_save_prediction, f"temp_sd{i+1}.wav")
        wavfile.write(temp_path, sample_rate, denoise_long[0, :].astype(np.float32))

        # PESQ
        rate_denoised, denoised = wavfile.read(temp_path)
        min_len = min(len(clean), len(denoised))
        clean_trimmed = clean[:min_len]
        denoised_trimmed = denoised[:min_len]

        try:
            mode = 'wb' if sample_rate == 16000 else 'nb'
            pesq_score = pesq(sample_rate, clean_trimmed, denoised_trimmed, mode)
            print(f" PESQ Sd{i+1} = {pesq_score:.3f}")
            if pesq_score > 2.5:
                print(f"Early exit : on utilise Sd{i+1}")
                final_audio = denoised_trimmed
                break
        except Exception as e:
            print(f" Erreur PESQ Sd{i+1} : {e}")

    # Si aucun PESQ > 2.5, on prend la dernière sortie 
    if final_audio is None:
        print(" Aucun PESQ > 2.5, on prend la dernière sortie (Sd4)")
        final_audio = denoised_trimmed

    #  Sauvegarde de l’audio final 
    save_path = os.path.join(dir_save_prediction, audio_output_prediction)
    wavfile.write(save_path, sample_rate, final_audio.astype(np.float32))
    print(f"Audio final sauvegardé dans : {save_path}")
