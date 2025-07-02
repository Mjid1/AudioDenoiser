import os
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from data_tools import (
    scaled_in, inv_scaled_ou,
    audio_files_to_numpy,
    numpy_audio_to_matrix_spectrogram,
    matrix_spectrogram_to_numpy_audio
)
from pesq import pesq

def prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction,
               audio_input_prediction, audio_output_prediction,
               sample_rate, min_duration, frame_length, hop_length_frame, n_fft, hop_length_fft):
    """
    Effectue la prédiction sur un audio bruité avec un modèle U-Net++, 
    applique early exit basé sur PESQ (> 2.5), et sauvegarde l’audio débruité.
    """
    # Chargement du modèle
    try:
        loaded_model = tf.keras.models.load_model(weights_path + '/model_unet.h5')
        print(" Modèle chargé depuis fichier .h5")
    except Exception as e:
        print(f" Erreur de chargement direct : {e}")
        from model_unet import unet
        loaded_model = unet(compile_model=False)
        loaded_model.load_weights(weights_path + '/model_unet.h5')
        print(" Modèle reconstruit et poids chargés")

    # Chargement audio bruité
    audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, sample_rate,
                                 frame_length, hop_length_frame, min_duration)

    # Transformation en spectrogramme
    dim_square_spec = int(n_fft / 2) + 1
    m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(audio, dim_square_spec, n_fft, hop_length_fft)
    
    # c est cela ce que j ai enlever maintenantX_in = scaled_in(m_amp_db_audio).reshape(m_amp_db_audio.shape[0], m_amp_db_audio.shape[1], m_amp_db_audio.shape[2], 1)
    
    X_in = scaled_in(m_amp_db_audio)
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
    
    
    
    
    
    # Prédiction [sd1, sd2, sd3, sd4]
    preds = loaded_model.predict(X_in)
    

    # Lecture du fichier clean (une seule fois)
    clean_path = os.path.join(audio_dir_prediction, audio_input_prediction[0])
    rate_clean, clean = wavfile.read(clean_path)

    # Early exit basé sur PESQ
    final_audio = None
    for i, sd in enumerate(preds):
        # Inverser la normalisation et reconstruire le spectrogramme propre
        inv_pred = inv_scaled_ou(sd)
        X_denoise = m_amp_db_audio - inv_pred[:, :, :, 0]
        X_denoise = np.clip(X_denoise, a_min=-80.0, a_max=0.0)                                           # éviter valeurs négatives
        #print(f"Stats inv_pred sd{i+1} : min={inv_pred.min():.4f}, max={inv_pred.max():.4f}, mean={inv_pred.mean():.4f}")
        
        # Reconstruire le signal audio
        audio_denoise = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, frame_length, hop_length_fft)
        denoise_long = audio_denoise.reshape(1, -1)

        # Sauvegarder sortie intermédiaire                                                                           (facultatif, utile pour debug)
        temp_path = os.path.join(dir_save_prediction, f"temp_sd{i+1}.wav")
        wavfile.write(temp_path, sample_rate, denoise_long[0, :].astype(np.float32))

        # Comparer avec l audio propre (PESQ)
        rate_denoised, denoised = wavfile.read(temp_path)
        min_len = min(len(clean), len(denoised))
        clean_trimmed = clean[:min_len]
        denoised_trimmed = denoised[:min_len]

        try:
            mode = 'wb' if sample_rate == 16000 else 'nb'
            pesq_score = pesq(sample_rate, clean_trimmed, denoised_trimmed, mode)
            print(f" PESQ Sd{i+1} = {pesq_score:.3f}")
            if pesq_score > 2.5:
                print(f" Early exit: on utilise Sd{i+1}")
                final_audio = denoised_trimmed
                break
        except Exception as e:
            print(f" Erreur PESQ Sd{i+1} : {e}")

    # Si aucune sortie n a depasse le seuil PESQ on prend la derniere
    if final_audio is None:
        print(" Aucun PESQ > 2.5, on prend la dernière sortie (Sd4)")
        final_audio = denoised_trimmed

    # Sauvegarde de l audio final
    save_path = os.path.join(dir_save_prediction, audio_output_prediction)
    wavfile.write(save_path, sample_rate, final_audio.astype(np.float32))
    print(f" Audio final sauvegardé dans : {save_path}")


