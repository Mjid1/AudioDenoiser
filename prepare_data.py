import os
import librosa
from data_tools import audio_files_to_numpy
from data_tools import blend_noise_randomly, numpy_audio_to_matrix_spectrogram
import numpy as np
from scipy.io import wavfile



def create_data(noise_dir, voice_dir, path_save_time_serie, path_save_sound, path_save_spectrogram, sample_rate,
min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft):
    """This function will randomly blend some clean voices from voice_dir with some noises from noise_dir
    and save the spectrograms of noisy voice, noise and clean voices to disk as well as complex phase,
    time series and sounds. This aims at preparing datasets for denoising training. It takes as inputs
    parameters defined in args module"""

    list_noise_files = os.listdir(noise_dir)
    list_voice_files = os.listdir(voice_dir)

    def remove_ds_store(lst):
        """remove mac specific file if present"""
        if '.DS_Store' in lst:
            lst.remove('.DS_Store')

        return lst

    list_noise_files = remove_ds_store(list_noise_files)
    list_voice_files = remove_ds_store(list_voice_files)

    nb_voice_files = len(list_voice_files)
    nb_noise_files = len(list_noise_files)


    # Extracting noise and voice from folder and convert to numpy
    noise = audio_files_to_numpy(noise_dir, list_noise_files, sample_rate,
                                     frame_length, hop_length_frame_noise, min_duration)

    voice = audio_files_to_numpy(voice_dir, list_voice_files,
                                     sample_rate, frame_length, hop_length_frame, min_duration)

    # Blend some clean voices with random selected noises (and a random level of noise)
    prod_voice, prod_noise, prod_noisy_voice = blend_noise_randomly(
            voice, noise, nb_samples, frame_length)

    # To save the long audio generated to disk to QC:
    noisy_voice_long = prod_noisy_voice.reshape(1, nb_samples * frame_length)
    wavfile.write(path_save_sound + 'noisy_voice_long.wav', sample_rate, noisy_voice_long[0, :].astype(np.float32))
    voice_long = prod_voice.reshape(1, nb_samples * frame_length)
    wavfile.write(path_save_sound + 'voice_long.wav', sample_rate, voice_long[0, :].astype(np.float32))
    noise_long = prod_noise.reshape(1, nb_samples * frame_length)
    wavfile.write(path_save_sound + 'noise_long.wav', sample_rate, noise_long[0, :].astype(np.float32))

    # Squared spectrogram dimensions
    dim_square_spec = int(n_fft / 2) + 1

    # Create Amplitude and phase of the sounds
    m_amp_db_voice,  m_pha_voice = numpy_audio_to_matrix_spectrogram(
            prod_voice, dim_square_spec, n_fft, hop_length_fft)
    m_amp_db_noise,  m_pha_noise = numpy_audio_to_matrix_spectrogram(
            prod_noise, dim_square_spec, n_fft, hop_length_fft)
    m_amp_db_noisy_voice,  m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
            prod_noisy_voice, dim_square_spec, n_fft, hop_length_fft)
    
    
    
    # pour resoudre prob spect 
    print("\n=== Validation finale ===")
    print(f"Voice spectrogram: min={np.min(m_amp_db_voice):.2f} dB, max={np.max(m_amp_db_voice):.2f} dB")
    print(f"Noise spectrogram: min={np.min(m_amp_db_noise):.2f} dB, max={np.max(m_amp_db_noise):.2f} dB")
    print(f"Noisy voice spectrogram: min={np.min(m_amp_db_noisy_voice):.2f} dB, max={np.max(m_amp_db_noisy_voice):.2f} dB")
    if np.all(m_amp_db_voice == 0):
                raise ValueError("ERREUR: Spectrogramme de la voix plat !")


    # Save to disk for Training / QC
    np.save(path_save_time_serie + 'voice_timeserie', prod_voice)
    np.save(path_save_time_serie + 'noise_timeserie', prod_noise)
    np.save(path_save_time_serie + 'noisy_voice_timeserie', prod_noisy_voice)


    np.save(path_save_spectrogram + 'voice_amp_db', m_amp_db_voice)
    np.save(path_save_spectrogram + 'noise_amp_db', m_amp_db_noise)
    np.save(path_save_spectrogram + 'noisy_voice_amp_db', m_amp_db_noisy_voice)

    np.save(path_save_spectrogram + 'voice_pha_db', m_pha_voice)
    np.save(path_save_spectrogram + 'noise_pha_db', m_pha_noise)
    np.save(path_save_spectrogram + 'noisy_voice_pha_db', m_pha_noisy_voice)
