import librosa
import numpy as np
import os
import warnings

def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):
    """Split audio into frames with validation"""
    if len(sound_data) < frame_length:
        raise ValueError(f"Audio too short ({len(sound_data)} samples) for frame length {frame_length}")
    
    sequence_sample_length = sound_data.shape[0]
    sound_data_list = [sound_data[start:start + frame_length] for start in range(
        0, sequence_sample_length - frame_length + 1, hop_length_frame)]
    
    if not sound_data_list:
        raise ValueError("No frames created - check audio length and hop size")
    
    return np.vstack(sound_data_list)

def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
    """Load audio files with enhanced validation"""
    list_sound_array = []

    for file in list_audio_files:
        filepath = os.path.join(audio_dir, file)
        try:
            y, sr = librosa.load(filepath, sr=sample_rate)
            
            # Signal validation
            if np.max(np.abs(y)) < 0.001:  # Seuil de détection de silence
                warnings.warn(f"Silent audio detected in {file} - max amplitude: {np.max(np.abs(y))}")
                continue
                
            total_duration = librosa.get_duration(y=y, sr=sr)
            if total_duration < min_duration:
                print(f"File {file} is too short ({total_duration:.2f}s < {min_duration}s)")
                continue
                
            frames = audio_to_audio_frame_stack(y, frame_length, hop_length_frame)
            list_sound_array.append(frames)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    if not list_sound_array:
        raise ValueError("No valid audio files found!")
    
    return np.vstack(list_sound_array)

def blend_noise_randomly(voice, noise, nb_samples, frame_length):
    """Mix voice and noise with amplitude checks"""
    # Validation des entrées
    if voice.size == 0 or noise.size == 0:
        raise ValueError("Empty voice or noise array")
    
    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))
    prod_noisy_voice = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        id_voice = np.random.randint(0, voice.shape[0])
        id_noise = np.random.randint(0, noise.shape[0])
        #level_noise = np.random.uniform(0.2, 0.8)
        level_noise = np.random.uniform(0.05, 0.4)
        
        voice_sample = voice[id_voice, :]
        noise_sample = noise[id_noise, :] * level_noise
        
        # Normalisation pour éviter le clipping
        max_val = max(np.max(np.abs(voice_sample)), np.max(np.abs(noise_sample)))
        if max_val > 0.8:  # Seuil de clipping
            voice_sample = voice_sample * 0.8 / max_val
            noise_sample = noise_sample * 0.8 / max_val
            
        prod_voice[i, :] = voice_sample
        prod_noise[i, :] = noise_sample
        prod_noisy_voice[i, :] = voice_sample + noise_sample

    return prod_voice, prod_noise, prod_noisy_voice

def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio):
    """Convert audio to spectrogram with robust dB scaling"""
    stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)
    
    # Protection contre les valeurs nulles
    if np.all(stftaudio_magnitude == 0):
        warnings.warn("Zero magnitude detected - replacing with small values")
        stftaudio_magnitude = np.full_like(stftaudio_magnitude, 1e-10)
    
    # Conversion dB avec référence absolue
    stftaudio_magnitude_db = librosa.amplitude_to_db(
        stftaudio_magnitude, ref=np.max, amin=1e-10)
    
    return stftaudio_magnitude_db, stftaudio_phase

def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
    """Convert audio frames to spectrograms with validation"""
    nb_audio = numpy_audio.shape[0]
    
    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)
    
    for i in range(nb_audio):
        # Vérification du signal d'entrée
        if np.all(numpy_audio[i] == 0):
            warnings.warn(f"Zero audio detected in frame {i}")
            
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            n_fft, hop_length_fft, numpy_audio[i])
        
        # Validation du spectrogramme
        if np.all(m_mag_db[i] == -np.inf):
            m_mag_db[i] = np.zeros_like(m_mag_db[i])
            warnings.warn(f"Flat spectrogram in frame {i}")

    print(f"Spectrogram range: {np.min(m_mag_db):.2f} to {np.max(m_mag_db):.2f} dB")
    return m_mag_db, m_phase

# [ces autres fonctions restent inchangées  pour affichage specto...]
def magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):
    """This functions reverts a spectrogram to an audio"""

    stftaudio_magnitude_rev = librosa.db_to_amplitude(stftaudio_magnitude_db, ref=1.0)

    # taking magnitude and phase of audio
    audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
    audio_reconstruct = librosa.core.istft(audio_reverse_stft, hop_length=hop_length_fft, length=frame_length)

    return audio_reconstruct

def matrix_spectrogram_to_numpy_audio(m_mag_db, m_phase, frame_length, hop_length_fft)  :
    """This functions reverts the matrix spectrograms to numpy audio"""

    list_audio = []

    nb_spec = m_mag_db.shape[0]

    for i in range(nb_spec):

        audio_reconstruct = magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, m_mag_db[i], m_phase[i])
        list_audio.append(audio_reconstruct)

    return np.vstack(list_audio)

def scaled_in(matrix_spec):
    "global scaling apply to noisy voice spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec

def scaled_ou(matrix_spec):
    "global scaling apply to noise models spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec -6 )/82
    return matrix_spec

def inv_scaled_in(matrix_spec):
    "inverse global scaling apply to noisy voices spectrograms"
    matrix_spec = matrix_spec * 50 - 46
    return matrix_spec

def inv_scaled_ou(matrix_spec):
    "inverse global scaling apply to noise models spectrograms"
    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec



