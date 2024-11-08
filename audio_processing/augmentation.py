import librosa
import numpy as np
import soundfile as sf
import io
import base64

def augment_audio(audio_bytes, method='time_stretch'):
    # Load audio from bytes
    audio_io = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_io)
    
    if method == 'time_stretch':
        # Random stretch factor between 0.8 and 1.2
        stretch_factor = np.random.uniform(0.8, 1.2)
        augmented = librosa.effects.time_stretch(y, rate=stretch_factor)
    
    elif method == 'pitch_shift':
        # Random pitch shift between -2 and 2 semitones
        n_steps = np.random.uniform(-2, 2)
        augmented = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    
    elif method == 'add_noise':
        # Add random Gaussian noise
        noise_factor = 0.005
        noise = np.random.normal(0, noise_factor, len(y))
        augmented = y + noise
        # Ensure the output is in the valid range [-1, 1]
        augmented = np.clip(augmented, -1, 1)
    
    # Save augmented audio
    output_buffer = io.BytesIO()
    sf.write(output_buffer, augmented, sr, format='wav')
    audio_data = output_buffer.getvalue()
    
    # Convert to base64 for web playback
    audio_b64 = base64.b64encode(audio_data).decode()
    return f'data:audio/wav;base64,{audio_b64}' 