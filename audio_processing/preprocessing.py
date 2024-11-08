import librosa
import numpy as np
import soundfile as sf
import io
import base64

def preprocess_audio(audio_bytes, method='resample', params=None):
    if params is None:
        params = {}
    
    # Load audio from bytes
    audio_io = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_io)
    
    if method == 'resample':
        # Default target sampling rate: 22050 Hz
        target_sr = params.get('target_sr', 22050)
        processed = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Save processed audio
        output_buffer = io.BytesIO()
        sf.write(output_buffer, processed, target_sr, format='wav')
        audio_data = output_buffer.getvalue()
        
    elif method == 'mfcc':
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Convert MFCC to audio-like representation for playback
        # This is a simplified conversion for demonstration
        processed = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=sr)
        
        # Save processed audio
        output_buffer = io.BytesIO()
        sf.write(output_buffer, processed, sr, format='wav')
        audio_data = output_buffer.getvalue()
    
    # Convert to base64 for web playback
    audio_b64 = base64.b64encode(audio_data).decode()
    return f'data:audio/wav;base64,{audio_b64}' 