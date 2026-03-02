#!/usr/bin/env python3
"""Generate noise reduction comparison visualization"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Create synthetic noisy audio
np.random.seed(42)
sr = 16000
duration = 3.0
t = np.linspace(0, duration, int(sr * duration))

# Speech-like signal with formants
signal = np.zeros_like(t)
formants = [700, 1220, 2600]
for freq in formants:
    signal += np.sin(2 * np.pi * freq * t) * np.exp(-0.002 * (t - 1.5)**2)

# Add realistic amplitude envelope (syllables)
envelope = 0.5 * (1 + np.sin(2 * np.pi * 2.5 * t))
signal = signal * envelope

# Add background noise (simulating cafe)
noise = 0.3 * np.random.randn(len(signal))
noisy_signal = signal + noise
noisy_signal = noisy_signal / np.max(np.abs(noisy_signal))

# Simple noise reduction via spectral subtraction
def spectral_subtraction(audio, sr, noise_duration=0.5):
    """Simple spectral subtraction for noise reduction"""
    noise_samples = int(noise_duration * sr)
    noise_segment = audio[:noise_samples]

    # Compute noise spectrum
    noise_stft = librosa.stft(noise_segment)
    noise_spectrum = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

    # Compute signal STFT
    signal_stft = librosa.stft(audio)
    signal_magnitude = np.abs(signal_stft)
    signal_phase = np.angle(signal_stft)

    # Subtract noise spectrum
    subtracted_magnitude = np.maximum(signal_magnitude - 1.5 * noise_spectrum, 0.0)

    # Reconstruct signal
    subtracted_stft = subtracted_magnitude * np.exp(1j * signal_phase)
    denoised_audio = librosa.istft(subtracted_stft)

    return denoised_audio

denoised_signal = spectral_subtraction(noisy_signal, sr)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Noisy audio mel-spectrogram
M_noisy = librosa.feature.melspectrogram(y=noisy_signal, sr=sr, n_mels=80)
M_noisy_db = librosa.power_to_db(M_noisy, ref=np.max)
img1 = librosa.display.specshow(M_noisy_db, sr=sr, x_axis='time', y_axis='mel',
                                 ax=axes[0, 0], cmap='viridis')
axes[0, 0].set_title('Noisy Audio Mel-Spectrogram', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Time (seconds)', fontsize=12)
axes[0, 0].set_ylabel('Mel Frequency', fontsize=12)
fig.colorbar(img1, ax=axes[0, 0], format='%+2.0f dB')

# Denoised audio mel-spectrogram
M_denoised = librosa.feature.melspectrogram(y=denoised_signal, sr=sr, n_mels=80)
M_denoised_db = librosa.power_to_db(M_denoised, ref=np.max)
img2 = librosa.display.specshow(M_denoised_db, sr=sr, x_axis='time', y_axis='mel',
                                 ax=axes[0, 1], cmap='viridis')
axes[0, 1].set_title('Denoised Audio Mel-Spectrogram', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Time (seconds)', fontsize=12)
axes[0, 1].set_ylabel('Mel Frequency', fontsize=12)
fig.colorbar(img2, ax=axes[0, 1], format='%+2.0f dB')

# Waveform comparison - noisy
axes[1, 0].plot(t[:8000], noisy_signal[:8000], linewidth=0.5, color='#F44336', alpha=0.7)
axes[1, 0].set_xlabel('Time (seconds)', fontsize=12)
axes[1, 0].set_ylabel('Amplitude', fontsize=12)
axes[1, 0].set_title('Noisy Waveform', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, 0.5)

# Waveform comparison - denoised
axes[1, 1].plot(t[:8000], denoised_signal[:8000], linewidth=0.5, color='#4CAF50', alpha=0.7)
axes[1, 1].set_xlabel('Time (seconds)', fontsize=12)
axes[1, 1].set_ylabel('Amplitude', fontsize=12)
axes[1, 1].set_title('Denoised Waveform', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlim(0, 0.5)

# Add SNR information
snr_noisy = 10 * np.log10(np.var(signal) / np.var(noise))
axes[1, 0].text(0.02, 0.95, f'SNR: {snr_noisy:.1f} dB',
               transform=axes[1, 0].transAxes,
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch50/diagrams/noise_reduction.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Generated noise_reduction.png")
print(f"  SNR (noisy): {snr_noisy:.1f} dB")
