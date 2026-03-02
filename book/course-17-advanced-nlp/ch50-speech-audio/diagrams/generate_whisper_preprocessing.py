#!/usr/bin/env python3
"""Generate Whisper audio preprocessing visualization"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Create a synthetic audio signal (5 seconds, 16kHz sampling rate)
np.random.seed(42)
sample_rate = 16000
duration = 5.0
t = np.linspace(0, duration, int(sample_rate * duration))

# Simulate speech-like signal: combination of harmonics with amplitude modulation
fundamental_freq = 150  # Hz (typical male voice pitch)
signal = np.zeros_like(t)
for harmonic in range(1, 6):
    signal += (1.0 / harmonic) * np.sin(2 * np.pi * harmonic * fundamental_freq * t)

# Add amplitude modulation (simulates syllables)
modulation = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # 3 Hz modulation
signal = signal * modulation

# Add some noise
signal += 0.1 * np.random.randn(len(signal))

# Normalize
signal = signal / np.max(np.abs(signal))

# Create figure with three subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 1. Raw waveform
axes[0].plot(t[:8000], signal[:8000], linewidth=0.5, color='#2196F3')
axes[0].set_xlabel('Time (seconds)', fontsize=12)
axes[0].set_ylabel('Amplitude', fontsize=12)
axes[0].set_title('Step 1: Raw Audio Waveform', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 0.5)

# 2. Spectrogram (linear frequency scale)
D = librosa.stft(signal, n_fft=2048, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
img1 = librosa.display.specshow(S_db, sr=sample_rate, hop_length=512,
                                 x_axis='time', y_axis='linear', ax=axes[1],
                                 cmap='viridis')
axes[1].set_xlabel('Time (seconds)', fontsize=12)
axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
axes[1].set_title('Step 2: Spectrogram (Linear Frequency Scale)', fontsize=13, fontweight='bold')
fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')

# 3. Mel-spectrogram (perceptually-motivated)
M = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=80, n_fft=2048, hop_length=512)
M_db = librosa.power_to_db(M, ref=np.max)
img2 = librosa.display.specshow(M_db, sr=sample_rate, hop_length=512,
                                 x_axis='time', y_axis='mel', ax=axes[2],
                                 cmap='viridis')
axes[2].set_xlabel('Time (seconds)', fontsize=12)
axes[2].set_ylabel('Mel Frequency', fontsize=12)
axes[2].set_title('Step 3: Mel-Spectrogram (Whisper Input)', fontsize=13, fontweight='bold')
fig.colorbar(img2, ax=axes[2], format='%+2.0f dB')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch50/diagrams/whisper_audio_preprocessing.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Audio preprocessing visualization saved")
print(f"  Duration: {duration} seconds")
print(f"  Sample rate: {sample_rate} Hz")
print(f"  Mel-spectrogram shape: {M.shape} (80 mel bins × {M.shape[1]} time frames)")
