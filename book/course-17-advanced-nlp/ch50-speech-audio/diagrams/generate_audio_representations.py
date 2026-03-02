import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

# Create a simple synthetic audio signal (1 second of a 440 Hz tone)
sr = 16000  # Sample rate
duration = 1.0
t = np.linspace(0, duration, int(sr * duration))
audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A440 note

# Create figure with three subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Color palette
COLOR_BLUE = '#2196F3'

# 1. Waveform (time domain)
axes[0].plot(t[:1000], audio[:1000], linewidth=1.2, color=COLOR_BLUE)
axes[0].set_title('Waveform (Time Domain)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time (seconds)', fontsize=12)
axes[0].set_ylabel('Amplitude', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, t[999])

# 2. Spectrogram (time-frequency domain)
D = librosa.stft(audio)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
img1 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis')
axes[1].set_title('Spectrogram (Linear Frequency Scale)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
axes[1].set_xlabel('Time (seconds)', fontsize=12)
fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')

# 3. Mel-Spectrogram (perceptually-motivated)
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
img2 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2], cmap='viridis')
axes[2].set_title('Mel-Spectrogram (80 Mel Bands - Whisper Input)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Mel Frequency', fontsize=12)
axes[2].set_xlabel('Time (seconds)', fontsize=12)
fig.colorbar(img2, ax=axes[2], format='%+2.0f dB')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch50/diagrams/audio_representations.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Generated audio_representations.png")
