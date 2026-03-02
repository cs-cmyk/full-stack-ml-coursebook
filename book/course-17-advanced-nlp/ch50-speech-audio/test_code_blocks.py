"""
Test script for code blocks in content.md
Tests each code block in order to verify they run correctly
"""

import sys
import traceback

def test_block(name, code, globals_dict):
    """Test a single code block and return success status"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    try:
        exec(code, globals_dict)
        print(f"✓ {name} PASSED")
        return True, None
    except Exception as e:
        error_msg = f"✗ {name} FAILED\n{traceback.format_exc()}"
        print(error_msg)
        return False, error_msg

# Track all failures
failures = []
globals_dict = {}

# Block 1: Visualization
block1 = """
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Create a simple synthetic audio signal (1 second of a 440 Hz tone)
sr = 16000  # Sample rate
duration = 1.0
t = np.linspace(0, duration, int(sr * duration))
audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A440 note

# Create figure with three subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 1. Waveform (time domain)
axes[0].plot(t[:1000], audio[:1000], linewidth=0.8)
axes[0].set_title('Waveform (Time Domain)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

# 2. Spectrogram (time-frequency domain)
D = librosa.stft(audio)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
img1 = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis')
axes[1].set_title('Spectrogram (Linear Frequency Scale)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Frequency (Hz)')
fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')

# 3. Mel-Spectrogram (perceptually-motivated)
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
img2 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2], cmap='viridis')
axes[2].set_title('Mel-Spectrogram (80 Mel Bands - Whisper Input)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Mel Frequency')
axes[2].set_xlabel('Time (seconds)')
fig.colorbar(img2, ax=axes[2], format='%+2.0f dB')

plt.tight_layout()
plt.savefig('audio_representations.png', dpi=150, bbox_inches='tight')
plt.close()

print("Visualization saved as 'audio_representations.png'")
print("\\nKey differences:")
print("- Waveform: Shows amplitude over time (1D signal)")
print("- Spectrogram: Shows frequency content over time (linear frequency scale)")
print("- Mel-Spectrogram: Emphasizes perceptually-relevant frequencies (Whisper's input)")
"""

success, error = test_block("Block 1: Visualization", block1, globals_dict)
if not success:
    failures.append(("Block 1: Visualization", error))

# Block 2: Part 1 - Loading and Preprocessing Audio
block2 = """
# Automatic Speech Recognition with Whisper
# All imports and complete setup
import whisper
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create a synthetic speech-like audio for demonstration
# In practice, you'd load a real audio file with librosa.load()
def create_sample_audio(filename="sample_speech.wav", duration=10):
    \"\"\"Create a sample audio file for demonstration\"\"\"
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))

    # Simulate speech-like frequencies (fundamental + harmonics)
    # This is simplified; real speech is much more complex
    frequencies = [200, 400, 600, 800, 1000]  # Fundamental + harmonics
    audio = np.zeros_like(t)
    for i, freq in enumerate(frequencies):
        amplitude = 0.2 / (i + 1)  # Decreasing amplitude for harmonics
        audio += amplitude * np.sin(2 * np.pi * freq * t)

    # Add some variation to simulate speech dynamics
    envelope = np.abs(np.sin(2 * np.pi * 0.5 * t))
    audio = audio * envelope

    # Save as WAV file
    sf.write(filename, audio, sr)
    return filename, sr

# Create sample audio
audio_file, sample_rate = create_sample_audio()
print(f"Created sample audio: {audio_file}")
print(f"Sample rate: {sample_rate} Hz")

# Load audio file with librosa (automatically resamples to target sr)
audio, sr = librosa.load(audio_file, sr=16000)
print(f"\\nLoaded audio shape: {audio.shape}")
print(f"Duration: {len(audio) / sr:.2f} seconds")
print(f"Sample rate: {sr} Hz")

# Verify audio properties
print(f"\\nAudio statistics:")
print(f"  Min amplitude: {audio.min():.4f}")
print(f"  Max amplitude: {audio.max():.4f}")
print(f"  Mean: {audio.mean():.4f}")
print(f"  Std: {audio.std():.4f}")
"""

success, error = test_block("Block 2: Part 1 - Loading and Preprocessing Audio", block2, globals_dict)
if not success:
    failures.append(("Block 2: Part 1", error))

# Block 3: Part 2 - Loading Whisper Models
block3 = """
# Load different Whisper model sizes
# Models: tiny, base, small, medium, large (v1, v2, v3)

print("Loading Whisper models...")
print("Note: First run will download models (may take a few minutes)\\n")

# Load base model (74M parameters)
model_base = whisper.load_model("base", device=device)
print(f"Loaded 'base' model")
print(f"  Parameters: ~74M")
print(f"  Multilingual: Yes")
print(f"  English-only version available: Yes")

# Model characteristics
model_info = {
    "tiny": {"params": "39M", "relative_speed": "~32x", "wer": "~10-15%"},
    "base": {"params": "74M", "relative_speed": "~16x", "wer": "~8-12%"},
    "small": {"params": "244M", "relative_speed": "~6x", "wer": "~6-9%"},
    "medium": {"params": "769M", "relative_speed": "~2x", "wer": "~4-7%"},
    "large": {"params": "1550M", "relative_speed": "1x", "wer": "~3-5%"}
}

print("\\nWhisper Model Comparison:")
print(f"{'Model':<10} {'Parameters':<12} {'Speed':<15} {'Typical WER':<15}")
print("-" * 55)
for model_name, info in model_info.items():
    print(f"{model_name:<10} {info['params']:<12} {info['relative_speed']:<15} {info['wer']:<15}")

print("\\nNote: Speed is relative to 'large' model")
print("WER (Word Error Rate) varies by language and audio quality")
"""

success, error = test_block("Block 3: Part 2 - Loading Whisper Models", block3, globals_dict)
if not success:
    failures.append(("Block 3: Part 2", error))

# Block 4: Part 3 - Basic Transcription
block4 = """
# Transcribe audio with Whisper
print("Transcribing audio with base model...\\n")

# Measure inference time
start_time = time.time()

# Transcribe (automatic language detection)
result = model_base.transcribe(audio_file, fp16=False)

inference_time = time.time() - start_time
audio_duration = len(audio) / sr

# Calculate Real-Time Factor
rtf = inference_time / audio_duration

print("Transcription Result:")
print("-" * 60)
print(f"Detected Language: {result['language']}")
print(f"\\nTranscript:\\n{result['text']}")
print("-" * 60)

print(f"\\nPerformance Metrics:")
print(f"  Audio Duration: {audio_duration:.2f} seconds")
print(f"  Inference Time: {inference_time:.2f} seconds")
print(f"  Real-Time Factor (RTF): {rtf:.3f}")
print(f"  Speed: {1/rtf:.1f}x real-time" if rtf < 1 else f"  Speed: {rtf:.2f}x slower than real-time")

# Note: Since we used synthetic audio, the transcription won't be meaningful
# With real speech audio, you'd see accurate transcription
print("\\n(Note: Synthetic audio produces no meaningful transcription)")
print("For real usage, load actual speech audio with:")
print("  audio, sr = librosa.load('path/to/speech.wav', sr=16000)")
"""

success, error = test_block("Block 4: Part 3 - Basic Transcription", block4, globals_dict)
if not success:
    failures.append(("Block 4: Part 3", error))

# Block 5: Part 4 - Word-Level Timestamps
block5 = """
# Get word-level timestamps for alignment
print("Transcribing with word-level timestamps...\\n")

result_with_timestamps = model_base.transcribe(
    audio_file,
    word_timestamps=True,
    fp16=False
)

print("Segments with Word-Level Timing:")
print("-" * 80)

# Display first 3 segments (or all if fewer)
for i, segment in enumerate(result_with_timestamps['segments'][:3]):
    print(f"\\nSegment {i+1}:")
    print(f"  Time: {segment['start']:.2f}s - {segment['end']:.2f}s")
    print(f"  Text: {segment['text']}")

    # Show individual words with timestamps (if available)
    if 'words' in segment and segment['words']:
        print(f"  Words:")
        for word_info in segment['words'][:5]:  # Show first 5 words
            print(f"    {word_info['start']:.2f}s - {word_info['end']:.2f}s: '{word_info['word']}'")

print("\\n" + "-" * 80)
"""

success, error = test_block("Block 5: Part 4 - Word-Level Timestamps", block5, globals_dict)
if not success:
    failures.append(("Block 5: Part 4", error))

# Block 6: Part 5 - Multilingual Transcription and Translation
block6 = """
# Demonstrate language detection and translation capabilities
print("Whisper Multilingual Capabilities\\n")

# Simulate loading different language audio files
# In practice, you'd have actual Spanish, French, etc. audio files
languages_demo = {
    "English": "sample_speech.wav",  # Our sample file
    # "Spanish": "spanish_speech.wav",  # Hypothetical
    # "French": "french_speech.wav",    # Hypothetical
}

print("1. Language Detection (Automatic):")
print("-" * 60)

# Whisper can automatically detect language
result_auto = model_base.transcribe(audio_file, task="transcribe")
print(f"Detected language: {result_auto['language']}")
print(f"Language confidence: {result_auto.get('language_probability', 'N/A')}")

print("\\n2. Forced Language (Override Detection):")
print("-" * 60)

# Force specific language
result_forced = model_base.transcribe(audio_file, language="en", task="transcribe")
print(f"Forced language: en (English)")
print(f"Transcript: {result_forced['text'][:100]}...")

print("\\n3. Translation to English:")
print("-" * 60)

# The 'translate' task converts any language to English
result_translate = model_base.transcribe(audio_file, task="translate")
print(f"Task: Translate to English")
print(f"Translated text: {result_translate['text'][:100]}...")
print("\\nNote: When source is already English, translation = transcription")

# Demonstration of how task specification works
print("\\n4. Task Token Mechanism:")
print("-" * 60)
print("Whisper uses special tokens to specify behavior:")
print("  <|startoftranscript|><|en|><|transcribe|><|notimestamps|>")
print("     ^                   ^      ^             ^")
print("     |                   |      |             |")
print("     Begin            Language Task      Timestamp mode")
print("\\nTask options:")
print("  - 'transcribe': Output in the source language")
print("  - 'translate': Always output in English")
"""

success, error = test_block("Block 6: Part 5 - Multilingual Transcription and Translation", block6, globals_dict)
if not success:
    failures.append(("Block 6: Part 5", error))

# Block 7: Part 6 - Evaluation with Word Error Rate
block7 = """
# Calculate Word Error Rate (WER)
import jiwer  # Library for WER calculation

# For demonstration, we'll create a hypothetical scenario
# In practice, you'd have ground truth transcripts

# Example with real speech scenario (hypothetical)
ground_truth = "the quick brown fox jumps over the lazy dog"
hypothesis = "the quik brown fox jump over the lazy dog"  # Simulated ASR output

# Calculate WER
wer = jiwer.wer(ground_truth, hypothesis)

# Get detailed metrics
measures = jiwer.compute_measures(ground_truth, hypothesis)

print("Word Error Rate (WER) Calculation")
print("=" * 70)
print(f"\\nGround Truth: {ground_truth}")
print(f"Hypothesis:   {hypothesis}")

print(f"\\nWER: {wer:.4f} ({wer*100:.2f}%)")
print(f"\\nDetailed Metrics:")
print(f"  Substitutions: {measures['substitutions']}")
print(f"  Insertions:    {measures['insertions']}")
print(f"  Deletions:     {measures['deletions']}")
print(f"  Total Words:   {measures['hits'] + measures['substitutions'] + measures['deletions']}")

# Show alignment
print("\\nAlignment:")
alignment = jiwer.process_words(ground_truth, hypothesis)
print(f"  Reference: {' '.join(alignment.references[0])}")
print(f"  Hypothesis: {' '.join(alignment.hypotheses[0])}")

# WER formula explanation
print("\\nWER Formula:")
print("  WER = (S + I + D) / N")
print("  where S=substitutions, I=insertions, D=deletions, N=total words")
S = measures['substitutions']
I = measures['insertions']
D = measures['deletions']
N = measures['hits'] + measures['substitutions'] + measures['deletions']
print(f"  WER = ({S} + {I} + {D}) / {N} = {(S+I+D)/N:.4f}")

# Benchmark WER values for context
print("\\nTypical WER Benchmarks:")
print("  Human transcriptionists:     ~4%")
print("  Whisper Large (clean audio):  3-5%")
print("  Whisper Base (clean audio):   8-12%")
print("  Challenging conditions:       15-30%")
print("  Child/teen speech (Whisper):  30-56%")
"""

success, error = test_block("Block 7: Part 6 - Evaluation with Word Error Rate", block7, globals_dict)
if not success:
    failures.append(("Block 7: Part 6", error))

# Print summary
print(f"\n\n{'='*70}")
print("TEST SUMMARY")
print(f"{'='*70}")
print(f"Total blocks tested: 7")
print(f"Failures: {len(failures)}")

if failures:
    print("\\nFailed blocks:")
    for name, error in failures:
        print(f"  - {name}")
else:
    print("\\n✓ All blocks passed!")
