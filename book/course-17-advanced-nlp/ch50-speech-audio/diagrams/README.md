# Chapter 50 Diagrams - Automatic Speech Recognition (Whisper)

## Summary

Successfully generated and integrated **5 educational diagrams** for Chapter 50 on Automatic Speech Recognition using the Whisper architecture.

## Generated Diagrams

### 1. Whisper Architecture (`whisper_architecture.png`)
- **Size:** 188 KB
- **Location in content.md:** Line 58
- **Description:** Comprehensive diagram showing the encoder-decoder transformer architecture
- **Key Elements:**
  - Audio input → Mel-spectrogram conversion
  - Transformer encoder with self-attention
  - Transformer decoder with cross-attention
  - Special token system for task control
  - Autoregressive feedback loop
  - Training details (680k hours, 97 languages, 39M-1550M parameters)

### 2. Audio Representations (`audio_representations.png`)
- **Size:** 221 KB
- **Location in content.md:** Line 114
- **Description:** Three-panel visualization showing audio signal transformations
- **Panels:**
  1. Waveform (time domain)
  2. Spectrogram (linear frequency scale)
  3. Mel-spectrogram (80 mel bands - Whisper's input format)
- **Generated using:** librosa library with 1-second 440 Hz tone

### 3. Model Comparison (`model_comparison.png`)
- **Size:** 189 KB
- **Location in content.md:** Line 258
- **Description:** Four-panel comparison of Whisper model variants
- **Panels:**
  1. Parameter counts (39M to 1550M)
  2. Relative processing speed (32× to 1×)
  3. Word Error Rate ranges (3-15%)
  4. Speed-accuracy tradeoff scatter plot
- **Models:** tiny, base, small, medium, large

### 4. WER Calculation (`wer_calculation.png`)
- **Size:** 164 KB
- **Location in content.md:** Line 580
- **Description:** Educational diagram explaining Word Error Rate calculation
- **Key Elements:**
  - Example sentence with reference and hypothesis
  - Visual highlighting of substitution errors
  - Formula: WER = (S + I + D) / N
  - Example calculation: 2 substitutions / 9 words = 22.22%
  - Benchmark comparison table
  - Error type legend (Substitution, Insertion, Deletion)

### 5. Pipeline Flow (`pipeline_flow.png`)
- **Size:** 193 KB
- **Location in content.md:** Line 489
- **Description:** Complete ASR pipeline from audio to text
- **Pipeline Stages:**
  1. Audio Input (raw waveform, 16 kHz)
  2. Preprocessing (resample, normalize, 30s chunks)
  3. Feature Extraction (STFT, mel filterbank, 80 bins)
  4. Encoder (transformer, self-attention)
  5. Decoder (autoregressive, cross-attention)
- **Additional Info:**
  - Task token inputs
  - Processing metrics (RTF for different models)
  - Key capabilities checklist

## Design Standards Applied

✅ **Color Palette:** Consistent use of:
- Blue (#2196F3) - Primary components
- Green (#4CAF50) - Success/output
- Orange (#FF9800) - Special features
- Red (#F44336) - Errors
- Purple (#9C27B0) - Decoder/feedback
- Gray (#607D8B) - Metadata

✅ **Typography:**
- Minimum font size: 12pt
- Bold titles and headers
- Italic annotations for emphasis
- Monospace for code/tokens

✅ **Image Quality:**
- Resolution: 150 DPI
- Maximum width: ~800px
- White background
- Clear borders and edges

✅ **Layout:**
- Clear annotations and labels
- Consistent spacing
- Axis labels on all plots
- Legends where appropriate
- `plt.tight_layout()` applied to all matplotlib figures

## Files Generated

```
diagrams/
├── audio_representations.png          (221 KB) ✓
├── whisper_architecture.png           (188 KB) ✓
├── wer_calculation.png                (164 KB) ✓
├── model_comparison.png               (189 KB) ✓
├── pipeline_flow.png                  (193 KB) ✓
├── generate_audio_representations.py  (2.0 KB)
├── generate_whisper_architecture.py   (6.2 KB)
├── generate_wer_calculation.py        (5.2 KB)
├── generate_model_comparison.py       (4.4 KB)
├── generate_pipeline_flow.py          (5.8 KB)
├── insert_diagrams.py                 (3.6 KB)
├── DIAGRAM_PLACEMENTS.md              (3.2 KB)
└── README.md                          (this file)
```

## Integration Status

✅ All 5 diagrams successfully inserted into `content.md`
✅ Figure captions added with clear descriptions
✅ Proper markdown image syntax used
✅ Sequential figure numbering (Figure 1-5)
✅ No orphaned diagram markers remaining

## Verification

Run these commands to verify:

```bash
# Check all diagram references
grep -n "!\[.*\](diagrams/" ../content.md

# Verify all PNG files exist
ls -lh *.png

# Check figure captions
grep -n "^\*Figure" ../content.md
```

## Regeneration

To regenerate any diagram:

```bash
python generate_audio_representations.py
python generate_whisper_architecture.py
python generate_wer_calculation.py
python generate_model_comparison.py
python generate_pipeline_flow.py
```

To reinsert all diagrams into content.md:

```bash
python insert_diagrams.py
```

## Notes

- All diagrams are educational and textbook-appropriate
- Diagrams complement the text without duplicating information
- Visual hierarchy guides the reader through complex concepts
- Color coding is consistent across all diagrams
- Diagrams are self-contained with titles and legends

---

**Generated:** 2026-03-01
**Chapter:** 50.1 - Automatic Speech Recognition (Whisper Architecture)
**Total Diagrams:** 5
**Total Size:** ~955 KB
