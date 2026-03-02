# Diagram Placements for Chapter 50

All diagrams have been successfully generated in the `diagrams/` directory. Here are the recommended placements in `content.md`:

## 1. Whisper Architecture Diagram
**File:** `diagrams/whisper_architecture.png`

**Location:** After line 56 (after the "Key Concept" callout)

**Markdown to add:**
```markdown
![Whisper Architecture](diagrams/whisper_architecture.png)

*Figure 1: Whisper's encoder-decoder transformer architecture. The encoder processes mel-spectrograms to extract acoustic features, while the decoder generates text autoregressively using special tokens to control the task (transcribe/translate), language, and timestamp behavior.*
```

---

## 2. Audio Representations Diagram
**File:** `diagrams/audio_representations.png`

**Location:** After line 108 (after the code block that ends with `print("- Mel-Spectrogram: ...")`)

**Markdown to add:**
```markdown
![Audio Representations](diagrams/audio_representations.png)

*Figure 2: Audio signal representations used in speech recognition. Top: Raw waveform showing amplitude over time. Middle: Spectrogram with linear frequency scale. Bottom: Mel-spectrogram with 80 frequency bins optimized for human perception—this is Whisper's input format.*
```

---

## 3. Model Comparison Diagram
**File:** `diagrams/model_comparison.png`

**Location:** After line 248 (after the comment "# WER (Word Error Rate) varies by language and audio quality")

**Markdown to add:**
```markdown
![Whisper Model Comparison](diagrams/model_comparison.png)

*Figure 3: Comparison of Whisper model variants. Top left: Parameter counts ranging from 39M (tiny) to 1.5B (large). Top right: Processing speed relative to large model. Bottom left: Word Error Rate ranges on clean audio. Bottom right: Speed-accuracy tradeoff showing the optimal balance—larger models are more accurate but slower.*
```

---

## 4. WER Calculation Diagram
**File:** `diagrams/wer_calculation.png`

**Location:** After line 562 (after the comment "# Child/teen speech (Whisper): 30-56%")

**Markdown to add:**
```markdown
![Word Error Rate Calculation](diagrams/wer_calculation.png)

*Figure 4: Word Error Rate (WER) calculation methodology. WER measures ASR accuracy by counting substitutions (S), insertions (I), and deletions (D) relative to the reference transcript. In this example, two substitutions ("quick"→"quik", "jumps"→"jump") result in WER = 22.22%. Lower WER indicates better performance, with human transcriptionists achieving ~4% and state-of-the-art models reaching 3-5% on clean speech.*
```

---

## 5. Pipeline Flow Diagram
**File:** `diagrams/pipeline_flow.png`

**Location:** After line 476 (after the paragraph ending with "This makes Whisper particularly valuable for international applications...")

**Markdown to add:**
```markdown
![Whisper ASR Pipeline](diagrams/pipeline_flow.png)

*Figure 5: Complete Whisper ASR pipeline from raw audio to text output. The pipeline includes preprocessing (resampling, normalization), feature extraction (mel-spectrogram), encoder processing, and autoregressive decoding with task-specific tokens. The diagram shows typical Real-Time Factors (RTF) for different model sizes and key capabilities like multilingual support and automatic language detection.*
```

---

## Generated Files Summary

All five diagrams have been successfully generated:

1. ✅ `audio_representations.png` (221 KB) - Shows waveform, spectrogram, and mel-spectrogram
2. ✅ `whisper_architecture.png` (188 KB) - Illustrates encoder-decoder architecture
3. ✅ `wer_calculation.png` (164 KB) - Explains WER metric with example
4. ✅ `model_comparison.png` (189 KB) - Compares all Whisper model sizes
5. ✅ `pipeline_flow.png` (193 KB) - Shows complete ASR pipeline

All diagrams follow the style guidelines:
- Consistent color palette (#2196F3, #4CAF50, #FF9800, #F44336, #9C27B0, #607D8B)
- White backgrounds
- 150 DPI resolution
- Font sizes ≥12pt
- Clear labels and annotations
- Maximum width ~800px
