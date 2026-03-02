#!/usr/bin/env python3
"""
Script to insert diagram references into content.md
"""

import re

# Read the original content
with open('../content.md', 'r') as f:
    content = f.read()

# Define diagram insertions with their anchor points
insertions = [
    {
        'anchor': '> **Key Concept:** Whisper uses weak supervision on 680,000 hours of multilingual audio to learn robust speech recognition across diverse conditions, languages, and tasks through a single encoder-decoder transformer architecture.\n\n## Visualization',
        'replacement': '''> **Key Concept:** Whisper uses weak supervision on 680,000 hours of multilingual audio to learn robust speech recognition across diverse conditions, languages, and tasks through a single encoder-decoder transformer architecture.

![Whisper Architecture](diagrams/whisper_architecture.png)

*Figure 1: Whisper's encoder-decoder transformer architecture. The encoder processes mel-spectrograms to extract acoustic features, while the decoder generates text autoregressively using special tokens to control the task (transcribe/translate), language, and timestamp behavior.*

## Visualization'''
    },
    {
        'anchor': 'print("- Mel-Spectrogram: Emphasizes perceptually-relevant frequencies (Whisper\'s input)")\n```\n\nThe visualization above shows the progression',
        'replacement': '''print("- Mel-Spectrogram: Emphasizes perceptually-relevant frequencies (Whisper's input)")
```

![Audio Representations](diagrams/audio_representations.png)

*Figure 2: Audio signal representations used in speech recognition. Top: Raw waveform showing amplitude over time. Middle: Spectrogram with linear frequency scale. Bottom: Mel-spectrogram with 80 frequency bins optimized for human perception—this is Whisper's input format.*

The visualization above shows the progression'''
    },
    {
        'anchor': '# Note: Speed is relative to \'large\' model\n# WER (Word Error Rate) varies by language and audio quality\n```\n\nWhisper comes in five model sizes',
        'replacement': '''# Note: Speed is relative to 'large' model
# WER (Word Error Rate) varies by language and audio quality
```

![Whisper Model Comparison](diagrams/model_comparison.png)

*Figure 3: Comparison of Whisper model variants. Top left: Parameter counts ranging from 39M (tiny) to 1.5B (large). Top right: Processing speed relative to large model. Bottom left: Word Error Rate ranges on clean audio. Bottom right: Speed-accuracy tradeoff showing the optimal balance—larger models are more accurate but slower.*

Whisper comes in five model sizes'''
    },
    {
        'anchor': '#   Child/teen speech (Whisper):  30-56%\n```\n\nWord Error Rate is the standard metric',
        'replacement': '''#   Child/teen speech (Whisper):  30-56%
```

![Word Error Rate Calculation](diagrams/wer_calculation.png)

*Figure 4: Word Error Rate (WER) calculation methodology. WER measures ASR accuracy by counting substitutions (S), insertions (I), and deletions (D) relative to the reference transcript. In this example, two substitutions ("quick"→"quik", "jumps"→"jump") result in WER = 22.22%. Lower WER indicates better performance, with human transcriptionists achieving ~4% and state-of-the-art models reaching 3-5% on clean speech.*

Word Error Rate is the standard metric'''
    },
    {
        'anchor': 'This makes Whisper particularly valuable for international applications where you need English text from multilingual audio sources.\n\n### Part 6: Evaluation with Word Error Rate',
        'replacement': '''This makes Whisper particularly valuable for international applications where you need English text from multilingual audio sources.

![Whisper ASR Pipeline](diagrams/pipeline_flow.png)

*Figure 5: Complete Whisper ASR pipeline from raw audio to text output. The pipeline includes preprocessing (resampling, normalization), feature extraction (mel-spectrogram), encoder processing, and autoregressive decoding with task-specific tokens. The diagram shows typical Real-Time Factors (RTF) for different model sizes and key capabilities like multilingual support and automatic language detection.*

### Part 6: Evaluation with Word Error Rate'''
    }
]

# Apply insertions
modified_content = content
for insertion in insertions:
    if insertion['anchor'] in modified_content:
        modified_content = modified_content.replace(insertion['anchor'], insertion['replacement'])
        print(f"✓ Inserted diagram at anchor: {insertion['anchor'][:60]}...")
    else:
        print(f"✗ Could not find anchor: {insertion['anchor'][:60]}...")

# Write the modified content
with open('../content.md', 'w') as f:
    f.write(modified_content)

print("\n✓ Successfully updated content.md with all diagram references!")
