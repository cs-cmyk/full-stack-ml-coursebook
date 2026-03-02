# Chapter 27 Classical NLP - Diagram Generation Summary

## Overview
Generated 5 educational diagrams for the Classical NLP chapter using matplotlib and seaborn.

## Generated Diagrams

### 1. Text Preprocessing Pipeline
**File:** `preprocessing_pipeline.png`
**Type:** Flowchart visualization
**Size:** 77 KB
**Description:** Illustrates the 6-step text preprocessing pipeline from raw text through tokenization, normalization, stop word removal, and stemming/lemmatization.

**Referenced at:** Line 111 of content.md

---

### 2. Bag-of-Words Document-Term Matrix
**File:** `bow_matrix_heatmap.png`
**Type:** Heatmap
**Size:** 60 KB
**Description:** Shows the document-term matrix as a heatmap with word counts per document, demonstrating sparsity patterns in text representation.

**Referenced at:** Line 330 of content.md

---

### 3. BoW vs TF-IDF Comparison
**File:** `bow_vs_tfidf.png`
**Type:** Side-by-side bar charts
**Size:** 43 KB
**Description:** Compares raw word counts (BoW) against TF-IDF weighted values for the same document, showing how IDF downweights common words.

**Referenced at:** Line 490 of content.md

---

### 4. N-gram Extraction
**File:** `ngram_extraction.png`
**Type:** Multi-panel visualization
**Size:** 67 KB
**Description:** Displays unigrams, bigrams, and trigrams extracted from sample text with frequency counts, illustrating how n-grams capture word sequences.

**Referenced at:** Line 717 of content.md

---

### 5. Document Similarity Comparison
**File:** `similarity_comparison.png`
**Type:** Matrix comparison with bar chart
**Size:** 32 KB
**Description:** Compares cosine similarity matrices using BoW and TF-IDF representations, showing how different vectorization methods affect similarity measurements.

**Referenced at:** Line 1872 of content.md

---

## Technical Specifications

All diagrams follow the textbook standards:
- **Resolution:** 150 DPI
- **Max Width:** 800px (actual range: 718-792px)
- **Background:** White
- **Color Palette:**
  - Blue: #2196F3
  - Green: #4CAF50
  - Orange: #FF9800
  - Red: #F44336
  - Purple: #9C27B0
  - Gray: #607D8B
- **Font Size:** Minimum 12pt for readability
- **Layout:** All use `plt.tight_layout()` for optimal spacing

## Files Created

1. `generate_all_diagrams.py` - Main script to regenerate all diagrams
2. `preprocessing_pipeline.png` - Diagram 1
3. `bow_matrix_heatmap.png` - Diagram 2
4. `bow_vs_tfidf.png` - Diagram 3
5. `ngram_extraction.png` - Diagram 4
6. `similarity_comparison.png` - Diagram 5
7. `update_image_refs.py` - Script to update content.md image references
8. `README.md` - Usage documentation
9. `DIAGRAM_SUMMARY.md` - This file

## Content.md Updates

Updated content.md to include proper markdown image references for all diagrams:
- Updated line 111: Added `diagrams/` path prefix
- Added line 330: New image reference for BoW matrix heatmap
- Added line 490: New image reference for BoW vs TF-IDF comparison
- Added line 717: New image reference for n-gram extraction
- Added line 1872: New image reference for similarity comparison

All image references now use the format: `![Description](diagrams/filename.png)`

## Regenerating Diagrams

To regenerate all diagrams:
```bash
cd /home/chirag/ds-book/book/course-06-nlp/ch27-classical-nlp/diagrams
python generate_all_diagrams.py
```

## Dependencies

Required Python packages:
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

Install with:
```bash
pip install matplotlib seaborn numpy pandas scikit-learn
```

---

**Generated:** 2026-03-01
**Agent:** Diagram Agent for Data Science Textbook
**Chapter:** 27 - Classical NLP
