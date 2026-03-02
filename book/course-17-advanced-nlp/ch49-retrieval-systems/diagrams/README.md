# Diagrams for Chapter 49: Advanced Retrieval Systems

This directory contains all generated diagrams for the chapter.

## Generated Diagrams

1. **bi_vs_cross_encoder.png** - Architectural comparison between bi-encoder and cross-encoder systems
   - Shows bi-encoder with separate query/document encoders
   - Shows cross-encoder with joint encoding
   - Highlights O(1) vs O(N) complexity tradeoff

2. **multi_stage_pipeline.png** - Multi-stage retrieval funnel architecture
   - Stage 1: BM25 sparse retrieval (1M → 1K docs)
   - Stage 2: Bi-encoder dense retrieval (1K → 100 docs)
   - Stage 3: Cross-encoder re-ranking (100 → 10 docs)
   - Shows latency and accuracy at each stage

3. **metrics_comparison.png** - Comparison of retrieval evaluation metrics
   - Recall@k for different k values
   - Mean Reciprocal Rank (MRR)
   - Normalized Discounted Cumulative Gain (nDCG@10)
   - Compares BM25, Bi-Encoder, Cross-Encoder, and Hybrid-RRF

4. **two_stage_analysis.png** - Latency vs accuracy tradeoff analysis
   - nDCG vs latency curve for different candidate set sizes
   - Per-query precision comparison
   - Shows diminishing returns of larger candidate sets

## Generation Scripts

The `generate_*.py` scripts can be re-run to regenerate any diagram:

```bash
python generate_bi_vs_cross_encoder.py
python generate_multi_stage_pipeline.py
python generate_metrics_comparison.py
python generate_two_stage_analysis.py
```

All diagrams follow the textbook's color palette:
- Blue (#2196F3) - Primary/Query
- Green (#4CAF50) - Document/Success
- Orange (#FF9800) - Intermediate
- Red (#F44336) - Score/Alert
- Purple (#9C27B0) - Combined/Special
- Gray (#607D8B) - Neutral

All figures are saved at 150 DPI with white backgrounds for print clarity.
