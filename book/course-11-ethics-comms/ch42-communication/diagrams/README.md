# Chapter 42 Communication Diagrams

This directory contains educational visualizations for the Data Science Communication chapter.

## Generated Diagrams

### 1. Stakeholder Pyramid (`stakeholder_pyramid.png`)
- **Type:** Matplotlib-based hierarchical diagram
- **Purpose:** Shows how information needs vary by organizational level
- **Features:**
  - Four organizational levels (Executives, Managers, Domain Experts, End Users)
  - Color-coded sections using textbook palette
  - Clear annotations showing decision-making flow
  - Annotations for increasing technical detail
- **Dimensions:** 800x640px @ 150 DPI
- **File size:** 128 KB

### 2. Inverted Pyramid Report Structure (`inverted_pyramid.png`)
- **Type:** Matplotlib-based pyramid diagram  
- **Purpose:** Illustrates the ideal structure for data science reports
- **Features:**
  - Three layers (Executive Summary, Methodology & Results, Technical Appendix)
  - Varying border widths to show importance hierarchy
  - Audience indicators on the left side
  - Reading depth annotations
- **Dimensions:** 800x720px @ 150 DPI
- **File size:** 173 KB

## Design Principles Applied

All diagrams follow the textbook's visual standards:
- ✅ Consistent color palette: #2196F3 (blue), #4CAF50 (green), #FF9800 (orange), #F44336 (red), #9C27B0 (purple)
- ✅ White backgrounds for printability
- ✅ Minimum 12pt font sizes for readability
- ✅ Clear labels and annotations
- ✅ 150 DPI resolution
- ✅ Maximum 800px width
- ✅ `plt.tight_layout()` applied before saving

## Source Files

The diagrams can be regenerated using:
```bash
python stakeholder_pyramid.py
python inverted_pyramid.py
```

## Notes

The content.md file contains embedded mermaid diagrams that will render in markdown viewers. The PNG versions in this directory provide:
- Maximum compatibility across all platforms
- Print-ready quality
- Consistent visual styling with the rest of the textbook
