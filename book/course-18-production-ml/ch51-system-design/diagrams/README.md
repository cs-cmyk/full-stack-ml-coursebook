# Chapter 51 Diagrams

This directory contains all diagrams for Chapter 51: System Design for ML

## Generated Diagrams

### 1. ML System Design Patterns Comparison
**File:** `diagram1_system_patterns.png`  
**Type:** Matplotlib visualization  
**Description:** Comparison of three fundamental ML serving patterns:
- Batch Prediction (Offline) - Hours to days latency
- Real-Time Inference (Online) - 10-100ms latency  
- Streaming Inference - Seconds latency

### 2. Feature Store Architecture
**File:** `diagram2_feature_store.png`  
**Type:** Matplotlib visualization  
**Description:** Dual-path architecture showing:
- Data Sources (Batch and Streaming)
- Feature Computation pipelines
- Offline Store (historical features, point-in-time correctness)
- Online Store (low-latency serving)
- Training and Serving consumption paths

### 3. Model Serving Stack
**File:** `diagram3_model_serving.png`  
**Type:** Matplotlib visualization  
**Description:** Production serving infrastructure with:
- Client applications (Web, Mobile, Backend)
- Load Balancer/API Gateway
- Auto-scaled Model Server replicas
- Model Repository and Feature Store
- Observability stack (Logging, Metrics, Tracing)

## Technical Specifications

- **Resolution:** 150 DPI
- **Max Width:** 800px (adaptive based on content)
- **Format:** PNG with white background
- **Color Palette:** 
  - Blue (#2196F3) - Data sources/clients
  - Green (#4CAF50) - Processing/compute
  - Orange (#FF9800) - Storage/repositories
  - Red (#F44336) - Observability/monitoring
  - Purple (#9C27B0) - API/interfaces
  - Gray (#607D8B) - Metadata/registry

## Source Files

Each diagram has a corresponding Python script (`.py` file) that generates the visualization. 
To regenerate any diagram:

```bash
python diagram1_system_patterns.py
python diagram2_feature_store.py
python diagram3_model_serving.py
```

## Integration

All diagrams are referenced in `content.md` using standard markdown image syntax:
```markdown
![Diagram Title](diagrams/diagram_filename.png)
```
