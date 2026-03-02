# Production ML Systems

Build, deploy, and maintain reliable machine learning systems at scale. System design patterns, data engineering practices, and monitoring strategies to go from research prototypes from production systems handling millions of predictions.

## Prerequisites

Courses 8 (Big Data & Data Engineering Fundamentals) and 9 (MLOps & Production Machine Learning) or experience with distributed systems, model deployment, and ML pipelines.

## Chapters

| Chapter | Title | Key Topics |
|---------|-------|------------|
| ch51 | System Design for ML | Batch vs. real-time inference, feature stores, model serving architectures, GPU optimization, edge deployment, cost modeling |
| ch52 | Data Engineering for ML | Data versioning (DVC, LakeFS), feature pipelines (Spark, Flink), data quality at scale, labeling pipelines, active learning, synthetic data, privacy-preserving techniques |
| ch53 | Monitoring and Reliability | Data drift detection (PSI, KS test, Wasserstein), concept drift, shadow deployments, canary releases, online evaluation, incident response, ML observability platforms |

## How to Use

Each chapter is available as:
- **content.md** — Read as markdown
- **content.ipynb** — Open as Jupyter notebook (runnable code + visualizations)

```bash
cd book/course-18/
jupyter lab
```
