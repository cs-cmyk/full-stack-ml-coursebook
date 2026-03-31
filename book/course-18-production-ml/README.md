# Course 18: Production ML Systems

Build, deploy, and maintain reliable ML systems at scale — system design, data engineering, drift detection, and observability.

## Prerequisites

Courses 8 (Big Data) and 9 (MLOps).

## Chapters

| Chapter | Title | Key Topics |
|---------|-------|------------|
| [51](ch51-system-design/) | System Design for ML | Feature stores, model serving (Triton, vLLM), batch vs real-time inference, edge deployment, cost modeling |
| [52](ch52-data-engineering/) | Data Engineering for ML | Data versioning (DVC, LakeFS), labeling pipelines, active learning, synthetic data, privacy-preserving ML |
| [53](ch53-monitoring-reliability/) | Monitoring and Reliability | Data drift (PSI, KS test), concept drift, prediction drift, shadow deployments, canary releases, incident response, ML observability |

## How to Use

Each chapter is available as:
- **content.md** — Read as markdown
- **content.ipynb** — Open as Jupyter notebook (runnable code + visualizations)

```bash
cd book/course-18-production-ml/
jupyter lab
```
