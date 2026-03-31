# Full-Stack ML Coursebook

A comprehensive Data Science & Machine Learning coursebook — from linear algebra to production systems and frontier research. 22 courses, 65+ chapters with runnable code, visualizations, and practice exercises.

Every chapter includes complete Python code examples (using scikit-learn, PyTorch, and real datasets), mathematical foundations, intuitive explanations, common pitfalls, and practice exercises with solutions.

---

## How to Use This Book

**Read online:** Click any `.ipynb` file — GitHub renders notebooks with code, outputs, and visualizations inline.

**Run locally:**

```bash
git clone https://github.com/cs-cmyk/full-stack-ml-coursebook.git
cd full-stack-ml-coursebook
pip install -r requirements.txt
jupyter lab
```

Each chapter is a standalone notebook.

---

## Curriculum

### Part I: Foundations

| Course | Chapters | Topics |
|--------|----------|--------|
| **Course 01: [Mathematical Foundations](book/course-01-foundations/)** | Ch 1–4 | Vectors, Derivatives, Sample spaces, Descriptive stats |
| **Course 02: [Programming for Data Science](book/course-02-programming/)** | Ch 5–9 | Data structures, Arrays, Matplotlib, SQL queries, Git fundamentals |
| **Course 03: [EDA & Feature Engineering](book/course-03-eda-features/)** | Ch 10–17 | Distributions, Missing values, Pearson, Scaling, Encoding methods, ... |

### Part II: Core Machine Learning

| Course | Chapters | Topics |
|--------|----------|--------|
| **Course 04: [Machine Learning](book/course-04-ml/)** | Ch 18–21 | OLS, Logistic regression, K-means, Cross-validation strategies |
| **Course 05: [Deep Learning](book/course-05-deep-learning/)** | Ch 22–26 | Perceptrons, Convolutions, LSTMs, Self-attention, VAEs |

### Part III: Domains & Applications

| Course | Chapters | Topics |
|--------|----------|--------|
| **Course 06: [Natural Language Processing](book/course-06-nlp/)** | Ch 27–28 | Tokenization, BERT |
| **Course 07: [Time Series & Forecasting](book/course-07-time-series/)** | Ch 29–30 | Stationarity, Deep forecasting |
| **Course 08: [Big Data & Distributed Computing](book/course-08-big-data/)** | Ch 31–32 | Hadoop, ETL/ELT patterns |

### Part IV: Production & Practice

| Course | Chapters | Topics |
|--------|----------|--------|
| **Course 09: [MLOps & Deployment](book/course-09-mlops/)** | Ch 33–34 | FastAPI, Experiment tracking |
| **Course 10: [Advanced Topics Survey](book/course-10-advanced/)** | Ch 35–39 | Collaborative filtering, MDPs, Image preprocessing, Graph representations, A/B testing |
| **Course 11: [Ethics, Interpretability & Communication](book/course-11-ethics-comms/)** | Ch 40–42 | Fairness metrics, SHAP, Data storytelling |

### Part V: Advanced & Expert

| Course | Chapters | Topics |
|--------|----------|--------|
| **Course 12: [Probabilistic Machine Learning](book/course-12-probabilistic-ml/)** | Ch 32–35 | Prior, Kernels, Bayes by Backprop, PyMC |
| **Course 13: [Advanced Optimization](book/course-13-optimization/)** | Ch 36–38 | Convex optimization, Learning rate schedules, Acquisition functions |
| **Course 14: [Advanced Deep Learning Architectures](book/course-14-advanced-dl/)** | Ch 39–41 | Flash attention, Diffusion models, CLIP |
| **Course 15: [Large Language Models](book/course-15-llms/)** | Ch 42–45 | Tokenization (BPE), SFT, RAG, Benchmarks (MMLU |
| **Course 16: [Advanced Computer Vision](book/course-16-advanced-cv/)** | Ch 46–47 | YOLO, 3D vision |
| **Course 17: [Advanced NLP & Information Retrieval](book/course-17-advanced-nlp/)** | Ch 48–50 | Knowledge graphs, Dense retrieval, Whisper |
| **Course 18: [Production ML Systems](book/course-18-production-ml/)** | Ch 51–53 | Feature stores, Data versioning (DVC, Data drift (PSI |
| **Course 19: [Causal Machine Learning](book/course-19-causal-ml/)** | Ch 54–56 | DAGs, ATE/CATE, Uplift modeling |
| **Course 20: [Geometric & Structured Learning](book/course-20-geometric-learning/)** | Ch 57–59 | GCN, Drug discovery, Point clouds |
| **Course 21: [Advanced Reinforcement Learning](book/course-21-advanced-rl/)** | Ch 60–62 | World models, Multi-agent RL, RLHF internals |
| **Course 22: [Research Methods & Frontier Topics](book/course-22-research-frontiers/)** | Ch 64–65 | Test-time compute, The alignment problem |

> **Note on chapter numbering:** Chapters are numbered sequentially within the full coursebook. Courses 10–11 (Part IV) and Courses 12–14 (Part V) cover different topics at different depths but share some chapter number ranges in their directory names. This reflects the coursebook's design: Part IV provides survey-level introductions, while Part V offers deep dives. Always navigate by course directory for clarity.

---

## Chapter Structure

Every chapter follows a consistent structure:

1. **Why This Matters** — Real-world motivation
2. **Intuition** — Plain-language explanation with concrete analogies
3. **Formal Definition** — Mathematical foundations
4. **Key Concept** — Core idea highlighted
5. **Visualization** — Diagrams and plots
6. **Examples** — Complete, runnable Python code with interleaved walkthroughs
7. **Common Pitfalls** — What beginners get wrong and why
8. **Practice** — Graded exercises (with solutions at the end)
9. **Key Takeaways** — Summary
10. **Next** — What comes next and how it connects

---

## Quick Topic Index

Looking for a specific topic? Here's where to find it:

| Topic | Chapter | Course |
|-------|---------|--------|
| KL Divergence & Cross-Entropy | Ch 22, 26, 34, 43 | Courses 5, 12, 15 |
| Feature Importance (SHAP, LIME, Permutation) | Ch 41 | Course 11 |
| Bayesian A/B Testing | Ch 35 | Course 12 |
| Data Drift & Concept Drift | Ch 53 | Course 18 |
| Simpson's Paradox & DAGs | Ch 54 | Course 19 |
| Mechanistic Interpretability | Ch 64 | Course 22 |
| LLM Evaluation & Red-Teaming | Ch 45 | Course 15 |
| Model Calibration | Ch 41 | Course 11 |
| RAG & Agents | Ch 44 | Course 15 |
| RLHF Internals | Ch 43, 62 | Courses 15, 21 |

---

## Requirements

Full list in `requirements.txt`. Python 3.10+ recommended.

---

## License

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International

Copyright (c) 2026 Chirag Shinde

You are free to:

- **Share:** copy and redistribute the material in any medium or format
- **Adapt:** remix, transform, and build upon the material

Under the following terms:

- **Attribution:** You must provide appropriate credit
- **Non-Commercial:** You may not use the material for commercial purposes
- **ShareAlike:** If you remix, you must distribute under the same license

Full license text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

---

## Contributing

Found an error? Have a suggestion? Open an issue or submit a pull request. Contributions welcome — especially corrections to code examples, improved explanations, and additional practice exercises.
