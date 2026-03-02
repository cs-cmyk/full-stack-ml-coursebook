# Data Pipeline Diagrams

This directory contains educational visualizations for Chapter 32: Data Pipelines.

## Generated Diagrams

### 1. ETL vs ELT Performance Comparison
**File:** `etl_vs_elt_performance.png`

**Purpose:** Compare ETL and ELT patterns across different dimensions

**Content:**
- Left chart: Processing time comparison by data volume (Small to Very Large)
- Right chart: Suitability scores for different selection criteria

**Suggested placement in content.md:**
After Figure 1 (around line 71), add:

```markdown
![ETL vs ELT Performance Comparison](diagrams/etl_vs_elt_performance.png)

**Figure 1.1:** Performance comparison between ETL and ELT patterns. Left: Processing time varies by data volume, with ELT becoming more efficient at scale due to cloud warehouse compute power. Right: Selection criteria showing when each pattern is preferable—ETL excels with high compliance needs and complex transformations, while ELT wins with large volumes, powerful warehouse compute, and need for flexibility.
```

---

### 2. Task Execution Timeline
**File:** `task_execution_timeline.png`

**Purpose:** Visualize Airflow task execution, parallelism, and retry behavior

**Content:**
- Top chart: Normal execution showing parallel task execution
- Bottom chart: Execution with retry logic showing failure and recovery

**Suggested placement in content.md:**
After Figure 3 (around line 121), add:

```markdown
![Airflow Task Execution Timeline](diagrams/task_execution_timeline.png)

**Figure 3.1:** Airflow task execution timelines showing real-world patterns. Top: Parallel execution where transform_sales and transform_customers run simultaneously, reducing total pipeline time from 45 to 32 minutes. Bottom: Retry behavior with automatic recovery—after two failures with 5-minute delays, the load task succeeds on the third attempt, adding only 8 minutes to total runtime instead of requiring manual intervention.
```

---

### 3. Data Quality Metrics Dashboard
**File:** `data_quality_metrics.png`

**Purpose:** Comprehensive data quality monitoring visualization

**Content:**
- Quality check pass/fail rates by type
- Data volume trends with anomaly detection
- Null rates by column with thresholds
- Pipeline success rate over time
- Quality score breakdown by dimension
- Overall quality summary with SLA status

**Suggested placement in content.md:**
After Example 5 (around line 1169, in the Common Pitfalls or Practice Exercises section), add:

```markdown
![Data Quality Metrics Dashboard](diagrams/data_quality_metrics.png)

**Figure 4:** Production data quality dashboard showing comprehensive monitoring. Top row: Quality check results over 30 days and daily volume trends with anomaly detection. Middle row: Null rate analysis by column and weekly pipeline success rates. Bottom row: Multi-dimensional quality scores and overall SLA compliance. This dashboard enables data teams to detect issues early—notice the volume anomaly (red X markers), the location column exceeding 3% null threshold (orange), and consistent success rates above 95% SLA target.
```

---

### 4. Backfill Execution Pattern
**File:** `backfill_execution.png`

**Purpose:** Illustrate sequential vs parallel backfill strategies

**Content:**
- Top chart: Sequential backfill with max_active_runs=1
- Bottom chart: Parallel backfill with max_active_runs=3
- Shows speedup and resource trade-offs

**Suggested placement in content.md:**
After the incremental loading example discussion (around line 806), add:

```markdown
![Backfill Execution Patterns](diagrams/backfill_execution.png)

**Figure 5:** Backfill execution strategies for processing historical dates. Top: Sequential processing (max_active_runs=1) executes one date at a time, taking 70 minutes for 7 dates but minimizing load on source systems—ideal for fragile APIs or databases. Bottom: Parallel processing (max_active_runs=3) runs multiple dates simultaneously in batches, completing in 30 minutes (2.3x speedup) but requiring more compute resources and potentially overwhelming source systems. The trade-off: speed versus system load.
```

---

## Color Palette

All diagrams use a consistent color scheme:
- **Blue (#2196F3)**: Extract/Load operations, primary data
- **Green (#4CAF50)**: Transform operations, success states
- **Orange (#FF9800)**: Validation, warnings, retries
- **Red (#F44336)**: Failures, errors, critical issues
- **Purple (#9C27B0)**: Load operations (when distinguished from extract)
- **Gray (#607D8B)**: Metadata, delays, infrastructure

## Technical Specifications

- **Resolution:** 150 DPI
- **Max width:** ~800px for most charts
- **Background:** White
- **Font size:** Minimum 10pt for readability
- **Format:** PNG with tight bounding box

## Usage Notes

These diagrams complement the existing mermaid flowcharts by providing:
1. Quantitative comparisons (performance, success rates)
2. Timeline visualizations (execution order, parallelism)
3. Monitoring dashboards (quality metrics, trends)
4. Trade-off analysis (backfill strategies)

The mermaid diagrams remain excellent for architecture and dependency visualization, while these matplotlib charts excel at showing data-driven insights and operational patterns.
