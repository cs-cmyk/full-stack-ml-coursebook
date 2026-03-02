# Diagram Integration Guide
## Chapter 33: Model Deployment

This guide provides step-by-step instructions for integrating the generated diagrams into content.md.

---

## Quick Start

**Option 1: Manual Integration (Recommended)**
1. Open `content.md` in your editor
2. Navigate to each insertion point listed below
3. Copy and paste the markdown text provided
4. Save and preview

**Option 2: Automated Integration (Requires Approval)**
The diagram agent can update content.md automatically if granted write permission.

---

## Integration Steps

### Step 1: Insert Serialization Comparison (Figure 3)

**Location:** After line 261
**Search for:** "The timing differences are negligible for small models"
**Insert after this paragraph:** The one ending with "...ensemble models with hundreds of megabytes."

**Markdown to insert:**
```markdown

![Serialization Format Comparison](diagrams/serialization_comparison.png)

**Figure 3: Serialization Format Comparison.** Performance and file size comparison between Pickle, Joblib, and ONNX serialization formats. Pickle offers the fastest save/load times but is Python-specific. Joblib provides better compression for NumPy arrays. ONNX enables cross-platform deployment at the cost of longer serialization time.
```

---

### Step 2: Insert Docker Layers (Figure 4)

**Location:** After line 639
**Search for:** "no 'works on my machine' problems"
**Insert after this paragraph:** The one ending with "...no 'works on my machine' problems."

**Markdown to insert:**
```markdown

![Docker Layer Architecture](diagrams/docker_layers.png)

**Figure 4: Docker Multi-Stage Build and Layer Caching.** Left panel shows multi-stage build architecture, separating build dependencies from runtime to reduce final image size from 450MB to 215MB. Right panel compares optimal layer ordering (requirements.txt → dependencies → code) enabling 5-second rebuilds versus poor ordering (code → dependencies) requiring 2+ minute reinstalls on every code change.
```

---

### Step 3: Insert Kubernetes Architecture (Figure 5)

**Location:** After line 851
**Search for:** "monitoring with Prometheus/Grafana"
**Insert after this paragraph:** The one ending with "...monitoring with Prometheus/Grafana."

**Markdown to insert:**
```markdown

![Kubernetes Architecture](diagrams/kubernetes_architecture.png)

**Figure 5: Kubernetes Deployment Architecture.** Complete system architecture showing traffic flow from external users through the load balancer and service to three pod replicas. The Horizontal Pod Autoscaler monitors CPU usage and scales between 2-10 replicas based on 70% CPU target. Each pod includes health probes (startup, liveness, readiness) for monitoring and automatic recovery. Resource limits and requests ensure stable performance.
```

---

### Step 4: Insert Deployment Strategies (Figure 6)

**Location:** After line 15
**Search for:** "context determines the right choice"
**Insert after this paragraph:** The one ending with "...context determines the right choice."

**Markdown to insert:**
```markdown

![Deployment Strategy Comparison](diagrams/deployment_strategies.png)

**Figure 6: Deployment Strategy Decision Matrix.** Comprehensive comparison of three deployment approaches: serverless functions for sporadic traffic with pay-per-invocation pricing but cold-start latency; single VMs for simple, low-traffic applications with predictable costs but manual scaling; and Kubernetes for high-traffic production systems with auto-scaling and high availability but higher complexity and minimum costs.
```

---

## Verification

After integration, verify:

1. **All images render:** Preview the markdown and ensure all 6 figures display
   - Figures 1-2: Existing mermaid diagrams
   - Figures 3-6: New matplotlib diagrams

2. **Paths are correct:** Image references should be `diagrams/<filename>.png`

3. **Figure numbers are sequential:** Figures 1, 2, 3, 4, 5, 6

4. **Captions are formatted correctly:** Bold "Figure X:" followed by descriptive text

5. **Spacing is consistent:** Blank line before and after each image

---

## Troubleshooting

### Images don't render
- **Problem:** Broken image links
- **Solution:** Verify the path is `diagrams/<filename>.png` relative to content.md

### Figure numbers are wrong
- **Problem:** Numbers don't follow 1, 2, 3, 4, 5, 6
- **Solution:** Renumber manually to maintain sequence

### Line numbers don't match
- **Problem:** Previous edits shifted line numbers
- **Solution:** Use the "Search for" text to find the correct location

### Images are too large/small
- **Problem:** Display size not appropriate
- **Solution:** Diagrams are already optimized. If needed, add width constraints:
  ```markdown
  <img src="diagrams/filename.png" alt="Description" width="600">
  ```

---

## Final Checklist

- [ ] Figure 3 inserted after line 261
- [ ] Figure 4 inserted after line 639
- [ ] Figure 5 inserted after line 851
- [ ] Figure 6 inserted after line 15
- [ ] All images render correctly
- [ ] Figure numbers are sequential (1-6)
- [ ] Captions are properly formatted
- [ ] No broken links
- [ ] Preview looks professional

---

## Regenerating Diagrams

If you need to modify or regenerate any diagram:

```bash
# Regenerate all
cd book/course-09-mlops/ch33-model-deployment/diagrams/
python generate_all.py

# Regenerate specific diagram
python serialization_comparison.py
python docker_layers.py
python kubernetes_architecture.py
python deployment_strategies.py

# Verify quality
python verify_diagrams.py
```

---

## Support Files

- **README.md** - Detailed descriptions of each diagram
- **SUMMARY.md** - Visual overview of all diagrams
- **content_updates.md** - Raw text for all insertions
- **DIAGRAM_COMPLETION_REPORT.md** - Full completion report
- **verify_diagrams.py** - Quality verification script
- **generate_all.py** - Batch regeneration script

---

**Last Updated:** 2026-03-01
**Status:** Ready for integration
