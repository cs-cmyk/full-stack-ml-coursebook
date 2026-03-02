# Content Updates for Chapter 33

This document contains the exact text additions to integrate the generated diagrams into content.md.

## Update 1: After Line 261 (Serialization Section)

**Location:** After the paragraph ending with "...ensemble models with hundreds of megabytes."

**Add:**
```markdown

![Serialization Format Comparison](diagrams/serialization_comparison.png)

**Figure 3: Serialization Format Comparison.** Performance and file size comparison between Pickle, Joblib, and ONNX serialization formats. Pickle offers the fastest save/load times but is Python-specific. Joblib provides better compression for NumPy arrays. ONNX enables cross-platform deployment at the cost of longer serialization time.
```

---

## Update 2: After Line 640 (Docker Section)

**Location:** After the paragraph ending with "...no Python installation, no dependency management, no 'works on my machine' problems."

**Add:**
```markdown

![Docker Layer Architecture](diagrams/docker_layers.png)

**Figure 4: Docker Multi-Stage Build and Layer Caching.** Left panel shows multi-stage build architecture, separating build dependencies from runtime to reduce final image size from 450MB to 215MB. Right panel compares optimal layer ordering (requirements.txt → dependencies → code) enabling 5-second rebuilds versus poor ordering (code → dependencies) requiring 2+ minute reinstalls on every code change.
```

---

## Update 3: After Line 851 (Kubernetes Section)

**Location:** After the paragraph ending with "...monitoring with Prometheus/Grafana."

**Add:**
```markdown

![Kubernetes Architecture](diagrams/kubernetes_architecture.png)

**Figure 5: Kubernetes Deployment Architecture.** Complete system architecture showing traffic flow from external users through the load balancer and service to three pod replicas. The Horizontal Pod Autoscaler monitors CPU usage and scales between 2-10 replicas based on 70% CPU target. Each pod includes health probes (startup, liveness, readiness) for monitoring and automatic recovery. Resource limits and requests ensure stable performance.
```

---

## Update 4: After Line 15 (Intuition Section - Deployment Choice)

**Location:** After the paragraph ending with "...context determines the right choice."

**Add:**
```markdown

![Deployment Strategy Comparison](diagrams/deployment_strategies.png)

**Figure 6: Deployment Strategy Decision Matrix.** Comprehensive comparison of three deployment approaches: serverless functions for sporadic traffic with pay-per-invocation pricing but cold-start latency; single VMs for simple, low-traffic applications with predictable costs but manual scaling; and Kubernetes for high-traffic production systems with auto-scaling and high availability but higher complexity and minimum costs.
```

---

## Verification

After making these changes:
1. Verify all figure numbers are sequential (Figures 1-6)
2. Check that image paths are correct: `diagrams/<filename>.png`
3. Ensure figure captions end with periods
4. Test that all images render correctly when viewing the markdown

## Quick Apply Script

To apply all updates automatically, you can use:

```bash
# This would need to be adapted to your markdown processor
# The exact line numbers may shift as content is added
```
