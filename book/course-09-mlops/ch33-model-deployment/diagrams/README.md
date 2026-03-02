# Model Deployment Diagrams

This directory contains supplementary diagrams for Chapter 33: Model Deployment.

## Generated Diagrams

### 1. serialization_comparison.png
**Purpose:** Compares performance and file size metrics for Pickle, Joblib, and ONNX serialization formats.

**Metrics visualized:**
- Serialization (save) time in milliseconds
- Deserialization (load) time in milliseconds
- File size in bytes

**Suggested insertion point in content.md:** After line 261, following the text "The timing differences are negligible for small models, but for production systems that load models at startup, joblib's compression can save seconds or even minutes for ensemble models with hundreds of megabytes."

**Reference text:**
```markdown
![Serialization Format Comparison](diagrams/serialization_comparison.png)

**Figure 3: Serialization Format Comparison.** Performance and file size comparison between Pickle, Joblib, and ONNX serialization formats. Pickle offers the fastest save/load times but is Python-specific. Joblib provides better compression for NumPy arrays. ONNX enables cross-platform deployment at the cost of longer serialization time.
```

---

### 2. docker_layers.png
**Purpose:** Illustrates Docker multi-stage builds and layer caching strategy.

**Concepts visualized:**
- Multi-stage build architecture (Builder stage vs Runtime stage)
- Layer caching best practices (optimal vs poor ordering)
- Size reduction from multi-stage builds (450MB → 215MB)
- Rebuild time comparison (5 seconds vs 2+ minutes)

**Suggested insertion point in content.md:** After line 640, following the Docker containerization example and before Part 5 (Kubernetes deployment).

**Reference text:**
```markdown
![Docker Layer Architecture](diagrams/docker_layers.png)

**Figure 4: Docker Multi-Stage Build and Layer Caching.** Left panel shows multi-stage build architecture, separating build dependencies from runtime to reduce final image size from 450MB to 215MB. Right panel compares optimal layer ordering (requirements.txt → dependencies → code) enabling 5-second rebuilds versus poor ordering (code → dependencies) requiring 2+ minute reinstalls on every code change.
```

---

### 3. kubernetes_architecture.png
**Purpose:** Comprehensive Kubernetes deployment architecture showing all components and their interactions.

**Components visualized:**
- External users and load balancer
- Kubernetes Service with cluster IP
- Pod replicas with containers
- Horizontal Pod Autoscaler (HPA) monitoring
- Health probes (startup, liveness, readiness)
- Deployment configuration
- Traffic flow paths

**Suggested insertion point in content.md:** After line 851, following the Kubernetes deployment example and before the "Common Pitfalls" section.

**Reference text:**
```markdown
![Kubernetes Architecture](diagrams/kubernetes_architecture.png)

**Figure 5: Kubernetes Deployment Architecture.** Complete system architecture showing traffic flow from external users through the load balancer and service to three pod replicas. The Horizontal Pod Autoscaler monitors CPU usage and scales between 2-10 replicas based on 70% CPU target. Each pod includes health probes (startup, liveness, readiness) for monitoring and automatic recovery. Resource limits and requests ensure stable performance.
```

---

### 4. deployment_strategies.png
**Purpose:** Decision matrix comparing serverless, single VM, and Kubernetes deployment strategies.

**Strategies compared:**
- Serverless (AWS Lambda, Cloud Functions)
- Single VM (EC2, GCE, DigitalOcean)
- Kubernetes (EKS, GKE, AKS)

**Comparison dimensions:**
- Pros and cons for each approach
- Best use cases
- Traffic patterns
- Latency requirements
- Cost models
- Setup time
- Required expertise

**Suggested insertion point in content.md:** After line 15 in the Intuition section, following the text about choosing deployment strategies based on traffic patterns and requirements.

**Reference text:**
```markdown
![Deployment Strategy Comparison](diagrams/deployment_strategies.png)

**Figure 6: Deployment Strategy Decision Matrix.** Comprehensive comparison of three deployment approaches: serverless functions for sporadic traffic with pay-per-invocation pricing but cold-start latency; single VMs for simple, low-traffic applications with predictable costs but manual scaling; and Kubernetes for high-traffic production systems with auto-scaling and high availability but higher complexity and minimum costs.
```

---

## Integration Instructions

To integrate these diagrams into the chapter:

1. Copy the reference text from each section above
2. Insert at the suggested line numbers in content.md
3. Adjust figure numbers if needed to maintain sequential numbering
4. The existing mermaid diagrams (Figures 1 and 2) should remain unchanged

## Regeneration

To regenerate any diagram, run:
```bash
cd book/course-09-mlops/ch33-model-deployment/diagrams/
python <diagram_name>.py
```

All diagrams use the consistent color palette:
- Blue (#2196F3) - Primary, load balancers, services
- Green (#4CAF50) - Success, containers, services
- Orange (#FF9800) - Warnings, pods, builds
- Red (#F44336) - Errors, HPA, constraints
- Purple (#9C27B0) - Users, serverless
- Gray (#607D8B) - Infrastructure, base images
