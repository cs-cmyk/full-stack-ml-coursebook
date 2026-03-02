"""Generate forward/reverse diffusion process diagram using mermaid"""

mermaid_diagram = """
graph LR
    subgraph "Forward Diffusion: Data to Noise"
    X0["x₀<br/>(clean image)"] -->|"add noise<br/>β₁"| X1["x₁<br/>(slight noise)"]
    X1 -->|"add noise<br/>β₂"| X2["x₂<br/>(noisier)"]
    X2 -->|"..."| X3["..."]
    X3 -->|"add noise<br/>βₜ"| XT["xₜ<br/>(pure noise)"]
    end

    subgraph "Reverse Diffusion: Noise to Data"
    XT2["xₜ<br/>(pure noise)"] -->|"denoise<br/>p_θ"| X32["..."]
    X32 -->|"..."| X22["x₂<br/>(less noise)"]
    X22 -->|"denoise<br/>p_θ"| X12["x₁<br/>(clearer)"]
    X12 -->|"denoise<br/>p_θ"| X02["x₀<br/>(clean image)"]
    end

    style X0 fill:#2196F3,stroke:#1976D2,stroke-width:3px,color:#fff
    style X02 fill:#4CAF50,stroke:#388E3C,stroke-width:3px,color:#fff
    style XT fill:#F44336,stroke:#D32F2F,stroke-width:3px,color:#fff
    style XT2 fill:#F44336,stroke:#D32F2F,stroke-width:3px,color:#fff
    style X1 fill:#64B5F6,stroke:#1976D2,stroke-width:2px,color:#fff
    style X2 fill:#90CAF9,stroke:#1976D2,stroke-width:2px,color:#fff
    style X12 fill:#81C784,stroke:#388E3C,stroke-width:2px,color:#fff
    style X22 fill:#A5D6A7,stroke:#388E3C,stroke-width:2px,color:#fff
"""

# Save as markdown file for rendering
with open('/home/chirag/ds-book/book/course-14/ch40/diagrams/diffusion_process.md', 'w') as f:
    f.write("```mermaid\n")
    f.write(mermaid_diagram)
    f.write("\n```\n")

print("Mermaid diagram saved to diffusion_process.md")
print("This can be rendered using mermaid-cli or online tools")
