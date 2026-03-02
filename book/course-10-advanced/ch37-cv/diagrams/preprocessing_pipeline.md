# Image Preprocessing and Augmentation Pipeline

## Pipeline Architecture (Mermaid Diagram)

```mermaid
flowchart TB
    Start([Raw Image<br/>PIL/numpy array]) --> Split{Training or<br/>Test/Val?}

    Split -->|Training| Aug[Data Augmentation]
    Split -->|Test/Val| Prep[Preprocessing Only]

    Aug --> Aug1[Random Horizontal Flip<br/>p=0.5]
    Aug1 --> Aug2[Random Rotation<br/>±15°]
    Aug2 --> Aug3[Random Crop<br/>with padding]
    Aug3 --> Aug4[Color Jitter<br/>brightness/contrast/saturation]
    Aug4 --> Aug5[Random Erasing<br/>optional]
    Aug5 --> Prep

    Prep --> Prep1[Resize to<br/>Fixed Dimensions]
    Prep1 --> Prep2[Convert to Tensor<br/>HWC → CHW]
    Prep2 --> Prep3{Normalization<br/>Strategy?}

    Prep3 -->|Min-Max| Norm1[Scale to [0, 1]<br/>X/255]
    Prep3 -->|Standardize| Norm2[μ=0, σ=1<br/>X - μ / σ]
    Prep3 -->|ImageNet| Norm3[ImageNet Stats<br/>μ=[0.485,0.456,0.406]<br/>σ=[0.229,0.224,0.225]]

    Norm1 --> Model[Model Input<br/>Tensor]
    Norm2 --> Model
    Norm3 --> Model

    Model --> Output([Predictions])

    style Start fill:#2196F3,stroke:#1976D2,color:#fff
    style Aug fill:#4CAF50,stroke:#388E3C,color:#fff
    style Prep fill:#FF9800,stroke:#F57C00,color:#fff
    style Model fill:#9C27B0,stroke:#7B1FA2,color:#fff
    style Output fill:#F44336,stroke:#D32F2F,color:#fff
    style Split fill:#607D8B,stroke:#455A64,color:#fff
    style Prep3 fill:#607D8B,stroke:#455A64,color:#fff
```

## Decision Tree: Which Augmentations to Use?

```mermaid
flowchart TD
    Start([Select Augmentations]) --> Q1{Task Type?}

    Q1 -->|Natural Images<br/>Dogs, Cats, Objects| Natural
    Q1 -->|Digits/Text| Digits
    Q1 -->|Medical Images| Medical

    Natural --> N1[✅ Horizontal Flip]
    Natural --> N2[✅ Rotation ±15°]
    Natural --> N3[✅ Color Jitter]
    Natural --> N4[✅ Random Crop]
    Natural --> N5[✅ Random Erasing]

    Digits --> D1[❌ Horizontal Flip<br/>Changes meaning]
    Digits --> D2[⚠️ Small Rotation<br/>±10° only]
    Digits --> D3[⚠️ Limited Color<br/>Brightness only]
    Digits --> D4[✅ Random Crop]

    Medical --> M1[⚠️ Careful with Flip<br/>Anatomy matters]
    Medical --> M2[✅ Rotation ±15°]
    Medical --> M3[❌ Color Jitter<br/>Diagnostic colors]
    Medical --> M4[✅ Random Crop]
    Medical --> M5[✅ Elastic Deformation]

    N1 & N2 & N3 & N4 & N5 --> Result1[High Augmentation<br/>Strong Regularization]
    D1 & D2 & D3 & D4 --> Result2[Conservative<br/>Augmentation]
    M1 & M2 & M3 & M4 & M5 --> Result3[Domain-Specific<br/>Augmentation]

    style Start fill:#2196F3,stroke:#1976D2,color:#fff
    style Q1 fill:#607D8B,stroke:#455A64,color:#fff
    style Natural fill:#4CAF50,stroke:#388E3C,color:#fff
    style Digits fill:#FF9800,stroke:#F57C00,color:#fff
    style Medical fill:#9C27B0,stroke:#7B1FA2,color:#fff
    style Result1 fill:#4CAF50,stroke:#388E3C,color:#fff
    style Result2 fill:#FF9800,stroke:#F57C00,color:#fff
    style Result3 fill:#9C27B0,stroke:#7B1FA2,color:#fff
```

## Training vs Inference Pipeline

```mermaid
flowchart LR
    subgraph Training["Training Pipeline"]
        T1[Raw Image] --> T2[Augmentation<br/>Stochastic]
        T2 --> T3[Preprocessing<br/>Deterministic]
        T3 --> T4[Model]
        T4 --> T5[Loss Calculation]
        T5 --> T6[Backpropagation]
    end

    subgraph Inference["Inference Pipeline"]
        I1[Raw Image] --> I2[Preprocessing<br/>Only]
        I2 --> I3[Model]
        I3 --> I4[Predictions]
    end

    Note1["⚠️ Key Difference:<br/>Augmentation ONLY<br/>during training"]

    style Training fill:#4CAF50,stroke:#388E3C,color:#fff,stroke-width:3px
    style Inference fill:#2196F3,stroke:#1976D2,color:#fff,stroke-width:3px
    style Note1 fill:#F44336,stroke:#D32F2F,color:#fff
```

## Normalization Impact on Model Performance

```mermaid
flowchart TB
    Input[Raw Image<br/>Pixel values: 0-255<br/>Wide range] --> Decision{Apply<br/>Normalization?}

    Decision -->|No| Bad[❌ Poor Training<br/>• Slow convergence<br/>• Unstable gradients<br/>• Lower accuracy]

    Decision -->|Yes| Good[✅ Normalized Input<br/>Values: ~[-3, 3]<br/>Consistent scale]

    Good --> Benefits[Benefits:<br/>• Faster convergence<br/>• Stable gradients<br/>• Better accuracy<br/>• Easier optimization]

    Bad --> Fix[Solution:<br/>Apply normalization!]
    Fix --> Good

    style Input fill:#607D8B,stroke:#455A64,color:#fff
    style Decision fill:#FF9800,stroke:#F57C00,color:#fff
    style Bad fill:#F44336,stroke:#D32F2F,color:#fff
    style Good fill:#4CAF50,stroke:#388E3C,color:#fff
    style Benefits fill:#2196F3,stroke:#1976D2,color:#fff
    style Fix fill:#9C27B0,stroke:#7B1FA2,color:#fff
```

---

## Notes

These mermaid diagrams can be rendered in markdown viewers that support mermaid (GitHub, GitLab, VS Code with extension, etc.).

For inclusion in the textbook:
1. Render to SVG using mermaid CLI: `mmdc -i diagram.md -o diagram.svg`
2. Or use online renderer: https://mermaid.live/
3. Export as PNG/SVG for embedding in the chapter

The diagrams use the standard color palette:
- Blue (#2196F3): Primary concepts
- Green (#4CAF50): Positive/recommended actions
- Orange (#FF9800): Caution/moderate risk
- Red (#F44336): Warnings/problems
- Purple (#9C27B0): Advanced/specialized
- Gray (#607D8B): Neutral/decision points
