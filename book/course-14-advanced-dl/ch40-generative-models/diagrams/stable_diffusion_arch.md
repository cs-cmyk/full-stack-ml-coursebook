# Stable Diffusion Architecture

```mermaid

flowchart TB
    subgraph Input["Input Processing"]
        A["Text Prompt:<br/>'A cat wearing a spacesuit'"] --> B["CLIP Tokenizer"]
        B --> C["77 tokens"]
        C --> D["CLIP Text Encoder<br/>(frozen)"]
        D --> E["Text Embeddings<br/>77 × 768"]
    end

    subgraph Generation["Iterative Generation"]
        F["Random Noise<br/>Latent: 4 × 64 × 64"] --> G["U-Net Denoiser<br/>(with cross-attention)"]
        E -.->|"conditioning"| G
        G --> H["Denoised Latent<br/>4 × 64 × 64"]
        H -.->|"repeat 50-100 steps"| G
    end

    subgraph Output["Output Processing"]
        H --> I["VAE Decoder"]
        I --> J["RGB Image<br/>3 × 512 × 512"]
    end

    Input --> Generation
    Generation --> Output

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:3px
    style E fill:#E8F5E9,stroke:#4CAF50,stroke-width:3px
    style F fill:#FFF3E0,stroke:#FF9800,stroke-width:3px
    style G fill:#F3E5F5,stroke:#9C27B0,stroke-width:3px
    style H fill:#E0F2F1,stroke:#009688,stroke-width:3px
    style J fill:#FCE4EC,stroke:#E91E63,stroke-width:3px

    classDef processBox fill:#f9f9f9,stroke:#607D8B,stroke-width:2px
    class B,C,D,I processBox

```

## Key Components

1. **CLIP Text Encoder** (frozen): Converts text to semantic embeddings
2. **U-Net with Cross-Attention** (learned): Performs diffusion in latent space
3. **VAE Decoder** (learned): Upsamples latent to full-resolution image

**Innovation**: 48× compression by working in latent space instead of pixel space
