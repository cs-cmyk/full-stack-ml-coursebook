```mermaid
flowchart TB
    subgraph Input["Audio Input Processing"]
        A[Raw Audio Waveform<br/>16 kHz, 30-second chunks] --> B[Short-Time Fourier<br/>Transform STFT]
        B --> C[Mel-Frequency Binning<br/>80 mel bins]
        C --> D[Log-Mel Spectrogram<br/>80 × T' frames]
    end

    subgraph Encoder["Transformer Encoder"]
        D --> E1[Positional Encoding]
        E1 --> E2[Conv1D + GELU]
        E2 --> E3[Multi-Head<br/>Self-Attention × N]
        E3 --> E4[Feed-Forward<br/>Networks]
        E4 --> E5[Encoder Output<br/>H ∈ ℝ^T' × d]
    end

    subgraph TaskSpec["Task Specification"]
        T1[&lt;|startoftranscript|&gt;]
        T2[&lt;|en|&gt; / &lt;|es|&gt; / ...]
        T3[&lt;|transcribe|&gt; /<br/>&lt;|translate|&gt;]
        T4[&lt;|notimestamps|&gt; /<br/>timestamp tokens]
    end

    subgraph Decoder["Transformer Decoder"]
        E5 --> D1[Cross-Attention<br/>to Encoder]
        T1 & T2 & T3 & T4 --> D2[Task Conditioning]
        D2 --> D1
        D1 --> D3[Masked Self-Attention<br/>× N layers]
        D3 --> D4[Feed-Forward<br/>Networks]
        D4 --> D5[Output Projection<br/>Vocabulary]
    end

    subgraph Output["Text Generation"]
        D5 --> O1[Autoregressive<br/>Decoding]
        O1 --> O2[Beam Search /<br/>Greedy Sampling]
        O2 --> O3[Text Tokens<br/>y₁, y₂, ..., yₗ]
    end

    style Input fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style Encoder fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style TaskSpec fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style Decoder fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style Output fill:#FCE4EC,stroke:#F44336,stroke-width:2px
```
