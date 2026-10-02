"""
> [!AML-DOC-FILE]
@file       catalog/research_specs.py
@description Catalog entries for the research-frontier layers in `nnarch.layers`,
             each carrying the paper it implements.
@module     nnarch.catalog.research_specs
@exports    build_research_specs
@created    2026-09-30
@context    Same declarative shape as `keras_specs.py` [amm: B.1, E.2]: a row names
            the class, its category and its tensor ranks, and the parameters are
            derived from the constructor signature. Layer ids are persisted in saved
            projects, so they are never renamed.
"""

from __future__ import annotations

from typing import Any, Sequence

from .introspect import derive_params
from .spec import CallStyle, Category, LayerSpec, Modality, PortSpec

_SEQ = [Modality.SEQUENCE, Modality.TEXT, Modality.AUDIO]
_IMAGE = [Modality.IMAGE]
_GRAPH = [Modality.GRAPH]
_ANY = [Modality.ANY]

_GRAPH_PORTS = [
    PortSpec(name="nodes", label="Node features", rank=3),
    PortSpec(name="adjacency", label="Adjacency", rank=3),
]


def _spec(
    module: str,
    name: str,
    label: str,
    category: Category,
    paper: str,
    description: str,
    *,
    rank_in: int | None = None,
    rank_out: int | None = None,
    modalities: Sequence[Modality] = _ANY,
    tags: Sequence[str] = (),
    starters: dict[str, Any] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    min_inputs: int = 1,
    max_inputs: int | None = 1,
    inputs: Sequence[PortSpec] | None = None,
    call_style: CallStyle = CallStyle.SINGLE,
    doc_url: str | None = None,
) -> LayerSpec:
    """
    > [!AML-DOC-UNIT]
    Build one LayerSpec for a research layer.
    @param module      submodule of `nnarch.layers` holding the class
    @param name        class name
    @param label       palette display name
    @param category    palette grouping
    @param paper       citation, shown in the property panel
    @param description palette tooltip
    @param rank_in     required input rank including batch, or None when agnostic
    @param rank_out    produced rank, or None when derived
    @param modalities  data kinds the layer suits
    @param tags        extra search keywords
    @param starters    opening values for mandatory constructor arguments, so a
                       dropped node is valid on arrival, as in `keras_specs`
    @param overrides   per-parameter field overrides, for names whose meaning differs
                       from the shared hint table
    @param min_inputs  minimum connected inputs
    @param max_inputs  maximum connected inputs
    @param inputs      explicit ordered ports, for the two-tensor graph layers
    @param call_style  tensor-passing convention [amm: E.4]
    @param doc_url     link to the paper
    @returns the assembled LayerSpec
    @raises ImportError when the class cannot be resolved, which means the layer
            module and this table have drifted apart
    """
    path = f"nnarch.layers.{module}.{name}"
    merged: dict[str, dict[str, Any] | None] = {
        param: {"default": value} for param, value in (starters or {}).items()
    }
    for param, fields in (overrides or {}).items():
        merged[param] = {**(merged.get(param) or {}), **fields}
    return LayerSpec(
        id=f"research.{name}",
        label=label,
        category=category,
        keras_path=path,
        params=derive_params(path, rank_in=rank_in, overrides=merged),
        inputs=list(inputs)
        if inputs is not None
        else [PortSpec(name="input", label="Input", rank=rank_in)],
        outputs=[PortSpec(name="output", label="Output", rank=rank_out)],
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        rank_in=rank_in,
        rank_out=rank_out,
        modalities=list(modalities),
        paper=paper,
        doc_url=doc_url,
        description=description,
        tags=[name.lower(), *tags],
        call_style=call_style,
        is_research=True,
    )


def build_research_specs() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Assemble the LayerSpec list for every research layer.
    @returns flat list across the research categories
    @raises ImportError when a declared class is missing from `nnarch.layers`
    @sideEffects imports the layer modules, and through them keras
    """
    specs: list[LayerSpec] = []

    # --- Gated feed-forward -------------------------------------------------
    for name, label, tags in (
        ("GLUFeedForward", "GLU Feed-Forward", ["glu", "gated", "mlp"]),
        ("SwiGLU", "SwiGLU", ["llama", "mistral", "silu", "gated"]),
        ("GeGLU", "GeGLU", ["palm", "t5", "gelu", "gated"]),
    ):
        specs.append(
            _spec(
                "ffn", name, label, Category.RESEARCH_FFN,
                "Shazeer 2020, arXiv:2002.05202",
                "Gated feed-forward block: two parallel projections, one gating the "
                "other. The transformer MLP used by most modern LLMs.",
                rank_in=None, modalities=_SEQ, tags=tags,
                starters={"hidden_dim": 256},
                doc_url="https://arxiv.org/abs/2002.05202",
            )
        )

    # --- Modern convolution -------------------------------------------------
    specs += [
        _spec("conv_mod", "DropPath", "Drop Path", Category.RESEARCH_CONV,
              "Huang et al. 2016, arXiv:1603.09382",
              "Stochastic depth: drops a whole residual branch for a sample, so the "
              "network trains as an ensemble of shallower ones.",
              modalities=_ANY, tags=["stochastic depth", "regularize", "convnext"],
              starters={"rate": 0.1},
              doc_url="https://arxiv.org/abs/1603.09382"),
        _spec("conv_mod", "SqueezeExcite", "Squeeze-and-Excite", Category.RESEARCH_CONV,
              "Hu et al. 2017, arXiv:1709.01507",
              "Channel attention: pools each feature map to one number and uses it to "
              "rescale the channels.",
              rank_in=4, rank_out=4, modalities=_IMAGE,
              tags=["se", "channel attention", "efficientnet"],
              doc_url="https://arxiv.org/abs/1709.01507"),
        _spec("conv_mod", "ConvNeXtBlock", "ConvNeXt Block", Category.RESEARCH_CONV,
              "Liu et al. 2022, arXiv:2201.03545",
              "A convnet block rebuilt with transformer habits: 7x7 depthwise mixing, "
              "layer norm, inverted bottleneck. Matches ViTs without attention.",
              rank_in=4, rank_out=4, modalities=_IMAGE,
              tags=["convnext", "residual", "depthwise", "modern cnn"],
              doc_url="https://arxiv.org/abs/2201.03545"),
        _spec("conv_mod", "MBConv", "MBConv (Inverted Residual)", Category.RESEARCH_CONV,
              "Sandler et al. 2018, arXiv:1801.04381; Tan & Le 2019, arXiv:1905.11946",
              "Expand, convolve depthwise, recalibrate channels, project back. The "
              "block behind MobileNetV2 and EfficientNet.",
              rank_in=4, rank_out=4, modalities=_IMAGE,
              tags=["mobilenet", "efficientnet", "inverted residual", "efficient"],
              starters={"filters": 32},
              doc_url="https://arxiv.org/abs/1905.11946"),
        _spec("conv_mod", "Involution", "Involution", Category.RESEARCH_CONV,
              "Li et al. 2021, arXiv:2103.06255",
              "Convolution inverted: the kernel is generated from the pixel itself, so "
              "it varies across space and is shared across channels.",
              rank_in=4, rank_out=4, modalities=_IMAGE,
              tags=["involution", "dynamic kernel", "efficient"],
              doc_url="https://arxiv.org/abs/2103.06255"),
    ]

    # --- Token mixing -------------------------------------------------------
    specs += [
        _spec("mixer", "PatchEmbedding", "Patch Embedding", Category.RESEARCH_MIXER,
              "Dosovitskiy et al. 2020, arXiv:2010.11929",
              "Cuts an image into square patches and projects each into a vector. The "
              "input stage of a vision transformer.",
              rank_in=4, rank_out=3, modalities=_IMAGE,
              tags=["vit", "vision transformer", "patches", "tokenize"],
              starters={"patch_size": 16, "embed_dim": 256},
              doc_url="https://arxiv.org/abs/2010.11929"),
        _spec("mixer", "ClassToken", "Class Token", Category.RESEARCH_MIXER,
              "Dosovitskiy et al. 2020, arXiv:2010.11929",
              "Prepends a learned token whose final state summarises the whole "
              "sequence, instead of pooling at the end.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["cls", "vit", "bert", "pooling"],
              doc_url="https://arxiv.org/abs/2010.11929"),
        _spec("mixer", "AddPositionEmbedding", "Position Embedding", Category.RESEARCH_MIXER,
              "Dosovitskiy et al. 2020, arXiv:2010.11929",
              "Adds a learned vector per position, so order carries information. "
              "Attention is permutation-invariant without it.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["position", "learned", "vit", "transformer"],
              doc_url="https://arxiv.org/abs/2010.11929"),
        _spec("mixer", "MLPMixerBlock", "MLP-Mixer Block", Category.RESEARCH_MIXER,
              "Tolstikhin et al. 2021, arXiv:2105.01601",
              "Mixes tokens and channels with two MLPs and no attention at all.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["mixer", "no attention", "mlp"],
              starters={"tokens_hidden": 128, "channels_hidden": 256},
              doc_url="https://arxiv.org/abs/2105.01601"),
        _spec("mixer", "GatedMLPBlock", "gMLP Block", Category.RESEARCH_MIXER,
              "Liu et al. 2021, arXiv:2105.08050",
              "A spatial gating unit in place of attention: half the projection mixes "
              "across positions and gates the other half.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["gmlp", "spatial gating", "no attention"],
              starters={"hidden_dim": 256},
              doc_url="https://arxiv.org/abs/2105.08050"),
        _spec("mixer", "FourierMixing", "FNet Fourier Mixing", Category.RESEARCH_MIXER,
              "Lee-Thorp et al. 2021, arXiv:2105.03824",
              "Replaces attention with a parameter-free Fourier transform. Much "
              "faster, and a useful baseline for what attention is really worth.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["fnet", "fourier", "fft", "no parameters"],
              doc_url="https://arxiv.org/abs/2105.03824"),
    ]

    # --- Attention variants -------------------------------------------------
    specs += [
        _spec("attention_mod", "RotaryPositionEmbedding", "Rotary Position (RoPE)",
              Category.RESEARCH_ATTENTION, "Su et al. 2021, arXiv:2104.09864",
              "Encodes position by rotating feature pairs, so attention depends only "
              "on relative distance and extrapolates past the trained length.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["rope", "rotary", "llama", "position"],
              doc_url="https://arxiv.org/abs/2104.09864"),
        _spec("attention_mod", "ALiBiAttention", "ALiBi Attention",
              Category.RESEARCH_ATTENTION, "Press et al. 2021, arXiv:2108.12409",
              "Self-attention that encodes position as a linear penalty on distance. "
              "Nothing to learn, and it extrapolates to longer inputs.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["alibi", "position", "extrapolation", "causal"],
              starters={"num_heads": 8, "head_dim": 64},
              doc_url="https://arxiv.org/abs/2108.12409"),
        _spec("attention_mod", "MultiQueryAttention", "Multi-Query Attention",
              Category.RESEARCH_ATTENTION, "Shazeer 2019, arXiv:1911.02150",
              "Many query heads sharing one key/value head, which shrinks the KV cache "
              "by the head count and speeds up generation.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["mqa", "kv cache", "efficient", "decoding"],
              starters={"num_heads": 8, "head_dim": 64},
              doc_url="https://arxiv.org/abs/1911.02150"),
        _spec("attention_mod", "MultiHeadLatentAttention", "Multi-Head Latent Attention",
              Category.RESEARCH_ATTENTION, "DeepSeek-AI 2024, arXiv:2405.04434",
              "DeepSeek's MLA: caches a small latent instead of full keys and values, "
              "with position carried by a separate rotary part.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["mla", "deepseek", "kv cache", "low rank", "efficient"],
              starters={"num_heads": 8, "head_dim": 64, "kv_latent_dim": 128},
              doc_url="https://arxiv.org/abs/2405.04434"),
    ]

    # --- Kolmogorov-Arnold --------------------------------------------------
    specs += [
        _spec("kan", "DenseKAN", "KAN (B-spline)", Category.RESEARCH_KAN,
              "Liu et al. 2024, arXiv:2404.19756",
              "Learns the activation function on every edge as a B-spline, instead of "
              "a weight. Each edge's function can be plotted and read.",
              rank_in=None, modalities=_ANY,
              tags=["kan", "spline", "interpretable", "kolmogorov arnold"],
              starters={"units": 64},
              doc_url="https://arxiv.org/abs/2404.19756"),
        _spec("kan", "ChebyshevKAN", "KAN (Chebyshev)", Category.RESEARCH_KAN,
              "SS et al. 2024, arXiv:2405.07200",
              "KAN using Chebyshev polynomials: no knot grid and the cheapest variant "
              "to evaluate.",
              rank_in=None, modalities=_ANY,
              tags=["kan", "chebyshev", "polynomial", "fast"],
              starters={"units": 64},
              doc_url="https://arxiv.org/abs/2405.07200"),
        _spec("kan", "FastKAN", "KAN (Radial Basis)", Category.RESEARCH_KAN,
              "Li 2024, arXiv:2405.06721",
              "KAN with Gaussian radial bases, which closely approximate cubic "
              "B-splines at a fraction of the cost.",
              rank_in=None, modalities=_ANY,
              tags=["kan", "rbf", "gaussian", "fast"],
              starters={"units": 64},
              doc_url="https://arxiv.org/abs/2405.06721"),
        _spec("kan", "WaveletKAN", "KAN (Wavelet)", Category.RESEARCH_KAN,
              "Bozorgasl & Chen 2024, arXiv:2405.12832",
              "KAN with a wavelet basis, localised in both position and frequency. "
              "More robust to noisy inputs than splines.",
              rank_in=None, modalities=_ANY,
              tags=["kan", "wavelet", "wav-kan", "robust"],
              starters={"units": 64},
              doc_url="https://arxiv.org/abs/2405.12832"),
    ]

    # --- State space --------------------------------------------------------
    specs += [
        _spec("ssm", "MambaBlock", "Mamba Block", Category.RESEARCH_SSM,
              "Gu & Dao 2023, arXiv:2312.00752",
              "A state-space block whose dynamics depend on the input, so it can keep "
              "one thing and forget another while still reading in linear time.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["mamba", "ssm", "s6", "selective", "linear time", "long context"],
              doc_url="https://arxiv.org/abs/2312.00752"),
        _spec("ssm", "S4DBlock", "S4D (Diagonal State Space)", Category.RESEARCH_SSM,
              "Gu et al. 2022, arXiv:2206.11893",
              "One fixed diagonal recurrence for every token. Simpler than Mamba and "
              "the right choice when the signal really is time-invariant.",
              rank_in=3, rank_out=3, modalities=[Modality.AUDIO, Modality.SEQUENCE],
              tags=["s4", "s4d", "ssm", "long range", "audio"],
              doc_url="https://arxiv.org/abs/2206.11893"),
    ]

    # --- Mixture of experts -------------------------------------------------
    specs += [
        _spec("moe", "SparseMoE", "Sparse Mixture of Experts", Category.RESEARCH_MOE,
              "Shazeer et al. 2017, arXiv:1701.06538; Fedus et al. 2021, arXiv:2101.03961",
              "Routes each token to its top-k experts out of many, so capacity grows "
              "while the work per token stays fixed. Use top_k=1 for Switch.",
              rank_in=None, modalities=_SEQ,
              tags=["moe", "mixtral", "switch", "router", "sparse", "experts"],
              starters={"num_experts": 8, "expert_dim": 256},
              doc_url="https://arxiv.org/abs/2101.03961"),
        _spec("moe", "SoftMoE", "Soft Mixture of Experts", Category.RESEARCH_MOE,
              "Puigcerver et al. 2023, arXiv:2308.00951",
              "Mixture of experts with no discrete routing: no balancing loss, no "
              "dropped tokens, fully differentiable.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["moe", "soft", "slots", "differentiable"],
              starters={"num_experts": 8, "expert_dim": 256},
              doc_url="https://arxiv.org/abs/2308.00951"),
    ]

    # --- Modern recurrent ---------------------------------------------------
    specs += [
        _spec("recurrent_mod", "Retention", "Retention (RetNet)",
              Category.RESEARCH_RECURRENT, "Sun et al. 2023, arXiv:2307.08621",
              "Attention with the softmax replaced by a fixed decay, which makes it "
              "associative: parallel to train, recurrent to generate.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["retnet", "retention", "linear attention", "causal"],
              starters={"num_heads": 4, "head_dim": 64},
              doc_url="https://arxiv.org/abs/2307.08621"),
        _spec("recurrent_mod", "MatrixLSTM", "mLSTM (xLSTM)",
              Category.RESEARCH_RECURRENT, "Beck et al. 2024, arXiv:2405.04517",
              "An LSTM with matrix memory and exponential gates, so it can revise a "
              "stored value rather than only decay it.",
              rank_in=3, rank_out=3, modalities=_SEQ,
              tags=["xlstm", "mlstm", "exponential gating", "causal"],
              starters={"num_heads": 4, "head_dim": 64},
              doc_url="https://arxiv.org/abs/2405.04517"),
        _spec("recurrent_mod", "CfC", "Closed-form Continuous-time (Liquid)",
              Category.RESEARCH_RECURRENT,
              "Hasani et al. 2022, Nature Machine Intelligence; arXiv:2106.13898",
              "A liquid network without the ODE solver: each neuron's time constant "
              "depends on its input. Unusually robust on control and sensor tasks.",
              rank_in=3, modalities=[Modality.SEQUENCE, Modality.TABULAR],
              tags=["liquid", "cfc", "ltc", "continuous time", "ode", "robotics"],
              starters={"units": 64},
              doc_url="https://arxiv.org/abs/2106.13898"),
    ]

    # --- Graph networks -----------------------------------------------------
    for name, label, paper, description, tags, starters, url in (
        ("GCNConv", "Graph Convolution (GCN)", "Kipf & Welling 2016, arXiv:1609.02907",
         "Averages each node with its neighbours, then projects. The layer that made "
         "graph networks practical.",
         ["gcn", "graph", "message passing"], {"units": 64},
         "https://arxiv.org/abs/1609.02907"),
        ("GATConv", "Graph Attention (GAT / GATv2)",
         "Veličković et al. 2017, arXiv:1710.10903; Brody et al. 2021, arXiv:2105.14491",
         "Learns how much each neighbour matters, rather than weighting by degree.",
         ["gat", "gatv2", "graph attention"], {"units": 64},
         "https://arxiv.org/abs/2105.14491"),
        ("GraphSAGEConv", "GraphSAGE", "Hamilton et al. 2017, arXiv:1706.02216",
         "Keeps a node's own features separate from its neighbours' summary, instead "
         "of averaging them together.",
         ["sage", "graph", "inductive"], {"units": 64},
         "https://arxiv.org/abs/1706.02216"),
        ("GINConv", "Graph Isomorphism Network (GIN)", "Xu et al. 2018, arXiv:1810.00826",
         "Sum aggregation plus an MLP: provably the most expressive message passing a "
         "graph network can do.",
         ["gin", "graph", "expressive", "weisfeiler lehman"], {"units": 64},
         "https://arxiv.org/abs/1810.00826"),
    ):
        specs.append(
            _spec("graph", name, label, Category.RESEARCH_GRAPH, paper, description,
                  rank_in=3, rank_out=3, modalities=_GRAPH, tags=tags, starters=starters,
                  min_inputs=2, max_inputs=2, inputs=_GRAPH_PORTS,
                  call_style=CallStyle.LIST, doc_url=url)
        )
    specs.append(
        _spec("graph", "GlobalAttentionPool", "Global Attention Pooling",
              Category.RESEARCH_GRAPH, "Li et al. 2015, arXiv:1511.05493",
              "Pools a set of nodes into one vector, weighting each by a learned gate "
              "rather than treating them alike.",
              rank_in=3, rank_out=2, modalities=_GRAPH,
              tags=["graph", "pooling", "readout", "attention"],
              starters={"units": 64},
              doc_url="https://arxiv.org/abs/1511.05493")
    )

    # --- Spiking and neuromorphic -------------------------------------------
    specs += [
        _spec("neuro", "SpikingLIF", "Spiking LIF Neurons", Category.RESEARCH_NEURO,
              "Neftci et al. 2019, arXiv:1901.09948",
              "Leaky integrate-and-fire neurons that communicate with discrete spikes, "
              "trained with a surrogate gradient.",
              rank_in=3, modalities=[Modality.SEQUENCE, Modality.AUDIO],
              tags=["spiking", "snn", "lif", "neuromorphic", "surrogate gradient"],
              starters={"units": 64},
              overrides={"threshold": {
                  "help": "Membrane potential at which the neuron fires a spike."}},
              doc_url="https://arxiv.org/abs/1901.09948"),
        _spec("neuro", "PoissonEncoder", "Poisson Spike Encoder", Category.RESEARCH_NEURO,
              "Tavanaei et al. 2019, arXiv:1804.08150",
              "Turns a static input into a spike train whose firing rate encodes its "
              "value. The usual front end of a spiking network.",
              rank_in=2, rank_out=3, modalities=_ANY,
              tags=["spiking", "snn", "rate coding", "encoder"],
              doc_url="https://arxiv.org/abs/1804.08150"),
    ]

    # --- Other --------------------------------------------------------------
    specs += [
        _spec("misc", "HighwayDense", "Highway Dense", Category.RESEARCH_MISC,
              "Srivastava et al. 2015, arXiv:1505.00387",
              "A dense layer with a learned gate deciding how much to transform and "
              "how much to carry through.",
              rank_in=None, modalities=_ANY, tags=["highway", "gated", "deep"],
              doc_url="https://arxiv.org/abs/1505.00387"),
        _spec("misc", "LoRADense", "LoRA Dense", Category.RESEARCH_MISC,
              "Hu et al. 2021, arXiv:2106.09685",
              "A frozen dense layer with a small trainable low-rank correction, which "
              "folds back into the original weights after training.",
              rank_in=None, modalities=_ANY,
              tags=["lora", "fine-tuning", "low rank", "peft", "adapter"],
              starters={"units": 64},
              doc_url="https://arxiv.org/abs/2106.09685"),
        _spec("misc", "VectorQuantizer", "Vector Quantizer (VQ-VAE)",
              Category.RESEARCH_MISC, "van den Oord et al. 2017, arXiv:1711.00937",
              "Snaps each vector to the nearest entry of a learned codebook, turning a "
              "continuous representation into a discrete one.",
              rank_in=None, modalities=_ANY,
              tags=["vqvae", "codebook", "discrete", "quantization", "tokenizer"],
              starters={"num_codes": 512, "code_dim": 64},
              doc_url="https://arxiv.org/abs/1711.00937"),
        _spec("misc", "SetAttentionPooling", "Set Attention Pooling (PMA)",
              Category.RESEARCH_MISC, "Lee et al. 2019, arXiv:1810.00825",
              "Pools a set with learned seed vectors attending over its elements, "
              "which can represent interactions that mean pooling cannot.",
              rank_in=3, rank_out=3, modalities=[Modality.SEQUENCE, Modality.GRAPH],
              tags=["set transformer", "pma", "pooling", "permutation invariant"],
              doc_url="https://arxiv.org/abs/1810.00825"),
        _spec("misc", "PerceiverCrossAttention", "Perceiver Cross-Attention",
              Category.RESEARCH_MISC, "Jaegle et al. 2021, arXiv:2103.03206",
              "Compresses an input of any size into a fixed number of latents, so cost "
              "is linear in input size instead of quadratic.",
              rank_in=3, rank_out=3, modalities=_ANY,
              tags=["perceiver", "latent", "cross attention", "any modality"],
              starters={"num_latents": 64, "latent_dim": 128},
              doc_url="https://arxiv.org/abs/2103.03206"),
        _spec("misc", "CapsuleLayer", "Capsule Layer", Category.RESEARCH_MISC,
              "Sabour et al. 2017, arXiv:1710.09829",
              "Capsules output vectors whose length is confidence and direction is "
              "pose, routed by agreement rather than pooled.",
              rank_in=3, rank_out=3, modalities=_IMAGE,
              tags=["capsule", "routing", "dynamic routing", "pose"],
              doc_url="https://arxiv.org/abs/1710.09829"),
    ]

    return specs
