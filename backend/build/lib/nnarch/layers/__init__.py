"""
> [!AML-DOC-FILE]
@file       layers/__init__.py
@description Research-frontier layers: architectures from cited papers, implemented
             as ordinary Keras 3 layers.
@module     nnarch.layers
@exports    every layer class across the eleven families
@created    2026-09-30
@context    [amm: B.5] every layer here needs a working `get_config()`, because the
            code generator copies its source into exported projects [task 06] and a
            saved project reloads through it [task 12].

            These modules import only keras and the standard library. An exported
            project carries their source but not this package, so a reference to
            `nnarch` inside a layer would break every export [task 06, enforced by
            `tests/test_research_layers.py`].
"""

from __future__ import annotations

from .attention_mod import (
    ALiBiAttention,
    MultiHeadLatentAttention,
    MultiQueryAttention,
    RotaryPositionEmbedding,
)
from .conv_mod import ConvNeXtBlock, DropPath, Involution, MBConv, SqueezeExcite
from .ffn import GeGLU, GLUFeedForward, SwiGLU
from .graph import (
    GATConv,
    GCNConv,
    GINConv,
    GlobalAttentionPool,
    GraphSAGEConv,
)
from .kan import ChebyshevKAN, DenseKAN, FastKAN, WaveletKAN
from .misc import (
    CapsuleLayer,
    HighwayDense,
    LoRADense,
    PerceiverCrossAttention,
    SetAttentionPooling,
    VectorQuantizer,
)
from .mixer import (
    AddPositionEmbedding,
    ClassToken,
    FourierMixing,
    GatedMLPBlock,
    MLPMixerBlock,
    PatchEmbedding,
)
from .moe import SoftMoE, SparseMoE
from .neuro import LIFCell, PoissonEncoder, SpikingLIF
from .recurrent_mod import CfC, CfCCell, MatrixLSTM, Retention
from .ssm import MambaBlock, S4DBlock

__all__ = [
    "ALiBiAttention", "AddPositionEmbedding", "CapsuleLayer", "CfC", "CfCCell",
    "ChebyshevKAN", "ClassToken", "ConvNeXtBlock", "DenseKAN", "DropPath", "FastKAN",
    "FourierMixing", "GATConv", "GCNConv", "GINConv", "GLUFeedForward", "GatedMLPBlock",
    "GeGLU", "GlobalAttentionPool", "GraphSAGEConv", "HighwayDense", "Involution",
    "LIFCell", "LoRADense", "MBConv", "MLPMixerBlock", "MambaBlock", "MatrixLSTM",
    "MultiHeadLatentAttention", "MultiQueryAttention", "PatchEmbedding",
    "PerceiverCrossAttention", "PoissonEncoder", "Retention",
    "RotaryPositionEmbedding", "S4DBlock", "SetAttentionPooling", "SoftMoE",
    "SparseMoE", "SpikingLIF", "SqueezeExcite", "SwiGLU", "VectorQuantizer",
    "WaveletKAN",
]
