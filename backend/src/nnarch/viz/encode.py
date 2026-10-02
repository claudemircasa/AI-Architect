"""
> [!AML-DOC-FILE]
@file       viz/encode.py
@description Turns tensors into things a browser can draw: colour-mapped PNG tiles,
             downsampled grids and compact numeric summaries.
@module     nnarch.viz.encode
@exports    COLORMAPS, normalise, to_png, tile_grid, downsample_1d, summarise
@created    2026-10-01
@context    A single convolution layer can hold tens of megabytes of activations.
            Sending them raw would stall the editor, so everything here downsamples
            and caps before encoding [task 05]. The caps are generous enough that
            what the user sees is still the real tensor, just smaller.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

MAX_CHANNELS = 64
"""Feature maps shown per layer. Beyond this a grid is unreadable anyway."""

MAX_TILE = 128
"""Longest side of one feature-map tile, in pixels."""

MAX_SERIES = 2048
"""Points kept in a one-dimensional trace such as a waveform."""


def _viridis() -> np.ndarray:
    """
    > [!AML-DOC-UNIT]
    Build the viridis colour ramp as a 256-entry lookup table.
    @returns an array of shape (256, 3), dtype uint8
    @sideEffects none
    @context Viridis is used because it is perceptually uniform and readable to
             colour-blind viewers, which matters when the colour *is* the data.
             The anchors are interpolated rather than listing 256 rows.
    """
    anchors = np.array([
        [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
        [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
        [180, 222, 44], [253, 231, 37],
    ], dtype="float32")
    positions = np.linspace(0.0, 1.0, len(anchors))
    targets = np.linspace(0.0, 1.0, 256)
    return np.stack(
        [np.interp(targets, positions, anchors[:, channel]) for channel in range(3)],
        axis=-1,
    ).astype("uint8")


def _magma() -> np.ndarray:
    """
    > [!AML-DOC-UNIT]
    Build the magma colour ramp as a 256-entry lookup table.
    @returns an array of shape (256, 3), dtype uint8
    """
    anchors = np.array([
        [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
        [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135],
        [252, 253, 191],
    ], dtype="float32")
    positions = np.linspace(0.0, 1.0, len(anchors))
    targets = np.linspace(0.0, 1.0, 256)
    return np.stack(
        [np.interp(targets, positions, anchors[:, channel]) for channel in range(3)],
        axis=-1,
    ).astype("uint8")


def _gray() -> np.ndarray:
    """
    > [!AML-DOC-UNIT]
    Build a linear grayscale lookup table.
    @returns an array of shape (256, 3), dtype uint8
    """
    ramp = np.arange(256, dtype="uint8")
    return np.stack([ramp, ramp, ramp], axis=-1)


COLORMAPS: dict[str, np.ndarray] = {
    "viridis": _viridis(),
    "magma": _magma(),
    "gray": _gray(),
}
"""Colour ramps offered to the viewer. Each is a 256-entry RGB lookup table."""


def normalise(values: np.ndarray, *, symmetric: bool = False) -> np.ndarray:
    """
    > [!AML-DOC-UNIT]
    Scale an array into [0, 1] for display.
    @param values    the array to scale
    @param symmetric centre the scale on zero, so positive and negative read
                     differently rather than being stretched independently
    @returns a float array in [0, 1] of the same shape
    @sideEffects none
    @context A constant array maps to the middle of the range rather than producing
             a division by zero, which is what a dead channel looks like.
    """
    finite = np.nan_to_num(values.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    if symmetric:
        limit = float(np.max(np.abs(finite))) or 1.0
        return (finite / (2.0 * limit)) + 0.5
    low = float(finite.min())
    high = float(finite.max())
    if high - low < 1e-12:
        return np.full_like(finite, 0.5)
    return (finite - low) / (high - low)


def to_png(values: np.ndarray, colormap: str = "viridis") -> str:
    """
    > [!AML-DOC-UNIT]
    Render a 2D array as a base64 PNG.
    @param values   a two-dimensional array, already normalised into [0, 1]
    @param colormap name of a ramp in COLORMAPS
    @returns a `data:` URI the browser can use directly as an image source
    @raises ValueError when the array is not two-dimensional
    @sideEffects none
    """
    from PIL import Image

    if values.ndim != 2:
        raise ValueError(f"to_png needs a 2D array, received shape {values.shape}")

    ramp = COLORMAPS.get(colormap, COLORMAPS["viridis"])
    indices = np.clip(values * 255.0, 0, 255).astype("uint8")
    image = Image.fromarray(ramp[indices], mode="RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _resize(plane: np.ndarray, limit: int) -> np.ndarray:
    """
    > [!AML-DOC-UNIT]
    Shrink a 2D array so neither side exceeds a limit, and enlarge a tiny one so it
    is visible.
    @param plane a two-dimensional array
    @param limit longest side allowed, in pixels
    @returns the resized array
    @sideEffects none
    @context Nearest-neighbour throughout: these are data, not photographs, and
             smoothing them would invent values the layer never produced.
    """
    height, width = plane.shape
    if height == 0 or width == 0:
        return plane

    if max(height, width) > limit:
        step = int(np.ceil(max(height, width) / limit))
        return plane[::step, ::step]

    if max(height, width) < 8:
        factor = int(np.ceil(8 / max(height, width)))
        return np.kron(plane, np.ones((factor, factor), dtype=plane.dtype))
    return plane


def tile_grid(
    planes: np.ndarray, colormap: str = "viridis", *, limit: int = MAX_CHANNELS
) -> list[str]:
    """
    > [!AML-DOC-UNIT]
    Render each channel of a feature map as its own image.
    @param planes   an array of shape (height, width, channels)
    @param colormap name of a ramp in COLORMAPS
    @param limit    how many channels to render
    @returns a list of `data:` URIs, one per rendered channel
    @raises ValueError when the array is not three-dimensional
    @sideEffects none
    @context Every channel is normalised against the whole layer rather than against
             itself, so a quiet channel looks quiet instead of being stretched to
             look as active as the loudest one.
    """
    if planes.ndim != 3:
        raise ValueError(f"tile_grid needs (h, w, c), received shape {planes.shape}")

    scaled = normalise(planes)
    count = min(planes.shape[-1], limit)
    return [
        to_png(_resize(scaled[..., index], MAX_TILE), colormap) for index in range(count)
    ]


def downsample_1d(values: np.ndarray, limit: int = MAX_SERIES) -> list[float]:
    """
    > [!AML-DOC-UNIT]
    Reduce a long trace to a drawable number of points.
    @param values a one-dimensional array
    @param limit  how many points to keep
    @returns the kept values as plain floats
    @sideEffects none
    @context Reduces by taking the extreme of each bucket, not the mean: averaging a
             waveform flattens exactly the peaks someone is looking for.
    """
    flat = np.nan_to_num(values.astype("float32").ravel())
    if flat.size <= limit:
        return [float(value) for value in flat]

    buckets = np.array_split(flat, limit)
    return [
        float(bucket[np.argmax(np.abs(bucket))]) if bucket.size else 0.0
        for bucket in buckets
    ]


def summarise(values: np.ndarray) -> dict[str, Any]:
    """
    > [!AML-DOC-UNIT]
    Describe a tensor in a handful of numbers.
    @param values the tensor
    @returns min, max, mean, standard deviation and the fraction of exact zeros
    @sideEffects none
    @context Sparsity is included because it is the fastest way to spot a layer that
             has died: a ReLU stack whose sparsity reaches 1.0 is passing nothing on.
    """
    flat = np.nan_to_num(values.astype("float32").ravel())
    if flat.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "sparsity": 0.0}
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "sparsity": float((flat == 0.0).mean()),
    }
