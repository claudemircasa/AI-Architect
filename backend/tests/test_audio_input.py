"""
> [!AML-DOC-FILE]
@file       tests/test_audio_input.py
@description Regression tests for feeding a model whose input is a particular
             representation rather than raw data — a mel spectrogram, in the case
             that prompted them.
@module     tests.test_audio_input
@exports    (pytest test functions)
@created    2026-10-01
@context    The preview used to reshape raw samples into whatever block the model
             asked for. It ran, and drew something that looked like a spectrogram and
             carried no spectral information at all [E-041]. These tests check the
             *property* that makes a spectrogram one — energy landing where the sound
             actually is — because a test on the shape would have passed throughout
             the entire time the feature was wrong.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile

from nnarch.catalog import Registry, bootstrap
from nnarch.data.adapters import DatasetError, decode_upload, take_upload_note
from nnarch.ir import GraphIR, compile_graph
from nnarch.ir.schema import Edge, Node

RATE = 22050


@pytest.fixture(scope="module")
def registry() -> Registry:
    """
    > [!AML-DOC-UNIT]
    The catalog, built once.
    @returns the bootstrapped registry
    """
    return bootstrap()


def _wav(wave: np.ndarray) -> bytes:
    """
    > [!AML-DOC-UNIT]
    Encode a waveform as a WAV file.
    @param wave mono samples
    @returns the file's bytes
    """
    buffer = io.BytesIO()
    soundfile.write(buffer, wave.astype("float32"), RATE, format="WAV")
    return buffer.getvalue()


def _tone(hz: float, seconds: float = 3.0) -> bytes:
    """
    > [!AML-DOC-UNIT]
    A pure sine tone.
    @param hz      frequency
    @param seconds duration
    @returns WAV bytes
    """
    t = np.linspace(0, seconds, int(RATE * seconds), dtype="float32")
    return _wav(0.5 * np.sin(2 * np.pi * hz * t))


def _concentration(block: np.ndarray) -> float:
    """
    > [!AML-DOC-UNIT]
    How much of the energy sits in the five loudest frequency bands.
    @param block a (bands, frames) spectrogram
    @returns the fraction, between 0 and 1
    @sideEffects none
    """
    per_band = block.mean(axis=1)
    per_band = per_band - per_band.min()
    return float(np.sort(per_band)[-5:].sum() / max(per_band.sum(), 1e-6))


def test_a_tone_lands_in_a_few_bands_and_noise_does_not() -> None:
    """
    > [!AML-DOC-UNIT]
    The transform carries real spectral structure.
    @raises AssertionError when a tone is no more concentrated than noise
    @context This is the test that the old behaviour could never have passed, and the
             reason it is written as a property rather than a shape: raw samples
             reshaped into an 80x128 block have exactly the right shape.
    """
    tone = decode_upload(_tone(440.0), "tone.wav", [80, 128, 1])[0, :, :, 0]
    take_upload_note()
    noise = decode_upload(
        _wav(0.5 * np.random.default_rng(0).standard_normal(RATE * 3)),
        "noise.wav",
        [80, 128, 1],
    )[0, :, :, 0]
    take_upload_note()

    assert _concentration(tone) > 0.5, "a pure tone did not concentrate its energy"
    assert _concentration(noise) < 0.3, "white noise was not spread out"
    assert _concentration(tone) > _concentration(noise) * 1.5


def test_a_higher_note_sits_higher_up() -> None:
    """
    > [!AML-DOC-UNIT]
    The frequency axis is a frequency axis.
    @raises AssertionError when pitch does not order the bands
    @context Concentration alone would be satisfied by any transform that happens to
             be peaky. This pins the axis down: a higher note must peak higher.
    """
    peaks = []
    for hz in (220.0, 880.0, 3520.0):
        block = decode_upload(_tone(hz), f"{hz}.wav", [80, 128, 1])[0, :, :, 0]
        take_upload_note()
        peaks.append(int(np.argmax(block.mean(axis=1))))

    assert peaks == sorted(peaks), f"pitch did not order the mel bands: {peaks}"
    assert peaks[0] != peaks[-1]


def test_the_window_is_a_window_not_the_whole_file() -> None:
    """
    > [!AML-DOC-UNIT]
    Frames are spaced by a musical hop, not by the file's length.
    @raises AssertionError when the hop is derived from the duration
    @context Spreading 128 frames across a four-minute song gives one every two
             seconds. It fills the array, and the frames have nothing to do with each
             other. The hop must not change when the file gets longer.
    """
    notes = []
    for seconds in (4.0, 60.0):
        t = np.linspace(0, seconds, int(RATE * seconds), dtype="float32")
        decode_upload(_wav(0.4 * np.sin(2 * np.pi * 440 * t)), "x.wav", [80, 128, 1])
        notes.append(take_upload_note() or "")

    assert "hop 512 samples" in notes[0], notes[0]
    assert "hop 512 samples" in notes[1], "a longer file changed the hop"
    assert "first 2.97s" in notes[0], notes[0]


def test_the_preview_says_what_it_did_to_the_file() -> None:
    """
    > [!AML-DOC-UNIT]
    Every chosen parameter is reported.
    @raises AssertionError when the note omits one
    @context The parameters are picked to fit the input, not read from the model, so
             they are very likely not the ones it was trained with. That is tolerable
             only because it is stated.
    """
    decode_upload(_tone(440.0), "clip.wav", [80, 128, 1])
    note = take_upload_note() or ""

    for fragment in ("80 mel bands", "128 frames", f"{RATE} Hz", "FFT 2048", "hop 512"):
        assert fragment in note, f"{fragment!r} missing from: {note}"
    assert "not read from the model" in note, note
    assert "Mel Spectrogram layer" in note, "the exact route was not offered"


def test_a_transposed_spectrogram_is_recognised() -> None:
    """
    > [!AML-DOC-UNIT]
    Frames-by-bands works too, and says so.
    @raises AssertionError when the orientation is not handled or not reported
    @context Which axis is frequency cannot be read off the shape, so the usual
             convention is tried first and the transpose only if it does not fit.
    """
    block = decode_upload(_tone(440.0), "clip.wav", [128, 80, 1])
    assert block.shape == (1, 128, 80, 1)
    take_upload_note()


def test_audio_into_an_impossible_shape_is_refused() -> None:
    """
    > [!AML-DOC-UNIT]
    A shape no spectrogram can fill is an error, not a reshape.
    @raises AssertionError when the file is forced into the shape anyway
    @context Filling it would be the original defect: something that runs and means
             nothing.
    """
    with pytest.raises(DatasetError) as caught:
        decode_upload(_tone(440.0), "clip.wav", [8, 8, 3])
    assert "channel" in str(caught.value).lower() or "spectrogram" in str(caught.value)


def test_the_waveform_route_still_works() -> None:
    """
    > [!AML-DOC-UNIT]
    A model that takes samples still gets samples.
    @raises AssertionError when a one-dimensional input is transformed
    @context Not every audio model wants a spectrogram, and the ones that take the
             waveform must not have one computed for them.
    """
    out = decode_upload(_tone(440.0), "clip.wav", [16000])
    assert out.shape == (1, 16000)
    assert float(np.abs(out).max()) <= 1.0, "a waveform came back on a dB scale"
    assert "waveform" in (take_upload_note() or "")


def test_the_transform_can_live_in_the_graph_instead(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A Mel Spectrogram layer compiles, so a model can take audio itself.
    @param registry the catalog
    @raises AssertionError when the layer will not build
    @context This is the exact answer to a model that wants a particular
             representation: compute it inside the architecture, where the parameters
             are visible, editable, and travel with the exported model.
    """
    graph = GraphIR(
        name="audio-in",
        nodes=[
            Node(id="wav", type="keras.Input", name="wav", params={"shape": [66150]}),
            Node(id="mel", type="keras.MelSpectrogram", name="mel", params={
                "num_mel_bins": 80, "sampling_rate": 22050,
                "sequence_stride": 512, "fft_length": 2048,
            }),
        ],
        edges=[Edge(id="e", source="wav", target="mel")],
    )
    result = compile_graph(graph, registry)
    assert result.model is not None, [
        d.message for d in result.diagnostics if d.severity == "error"
    ]
    mel = next(layer for layer in result.model.layers if layer.name == "mel")
    assert tuple(mel.output.shape)[1] == 80
