"""
> [!AML-DOC-FILE]
@file       data/adapters.py
@description Data sources behind one interface: synthetic tensors, Keras' builtin
             datasets, a single uploaded file, and a directory on disk.
@module     nnarch.data.adapters
@exports    DatasetAdapter, SyntheticAdapter, BuiltinAdapter, UploadAdapter,
            FolderAdapter, make_adapter, decode_upload
@created    2026-10-01
@context    RISK:MED [amm: E.4]. Each adapter answers both "give me the data now"
            and "write the Python that loads it", from the same definition. If those
            two ever diverge, a model trains on one thing in the app and another in
            the exported project, which is the failure this arrangement exists to
            prevent.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .spec import BUILTIN_DATASETS, DatasetChoice, DatasetKind, DatasetSpec, Modality


class DatasetError(ValueError):
    """
    > [!AML-DOC-UNIT]
    Raised when a source cannot provide what the model needs. The message is written
    for the person who chose the source, not for a log.
    """


class DatasetAdapter(ABC):
    """
    > [!AML-DOC-UNIT]
    One data source, resolved against the model it feeds.
    @param choice      the user's selection
    @param input_shape shape the model expects, excluding the batch axis
    @param num_classes number of outputs the model produces, when it classifies
    """

    def __init__(
        self,
        choice: DatasetChoice,
        input_shape: list[int],
        num_classes: int | None = None,
    ) -> None:
        self.choice = choice
        self.input_shape = [int(dim) if dim else 1 for dim in input_shape]
        self.num_classes = num_classes

    @abstractmethod
    def spec(self) -> DatasetSpec:
        """
        > [!AML-DOC-UNIT]
        Describe what this source provides.
        @returns the resolved DatasetSpec
        """

    @abstractmethod
    def sample(self, count: int = 1) -> tuple[np.ndarray, np.ndarray | None]:
        """
        > [!AML-DOC-UNIT]
        Fetch a few examples, ready to feed the model.
        @param count how many examples
        @returns (inputs, labels); labels is None when the source has none
        @raises DatasetError when the source cannot be read
        """

    @abstractmethod
    def codegen(self) -> str:
        """
        > [!AML-DOC-UNIT]
        Emit the body of the exported project's `data.py`.
        @returns Python source defining `load_data()`
        @context This is the other half of [amm: E.4]: whatever `sample` does here,
                 the emitted source must do in the user's project.
        """

    def training_data(self) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """
        > [!AML-DOC-UNIT]
        Training and validation splits for in-app training [task 07].
        @returns ((x_train, y_train), (x_test, y_test))
        @raises DatasetError when the source cannot provide labelled data
        """
        raise DatasetError(
            f"{type(self).__name__} provides single examples for preview, not a "
            f"labelled training set. Choose a builtin dataset or a folder to train."
        )


def _one_sample_shape(shape: list[int]) -> tuple[int, ...]:
    """
    > [!AML-DOC-UNIT]
    Turn a model input shape into a concrete shape, substituting any dynamic axis.
    @param shape the declared shape, excluding batch
    @returns a shape with no None entries
    """
    return tuple(int(dim) if dim else 1 for dim in shape)


class SyntheticAdapter(DatasetAdapter):
    """
    > [!AML-DOC-UNIT]
    Generated tensors of exactly the shape the model wants.
    @context The default, and deliberately so: it needs no download, no files and no
             network, so a graph can be previewed and an exported project can be
             proven to run before any real data exists.
    """

    def spec(self) -> DatasetSpec:
        """
        > [!AML-DOC-UNIT]
        Describe the generated data.
        @returns a DatasetSpec matching the model's own input shape
        """
        return DatasetSpec(
            modality=Modality.TENSOR,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            count=self.choice.samples,
            label=f"synthetic {self.choice.pattern}",
        )

    def _generate(self, count: int, seed: int) -> np.ndarray:
        """
        > [!AML-DOC-UNIT]
        Produce `count` examples of the chosen pattern.
        @param count number of examples
        @param seed  seed for the random generator
        @returns an array of shape (count, *input_shape)
        """
        rng = np.random.default_rng(seed)
        shape = (count, *_one_sample_shape(self.input_shape))
        pattern = self.choice.pattern

        if pattern == "ones":
            return np.ones(shape, dtype="float32")
        if pattern == "zeros":
            return np.zeros(shape, dtype="float32")
        if pattern == "ramp":
            flat = np.linspace(0.0, 1.0, int(np.prod(shape[1:])), dtype="float32")
            return np.broadcast_to(flat, (count, flat.size)).reshape(shape).copy()
        if pattern == "sine":
            flat = np.sin(
                np.linspace(0.0, 6.283, int(np.prod(shape[1:])), dtype="float32")
            )
            return np.broadcast_to(flat, (count, flat.size)).reshape(shape).copy()
        if pattern == "checkerboard":
            flat = np.indices(shape[1:]).sum(axis=0) % 2
            return np.broadcast_to(
                flat.astype("float32").ravel(), (count, flat.size)
            ).reshape(shape).copy()
        return rng.standard_normal(shape).astype("float32")

    def sample(self, count: int = 1) -> tuple[np.ndarray, np.ndarray | None]:
        """
        > [!AML-DOC-UNIT]
        Generate examples.
        @param count how many
        @returns (inputs, labels) with random labels when the model classifies
        """
        inputs = self._generate(count, self.choice.seed)
        if not self.num_classes:
            return inputs, None
        rng = np.random.default_rng(self.choice.seed + 1)
        return inputs, rng.integers(0, self.num_classes, size=(count,))

    def training_data(self):
        """
        > [!AML-DOC-UNIT]
        Generated training and validation splits.
        @returns ((x_train, y_train), (x_test, y_test))
        @context Labels are random, so a model fitted on this cannot learn anything.
                 That is the point: it proves the architecture trains end to end
                 before any real data is involved.
        """
        train = self.sample(self.choice.samples)
        validation = SyntheticAdapter(
            self.choice.model_copy(update={"seed": self.choice.seed + 7}),
            self.input_shape,
            self.num_classes,
        ).sample(max(self.choice.samples // 4, 1))
        return (train[0], train[1]), (validation[0], validation[1])

    def codegen(self) -> str:
        """
        > [!AML-DOC-UNIT]
        Emit a generator matching this model's shapes.
        @returns Python source defining `load_data()`
        """
        shape = ", ".join(str(dim) for dim in _one_sample_shape(self.input_shape))
        classes = self.num_classes or 1
        label_line = (
            f"        y = rng.integers(0, {classes}, size=(count,))"
            if self.num_classes
            else f'        y = rng.standard_normal((count, {classes})).astype("float32")'
        )
        return "\n".join([
            '"""Data loading for this model."""',
            "",
            "from __future__ import annotations",
            "",
            "import numpy as np",
            "",
            "",
            f"SAMPLES = {self.choice.samples}",
            f"SEED = {self.choice.seed}",
            "",
            "",
            "def load_data():",
            '    """Generate reproducible random data with this model\'s shapes.',
            "",
            "    Labels are random, so this checks that the architecture trains end to",
            "    end; it cannot learn anything. Replace this function with your own",
            "    loader when you have real data.",
            '    """',
            "    rng = np.random.default_rng(SEED)",
            "",
            "    def make(count):",
            f'        x = rng.standard_normal((count, {shape})).astype("float32")',
            label_line,
            "        return x, y",
            "",
            "    return make(SAMPLES), make(max(SAMPLES // 4, 1))",
            "",
        ])


class BuiltinAdapter(DatasetAdapter):
    """
    > [!AML-DOC-UNIT]
    One of the datasets Keras can download, reshaped to the model's input.
    """

    def _facts(self) -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Look up what is known about the chosen dataset.
        @returns the entry from BUILTIN_DATASETS
        @raises DatasetError when the name is not one Keras provides
        """
        facts = BUILTIN_DATASETS.get(self.choice.name)
        if facts is None:
            raise DatasetError(
                f"'{self.choice.name}' is not a builtin dataset. Available: "
                f"{', '.join(sorted(BUILTIN_DATASETS))}."
            )
        return facts

    def spec(self) -> DatasetSpec:
        """
        > [!AML-DOC-UNIT]
        Describe the chosen dataset.
        @returns the resolved DatasetSpec
        @raises DatasetError when the name is unknown
        """
        facts = self._facts()
        return DatasetSpec(
            modality=facts["modality"],
            input_shape=facts["shape"] or self.input_shape,
            num_classes=facts["classes"],
            class_names=facts["class_names"],
            label=facts["label"],
        )

    def _load(self):
        """
        > [!AML-DOC-UNIT]
        Download and shape the dataset.
        @returns ((x_train, y_train), (x_test, y_test))
        @sideEffects downloads to the Keras cache on first use
        """
        import keras

        facts = self._facts()
        (x_train, y_train), (x_test, y_test) = getattr(
            keras.datasets, self.choice.name
        ).load_data()

        if facts["modality"] is Modality.IMAGE:
            target = (-1, *_one_sample_shape(self.input_shape))
            x_train = (x_train.astype("float32") / 255.0).reshape(target)
            x_test = (x_test.astype("float32") / 255.0).reshape(target)
        else:
            length = self.input_shape[0] if self.input_shape else 256
            x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=length)
            x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=length)

        return (x_train, np.asarray(y_train).reshape(-1)), (
            x_test, np.asarray(y_test).reshape(-1)
        )

    def sample(self, count: int = 1) -> tuple[np.ndarray, np.ndarray | None]:
        """
        > [!AML-DOC-UNIT]
        Take a few examples from the training split.
        @param count how many
        @returns (inputs, labels), chosen at random so repeated previews differ
        """
        (x_train, y_train), _ = self._load()
        rng = np.random.default_rng(self.choice.seed)
        picked = rng.choice(len(x_train), size=min(count, len(x_train)), replace=False)
        return x_train[picked], y_train[picked]

    def training_data(self):
        """
        > [!AML-DOC-UNIT]
        The dataset's own training and test splits.
        @returns ((x_train, y_train), (x_test, y_test))
        """
        return self._load()

    def codegen(self) -> str:
        """
        > [!AML-DOC-UNIT]
        Emit a loader for this dataset, preprocessed exactly as `_load` does.
        @returns Python source defining `load_data()`
        """
        facts = self._facts()
        header = [
            '"""Data loading for this model."""',
            "",
            "from __future__ import annotations",
            "",
            "import keras",
            "import numpy as np",
            "",
            "",
            "def load_data():",
            f'    """Load {facts["label"]} and shape it for this model."""',
            f"    (x_train, y_train), (x_test, y_test) = "
            f"keras.datasets.{self.choice.name}.load_data()",
        ]
        if facts["modality"] is Modality.IMAGE:
            shape = ", ".join(str(dim) for dim in _one_sample_shape(self.input_shape))
            header += [
                '    x_train = x_train.astype("float32") / 255.0',
                '    x_test = x_test.astype("float32") / 255.0',
                f"    x_train = x_train.reshape((-1, {shape}))",
                f"    x_test = x_test.reshape((-1, {shape}))",
            ]
        else:
            length = self.input_shape[0] if self.input_shape else 256
            header += [
                "    x_train = keras.preprocessing.sequence.pad_sequences("
                f"x_train, maxlen={length})",
                "    x_test = keras.preprocessing.sequence.pad_sequences("
                f"x_test, maxlen={length})",
            ]
        header += [
            "    y_train = np.asarray(y_train).reshape((-1,))",
            "    y_test = np.asarray(y_test).reshape((-1,))",
            "    return (x_train, y_train), (x_test, y_test)",
            "",
        ]
        return "\n".join(header)


class UploadAdapter(DatasetAdapter):
    """
    > [!AML-DOC-UNIT]
    A single file the user dropped in, for watching one example flow through.
    @param payload raw file bytes
    @param filename original name, which decides how the bytes are read
    """

    def __init__(
        self,
        choice: DatasetChoice,
        input_shape: list[int],
        num_classes: int | None = None,
        payload: bytes = b"",
        filename: str = "",
    ) -> None:
        super().__init__(choice, input_shape, num_classes)
        self.payload = payload
        self.filename = filename

    def spec(self) -> DatasetSpec:
        """
        > [!AML-DOC-UNIT]
        Describe the uploaded example.
        @returns a DatasetSpec whose modality follows the file extension
        """
        return DatasetSpec(
            modality=detect_modality(self.filename),
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            count=1,
            label=self.filename or "uploaded file",
        )

    def sample(self, count: int = 1) -> tuple[np.ndarray, np.ndarray | None]:
        """
        > [!AML-DOC-UNIT]
        Decode the file into the shape the model expects.
        @param count ignored; an upload is one example
        @returns (inputs, None)
        @raises DatasetError when the file cannot be read as the model's input
        """
        return decode_upload(self.payload, self.filename, self.input_shape), None

    def codegen(self) -> str:
        """
        > [!AML-DOC-UNIT]
        An uploaded file is a preview, not a training set, so the exported project
        gets a synthetic loader and a comment saying where to put real data.
        @returns Python source defining `load_data()`
        """
        source = SyntheticAdapter(
            self.choice.model_copy(update={"kind": DatasetKind.SYNTHETIC}),
            self.input_shape,
            self.num_classes,
        ).codegen()
        return source.replace(
            '"""Data loading for this model."""',
            '"""Data loading for this model.\n\n'
            "The architecture was previewed with a single uploaded file, which is not a\n"
            "training set, so this generates data of the right shape instead. Replace\n"
            '`load_data()` with your own loader.\n"""',
        )


class FolderAdapter(DatasetAdapter):
    """
    > [!AML-DOC-UNIT]
    A directory on disk laid out as one subdirectory per class.
    """

    def spec(self) -> DatasetSpec:
        """
        > [!AML-DOC-UNIT]
        Describe the directory by reading its class subdirectories.
        @returns the resolved DatasetSpec
        @raises DatasetError when the path is missing or has no class folders
        """
        root = Path(self.choice.path).expanduser()
        if not root.is_dir():
            raise DatasetError(f"'{self.choice.path}' is not a directory this machine can read.")
        classes = sorted(
            entry.name for entry in root.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )
        if not classes:
            raise DatasetError(
                f"'{self.choice.path}' has no class subdirectories. Keras expects one "
                f"folder per class, each holding that class's files."
            )
        return DatasetSpec(
            modality=detect_modality(next(iter(root.rglob("*.*")), Path("x.bin")).name),
            input_shape=self.input_shape,
            num_classes=len(classes),
            class_names=classes,
            label=root.name,
        )

    def _datasets(self):
        """
        > [!AML-DOC-UNIT]
        Build the training and validation `tf.data` datasets.
        @returns (training, validation)
        @raises DatasetError when the directory cannot be read
        @sideEffects reads files from disk
        """
        import keras

        size = tuple(self.input_shape[:2]) if len(self.input_shape) >= 2 else (224, 224)
        shared = {
            "directory": str(Path(self.choice.path).expanduser()),
            "labels": "inferred",
            "label_mode": "int",
            "image_size": size,
            "batch_size": 32,
            "seed": self.choice.seed,
            "validation_split": self.choice.validation_split,
        }
        training = keras.utils.image_dataset_from_directory(subset="training", **shared)
        validation = keras.utils.image_dataset_from_directory(
            subset="validation", **shared
        )
        return training, validation

    def sample(self, count: int = 1) -> tuple[np.ndarray, np.ndarray | None]:
        """
        > [!AML-DOC-UNIT]
        Take a few examples from the directory.
        @param count how many
        @returns (inputs, labels)
        """
        training, _ = self._datasets()
        for images, labels in training.take(1):
            x = np.asarray(images)[:count].astype("float32") / 255.0
            return x, np.asarray(labels)[:count]
        raise DatasetError("The directory produced no batches; it may hold no readable files.")

    def training_data(self):
        """
        > [!AML-DOC-UNIT]
        Materialise the directory into arrays.
        @returns ((x_train, y_train), (x_test, y_test))
        @sideEffects reads every file, so this is slow for a large directory
        """
        training, validation = self._datasets()

        def collect(dataset):
            """
            > [!AML-DOC-UNIT]
            Read a whole tf.data dataset into arrays.
            @param dataset the batched dataset to drain
            @returns (inputs, labels) as concatenated arrays
            @sideEffects reads every file the dataset references
            """
            xs, ys = [], []
            for images, labels in dataset:
                xs.append(np.asarray(images).astype("float32") / 255.0)
                ys.append(np.asarray(labels))
            return np.concatenate(xs), np.concatenate(ys)

        return collect(training), collect(validation)

    def codegen(self) -> str:
        """
        > [!AML-DOC-UNIT]
        Emit a loader reading the same directory with the same preprocessing.
        @returns Python source defining `load_data()`
        """
        size = tuple(self.input_shape[:2]) if len(self.input_shape) >= 2 else (224, 224)
        return "\n".join([
            '"""Data loading for this model."""',
            "",
            "from __future__ import annotations",
            "",
            "import keras",
            "import numpy as np",
            "",
            "",
            f"DATA_DIR = {Path(self.choice.path).expanduser().as_posix()!r}",
            f"IMAGE_SIZE = {size!r}",
            f"VALIDATION_SPLIT = {self.choice.validation_split}",
            f"SEED = {self.choice.seed}",
            "",
            "",
            "def load_data():",
            '    """Read the image directory, one subdirectory per class."""',
            "    shared = dict(",
            "        directory=DATA_DIR,",
            '        labels="inferred",',
            '        label_mode="int",',
            "        image_size=IMAGE_SIZE,",
            "        batch_size=32,",
            "        seed=SEED,",
            "        validation_split=VALIDATION_SPLIT,",
            "    )",
            '    training = keras.utils.image_dataset_from_directory(subset="training", **shared)',
            '    validation = keras.utils.image_dataset_from_directory(subset="validation", **shared)',
            "",
            "    def collect(dataset):",
            "        xs, ys = [], []",
            "        for images, labels in dataset:",
            '            xs.append(np.asarray(images).astype("float32") / 255.0)',
            "            ys.append(np.asarray(labels))",
            "        return np.concatenate(xs), np.concatenate(ys)",
            "",
            "    return collect(training), collect(validation)",
            "",
        ])


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
_AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}
_TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json"}


def detect_modality(filename: str) -> Modality:
    """
    > [!AML-DOC-UNIT]
    Guess what a file holds from its extension.
    @param filename the original name
    @returns the modality, defaulting to a raw tensor
    @sideEffects none
    """
    suffix = Path(filename).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return Modality.IMAGE
    if suffix in _AUDIO_SUFFIXES:
        return Modality.AUDIO
    if suffix in _TEXT_SUFFIXES:
        return Modality.TEXT
    return Modality.TENSOR


#: Set while an upload is being decoded, so the preview can say what it did to it.
_LAST_NOTE: list[str] = []


def _note(message: str) -> None:
    """
    > [!AML-DOC-UNIT]
    Record what was done to an uploaded file.
    @param message one sentence, shown with the preview
    @sideEffects replaces the pending note
    """
    _LAST_NOTE[:] = [message]


def take_upload_note() -> str | None:
    """
    > [!AML-DOC-UNIT]
    Read and clear the note left by the last decode.
    @returns the note, or None when the file needed no interpreting
    @sideEffects clears it, so a later preview cannot inherit an old one
    """
    return _LAST_NOTE.pop() if _LAST_NOTE else None


def _audio_as_spectrogram(
    samples: np.ndarray, rate: int, shape: list[int], filename: str
) -> np.ndarray:
    """
    > [!AML-DOC-UNIT]
    Turn a waveform into the time-frequency block a model asks for.
    @param samples  mono audio
    @param rate     its sample rate
    @param shape    the model's input shape, batch axis excluded
    @param filename for the message
    @returns an array of exactly `(1, *shape)`
    @raises DatasetError when the shape is not one a spectrogram can fill
    @sideEffects records a note describing every parameter it chose
    @context Which axis is frequency cannot be known from the shape alone — (80, 128)
             is as plausibly 80 mel bands over 128 frames as the transpose. The usual
             convention is tried first and the transpose only if the first does not
             fit, and whichever was used is said out loud. The remaining parameters
             are chosen to hit the requested frame count exactly; they will not match
             the pipeline the model was trained with unless that pipeline happened to
             use them, which is the reason this says what it did [E-041].
    """
    import keras

    spatial = [int(dim) for dim in shape if dim is not None]
    if len(spatial) == 3 and spatial[2] != 1:
        raise DatasetError(
            f"This model's input is {list(shape)}. A spectrogram of one audio file "
            f"has a single channel, so it cannot fill {spatial[2]} of them."
        )
    if len(spatial) not in (2, 3):
        raise DatasetError(
            f"'{filename}' is audio, and this model's input is {list(shape)}, which is "
            f"neither a waveform nor a time-frequency block."
        )

    fft_length = 2048
    # A quarter of the window is the ordinary hop, and it is a musical interval —
    # about 12ms at 44.1kHz. Deriving the hop from the file's *length* instead would
    # spread 128 frames across a four-minute song, one every two seconds, which looks
    # like a spectrogram and is 128 unrelated snapshots [E-041].
    stride = fft_length // 4
    for bins, frames, transposed in (
        (spatial[0], spatial[1], False),
        (spatial[1], spatial[0], True),
    ):
        if bins > 512 or frames < 2:
            continue
        # The window these frames actually cover, taken from the start of the file.
        needed = stride * frames + fft_length
        padded = samples
        if padded.size < needed:
            padded = np.pad(padded, (0, needed - padded.size))

        layer = keras.layers.MelSpectrogram(
            num_mel_bins=bins,
            sampling_rate=rate,
            sequence_stride=stride,
            fft_length=fft_length,
        )
        block = np.asarray(layer(padded[None, :needed].astype("float32")))

        # Trim or pad the frame axis to land on exactly what was asked for.
        if block.shape[2] < frames:
            block = np.pad(block, ((0, 0), (0, 0), (0, frames - block.shape[2])))
        block = block[:, :, :frames]

        if transposed:
            block = np.transpose(block, (0, 2, 1))
        if len(spatial) == 3:
            block = block[..., None]

        hop_ms = 1000.0 * stride / max(rate, 1)
        window_s = (stride * frames) / max(rate, 1)
        _note(
            f"'{filename}' became a mel spectrogram: {bins} mel bands × {frames} "
            f"frames{' (transposed to match)' if transposed else ''} covering the "
            f"first {window_s:.2f}s, {rate} Hz, FFT {fft_length}, hop {stride} samples "
            f"({hop_ms:.0f} ms), dB scale. These fit the input; they are not read from "
            f"the model. If it was trained on others, put a Mel Spectrogram layer in "
            f"the graph instead, or upload the features as .npy."
        )
        return block.astype("float32")

    raise DatasetError(
        f"This model's input is {list(shape)}, which does not look like a spectrogram "
        f"of any orientation, so '{filename}' cannot be turned into it."
    )

def decode_upload(payload: bytes, filename: str, input_shape: list[int]) -> np.ndarray:
    """
    > [!AML-DOC-UNIT]
    Read an uploaded file into a batch of one, shaped for the model.
    @param payload     raw file bytes
    @param filename    original name, which decides the decoder
    @param input_shape shape the model expects, excluding the batch axis
    @returns an array of shape (1, *input_shape)
    @raises DatasetError when the file cannot be decoded or reshaped
    @sideEffects none
    @context Images are resized and channel-matched to the model rather than
             refused, because a model wanting 28x28 grayscale should still accept a
             photograph: the point is to watch it flow through.
    """
    modality = detect_modality(filename)
    shape = _one_sample_shape(input_shape)

    try:
        if modality is Modality.IMAGE:
            from PIL import Image

            image = Image.open(io.BytesIO(payload))
            if len(shape) >= 3:
                height, width, channels = shape[0], shape[1], shape[2]
            elif len(shape) == 2:
                height, width, channels = shape[0], shape[1], 1
            else:
                raise DatasetError(
                    f"This model takes a {len(shape)}-dimensional input, so an image "
                    f"cannot be fed to it directly."
                )
            image = image.convert("L" if channels == 1 else "RGB")
            image = image.resize((width, height))
            array = np.asarray(image, dtype="float32") / 255.0
            if array.ndim == 2:
                array = array[..., None]
            return array.reshape((1, *shape))

        if modality is Modality.AUDIO:
            import soundfile

            audio, rate = soundfile.read(
                io.BytesIO(payload), dtype="float32", always_2d=True
            )
            flat = audio.mean(axis=1)

            if len(shape) == 1:
                # The model takes the waveform itself.
                needed = int(shape[0])
                if flat.size < needed:
                    flat = np.pad(flat, (0, needed - flat.size))
                seconds = needed / max(rate, 1)
                _note(
                    f"{needed:,} samples at {rate} Hz — the first {seconds:.2f}s of "
                    f"'{filename}', as the waveform."
                )
                return flat[:needed].reshape((1, *shape)).astype("float32")

            # Anything wider than a waveform is a time-frequency representation, and
            # reshaping raw samples into that block was a lie: it produced a picture
            # that looked like a spectrogram and meant nothing [E-041]. A real one is
            # computed instead, and every parameter used is stated, because a guessed
            # pipeline that differs from the one the model was trained on is a thing
            # the user has to be able to see and correct.
            return _audio_as_spectrogram(flat, int(rate), shape, filename)

        if modality is Modality.TEXT:
            text = payload.decode("utf-8", errors="replace")
            codes = np.array([ord(char) % 256 for char in text], dtype="float32")
            needed = int(np.prod(shape))
            if codes.size < needed:
                codes = np.pad(codes, (0, needed - codes.size))
            return codes[:needed].reshape((1, *shape))

        array = np.load(io.BytesIO(payload), allow_pickle=False)
        return np.asarray(array, dtype="float32").reshape((1, *shape))

    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(
            f"'{filename}' could not be read as this model's input "
            f"{list(shape)}: {exc}"
        ) from exc


def make_adapter(
    choice: DatasetChoice,
    input_shape: list[int],
    num_classes: int | None = None,
    *,
    payload: bytes = b"",
    filename: str = "",
) -> DatasetAdapter:
    """
    > [!AML-DOC-UNIT]
    Build the adapter a choice names.
    @param choice      the user's selection
    @param input_shape shape the model expects, excluding the batch axis
    @param num_classes number of outputs, when the model classifies
    @param payload     file bytes, for an upload
    @param filename    original name, for an upload
    @returns the adapter
    @raises DatasetError when the choice names a source that does not exist
    """
    if choice.kind is DatasetKind.BUILTIN:
        return BuiltinAdapter(choice, input_shape, num_classes)
    if choice.kind is DatasetKind.UPLOAD:
        return UploadAdapter(choice, input_shape, num_classes, payload, filename)
    if choice.kind is DatasetKind.FOLDER:
        return FolderAdapter(choice, input_shape, num_classes)
    return SyntheticAdapter(choice, input_shape, num_classes)
