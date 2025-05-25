# audio_signal

A modern Python library for audio_signal processing, built on PyTorch and torchaudio.

## Features

-   **`AudioSignal` Class:** A specialized `torch.Tensor` subclass that incorporates `sample_rate` awareness, forming the core of the library.
-   **Versatile Input Options:** Load audio data from file paths, NumPy arrays, or raw PyTorch tensors.
-   **Sample Rate Management:** Automatic handling and propagation of sample rates through various operations.
-   **Resampling:** Easily change the sample rate of audio signals.
-   **Waveform Generation:** Create common waveforms, such as sine waves, using the `AudioSignal.wave()` method.
-   **IPython Integration:** Play audio directly within Jupyter notebooks or IPython environments using the `.play()` method.
-   **File Operations:** Save `AudioSignal` objects to audio files.
-   **Audio Manipulation:**
    -   Adjust playback speed.
    -   Perform convolution with custom kernels.
    -   Perform correlation with custom kernels.
-   **Seamless PyTorch Integration:**
    -   Use `AudioSignal` objects directly in most PyTorch operations.
    -   Compatible with `torch.nn.Module` for building audio processing pipelines.
-   **`CoreAudioSignal` Class:** Provides a lower-level API for more fine-grained control.

## Installation

You can install `audio_signal` using pip or uv. We recommend [uv](https://github.com/astral-sh/uv) for fast dependency management.

### From PyPI (Recommended)

To install the latest stable release from PyPI:

```bash
pip install audio-signal
```

Or with `uv`:

```bash
uv pip install audio-signal
```

### From GitHub (Latest Version)

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/SaarShafir/audio_signal.git
```

Or with `uv`:

```bash
uv pip install git+https://github.com/SaarShafir/audio_signal.git
```

### Development Setup

If you want to contribute to the development of `audio_signal`, you can set up a local development environment:

```bash
git clone https://github.com/SaarShafir/audio_signal.git
cd audio_signal
pip install -e .[dev]
```

Or with `uv`:

```bash
uv pip install -e .[dev]
```
This will install the package in editable mode and include development dependencies like `pytest`, `black`, and `flake8`.

### Prerequisites
- Python 3.8 or newer.
- PyTorch and torchaudio.

(The installation commands above will handle PyTorch and torchaudio installation if they are not already present).

## Key Concepts

### The `AudioSignal` and `CoreAudioSignal` Classes

The library revolves around two main classes:

-   **`CoreAudioSignal`**: This class is a low-level subclass of `torch.Tensor`. Its primary role is to bundle audio data (as a tensor) with its `sample_rate`. It forms the foundation for audio operations where sample rate awareness is crucial. It handles the logic for loading audio from various sources (files, NumPy arrays, tensors) and ensures that the sample rate is correctly initialized.

-   **`AudioSignal`**: This is a higher-level class that inherits from `CoreAudioSignal`. It provides a more user-friendly API with convenient methods for common audio tasks, such as:
    *   Generating standard waveforms (`.wave()`)
    *   Playing audio in IPython environments (`.play()`)
    *   Saving audio to files (`.save()`)
    *   Adjusting playback speed (`.speed()`)
    *   Performing convolutions (`.convolve()`) and correlations (`.correlate()`)

Most users will primarily interact with the `AudioSignal` class.

### Sample Rate (`sample_rate`)

The `sample_rate` attribute is fundamental to `audio_signal`. It represents the number of samples per second in the audio data.
-   **Initialization**: When you load or create an `AudioSignal`, the library attempts to infer the sample rate (e.g., from audio file metadata) or requires you to provide it.
-   **Propagation**: The library is designed to intelligently manage and propagate the `sample_rate` through various operations. When you apply PyTorch functions or `AudioSignal` methods, the resulting `AudioSignal` object will retain the correct `sample_rate` if the operation doesn't fundamentally change the time domain in a way that makes the original sample rate irrelevant (e.g. STFT).
-   **Resampling**: If you need to change the sample rate, you can use the `.resample()` method.

### PyTorch Integration (`__torch_function__`)

`AudioSignal` objects are designed to work seamlessly with most PyTorch functions. This is achieved through Python's `__torch_function__` protocol.
-   When you use a PyTorch function (e.g., `torch.mean(sig)`) with an `AudioSignal` instance, `AudioSignal` can intercept the call.
-   It then executes the original PyTorch operation.
-   After the operation, it re-wraps the resulting tensor as an `AudioSignal`, ensuring that the `sample_rate` (and other relevant metadata, if any) is preserved or correctly adjusted.
-   If an operation results in a tensor that can no longer be meaningfully represented as an audio signal with the original sample rate (e.g., it becomes complex-valued, or its dimensions change significantly like after an STFT), it will revert to a standard `torch.Tensor`.

This allows you to leverage the full power of PyTorch while still benefiting from the sample rate awareness and convenience methods of `AudioSignal`.

## Usage

Here are some examples demonstrating how to use `audio_signal`.

### Creating and Playing Audio

**Generate a sine wave:**

```python
from audio_signal import AudioSignal # Or from audio_signal.core import AudioSignal

# Generate a 2-second, 440 Hz sine wave at a 16kHz sample rate
sig = AudioSignal.wave(freq=440, time=2, sr=16000)

# In an IPython/Jupyter environment, you can play it directly:
sig.play()
```

### Loading and Saving Audio

**Load from a file:**

```python
# Assume 'audio.wav' exists and has a sample rate of 16000 Hz
# If sample_rate is not provided, the library attempts to infer it from the file.
# If it cannot be inferred, you must provide it.
try:
    sig_from_file = AudioSignal('audio.wav')
    print(f"Loaded audio with sample rate: {sig_from_file.sample_rate}")
except FileNotFoundError:
    print("Dummy file 'audio.wav' not found. Skipping load example.")
    # Create a dummy signal for subsequent examples if file doesn't exist
    sig_from_file = AudioSignal.wave(freq=100, time=1, sr=16000)


# If the sample rate in the file metadata differs from a provided sr,
# a warning is issued and the signal is resampled.
# sig_from_file_resampled = AudioSignal('audio.wav', sample_rate=44100)
```

**Save to a file:**

```python
# Save the (potentially dummy) signal to 'output.wav'
sig_from_file.save('output.wav')
print("Signal saved to 'output.wav'")
```

### Audio Manipulations

**Change playback speed:**
The `.speed()` method adjusts the `sample_rate` attribute, effectively changing the playback speed without resampling the data.

```python
# Create a clone to avoid modifying the original
sig_fast = sig.clone().speed(1.5)  # Play 1.5x faster
sig_slow = sig.clone().speed(0.75) # Play 0.75x speed (slower)

# sig_fast.play() # Plays faster
# sig_slow.play() # Plays slower
print(f"Original sample rate: {sig.sample_rate}, Fast sample rate: {sig_fast.sample_rate}")
```

**Convolution and Correlation:**
You can convolve or correlate the signal with a kernel (which should be a `torch.Tensor`).

```python
import torch

# Example: Simple moving average filter as a kernel
kernel_size = 100
kernel = torch.ones(1, kernel_size) / kernel_size # Kernel needs to be 1D or 2D (1, L) or (C, L)

# Ensure signal is at least 2D (1, L) for convolution if it's mono
mono_signal = sig.clone()
if mono_signal.ndim == 1:
    mono_signal = mono_signal.unsqueeze(0) # Add channel dimension: (L,) -> (1, L)


convolved_signal = mono_signal.convolve(kernel)
# correlated_signal = mono_signal.correlate(kernel) # correlate expects kernel to be 1D

# print("Convolved signal shape:", convolved_signal.shape)
# print("Correlated signal shape:", correlated_signal.shape)
# convolved_signal.play()
```
*Note: The exact shapes and behavior of convolution/correlation can depend on the kernel and signal dimensions, and PyTorch/torchaudio's underlying implementations.*


### Seamless PyTorch Integration

A key strength of `AudioSignal` is its compatibility with PyTorch.

**`AudioSignal` *is* a `torch.Tensor`:**
You can often use `AudioSignal` instances wherever you would use a `torch.Tensor`.

```python
from matplotlib import pyplot as plt
import torch # ensure torch is imported

# Create two sine waves
a = AudioSignal.wave(freq=1000, time=1, sr=8000) # 1kHz tone
b = AudioSignal.wave(freq=2000, time=1, sr=8000) # 2kHz tone

# Concatenate them using a standard PyTorch operation
c = torch.hstack([a, b])

print(f"Sample rate of concatenated signal: {c.sample_rate}") # Still 8000 Hz

# Compute STFT and plot spectrogram (STFT result is a plain tensor)
# Ensure signal 'c' is 1D or 2D for STFT
if c.ndim == 2 and c.shape[0] == 1: # If (1, L), squeeze to (L,)
    c_stft = c.squeeze(0)
else:
    c_stft = c

spectrogram = torch.stft(c_stft, n_fft=256, hop_length=64, return_complex=True).abs()

# # Plotting (optional, requires matplotlib)
# plt.figure(figsize=(6, 4))
# plt.imshow(spectrogram.log1p(), aspect='auto', origin='lower') # Use log1p for better visualization
# plt.xlabel("Time Frames")
# plt.ylabel("Frequency Bins")
# plt.title("Spectrogram of Concatenated Signals")
# plt.show()
# # To display this image in the README, we'll link to the existing one:
```
![Spectrogram Example](docs/assets/spectrogram.png)
*The image above shows a spectrogram generated from concatenated audio signals.*

**Use with `torch.nn.Module`:**
`AudioSignal` instances can be passed directly to PyTorch neural network modules.

```python
from torch import nn

# Define a simple 1D convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        # Ensure input x has a channel dimension
        if x.ndim == 1: # (L,)
            x = x.unsqueeze(0) # (1, L)
        if x.ndim == 2 and x.shape[0] != 1 and x.shape[1] == 1 : # (L, 1)
             x = x.T # (1,L)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Create a model instance
model = ConvBlock()

# Create a dummy audio signal
input_signal = AudioSignal.wave(freq=500, time=1, sr=8000) # 1 second, 500Hz, 8kHz SR

# Pass the signal (potentially unsqueezed) to the model
# The model expects input like (Batch, Channels, Length)
# Our signal is (Channels, Length) or (Length,)
# If mono (Length,), add Batch and Channel: input_signal.unsqueeze(0).unsqueeze(0)
# If mono (Channels, Length) e.g. (1,L), add Batch: input_signal.unsqueeze(0)

processed_signal_tensor = model(input_signal.unsqueeze(0)) # Add batch dimension

print(f"Original signal sample rate: {input_signal.sample_rate}")
print(f"Shape of output tensor from model: {processed_signal_tensor.shape}")

# Note: The output of the nn.Module is a plain torch.Tensor.
# If you need to convert it back to an AudioSignal, you'd do:
# processed_audio_signal = AudioSignal(processed_signal_tensor.squeeze(0), sample_rate=input_signal.sample_rate / pool_size)
# The sample rate would change due to operations like pooling.
# Here, MaxPool1d(2) halves the length, so effective sample rate is also halved.
new_sr = input_signal.sample_rate / 2 # Due to MaxPool1d(2)
processed_audio_signal = AudioSignal(processed_signal_tensor.squeeze(0), sample_rate=new_sr)
print(f"Processed audio signal sample rate: {processed_audio_signal.sample_rate}")

```

These examples cover the basic functionalities. Explore the methods of the `AudioSignal` class for more advanced operations.

## API Overview

The `audio_signal` library primarily exposes two classes:

-   **`audio_signal.core.CoreAudioSignal`**: A lower-level class that directly subclasses `torch.Tensor`. It's responsible for the fundamental association of a sample rate with tensor-based audio data and handles loading from various sources.
-   **`audio_signal.core.AudioSignal` (or `audio_signal.AudioSignal`)**: The main high-level class, inheriting from `CoreAudioSignal`. This class provides a suite of convenient methods for common audio processing tasks such as waveform generation, playback, file I/O, and signal manipulation. Most users will interact primarily with `AudioSignal`.

Key methods and properties to look out for in the `AudioSignal` class include:
-   `sample_rate` (property): Access or (in some cases, like with `.speed()`) modify the sample rate.
-   `wave()`: Class method to generate waveforms.
-   `play()`: For playback in IPython/Jupyter.
-   `save()`: To write audio to a file.
-   `load()`: (Handled by constructor) For loading audio from files or other sources.
-   `resample()`: To change the sample rate of the audio data.
-   `convolve()`, `correlate()`: For convolution and correlation operations.
-   `speed()`: To adjust playback speed by modifying the sample rate.

For a complete and detailed API reference, including all methods, parameters, and their descriptions, please see our [Full API Documentation](docs/api.md) (Note: This links to the markdown file in the repository; a hosted version may be available elsewhere if the project uses services like ReadTheDocs or GitHub Pages with MkDocs).

## Contributing

Contributions are welcome! We appreciate your help in improving `audio_signal`.

### Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/YourUsername/audio_signal.git
    cd audio_signal
    ```
3.  **Set up a development environment** (see [Development Setup](#development-setup) under Installation). This will install development tools like `pytest`, `black`, and `flake8`.
4.  **Create a new branch** for your changes:
    ```bash
    git checkout -b my-awesome-feature
    ```

### Making Changes

-   Write clear and concise code.
-   Ensure your changes are well-tested. Add new tests for new features or bug fixes.
-   Run tests using `pytest`:
    ```bash
    pytest
    ```
-   Format your code with Black:
    ```bash
    black .
    ```
-   Check for linting issues with Flake8:
    ```bash
    flake8 .
    ```
-   Ensure all tests and checks pass before submitting a pull request.

### Submitting a Pull Request

1.  Push your changes to your fork on GitHub.
2.  Submit a pull request to the `main` branch of the original `SaarShafir/audio_signal` repository.
3.  Provide a clear description of your changes in the pull request.

### Reporting Issues

If you encounter any bugs or have suggestions for improvements, please open an issue on the [GitHub Issues page](https://github.com/SaarShafir/audio_signal/issues). Provide as much detail as possible, including steps to reproduce the issue if it's a bug.

## License

The `audio_signal` library is currently pending formal license assignment.

It is common for open-source Python projects to adopt permissive licenses like the MIT License or Apache License 2.0. Please check back later or consult the project maintainers for specific license information.

If you are the project owner, you can add a `LICENSE` file to the repository (e.g., by choosing a license template on GitHub) and update this section accordingly.
