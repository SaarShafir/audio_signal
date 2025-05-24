# audio_signal

A modern Python library for audio signal processing, built on PyTorch and torchaudio.

## Features
- Audio signal class with sample rate awareness
- File, numpy, and tensor input support
- Resampling, convolution, and correlation utilities
- Easy playback and saving

## Installation

```bash
uv pip install audio_signal
```

## Usage

```python
from audio_signal import AudioSignal

# Generate a sine wave and play it
sig = AudioSignal.wave(freq=440, time=2, sr=16000)
sig.play()
```

See [docs/usage.md](docs/usage.md) for more examples.

# ...no changes to README.md, just shell commands below...

git add .
git commit -m "Update docs and theme; improve usage and appearance"
git push