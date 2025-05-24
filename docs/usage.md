# Usage Examples

## Generate and play a sine wave

```python
from audio_signal import AudioSignal
sig = AudioSignal.wave(freq=440, time=2, sr=16000)
sig.play()
```

## Load from file

```python
sig = AudioSignal('audio.wav', sample_rate=16000)
```
