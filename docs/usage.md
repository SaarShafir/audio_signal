# Usage Examples

## Generate and play a sine wave

```python
from audio_signal.core import AudioSignal
sig = AudioSignal.wave(freq=440, time=2, sr=16000)
sig.play()
```

## Load from file

```python
from audio_signal.core import AudioSignal
sig = AudioSignal('audio.wav', sample_rate=16000)
```

## Save to file

```python
sig.save('output.wav')
```

## Change playback speed

```python
sig_fast = sig.clone().speed(1.5)  # 1.5x faster
sig_fast.play()
```

## Convolve and correlate

```python
import torch
kernel = torch.ones(1, 100) / 100  # simple moving average filter

convolved = sig.convolve(kernel)
correlated = sig.correlate(kernel)
```
