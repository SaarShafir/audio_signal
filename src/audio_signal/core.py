import torch
import torchaudio as ta
import numpy as np
from typing import Union
from IPython.display import Audio
import warnings

class CoreAudioSignal(torch.Tensor):
    """
    A low-level subclass of torch.Tensor that represents an audio signal with an attached sample rate.
    Supports initialization from various formats (file path, numpy array, torch tensor)
    """
    def __new__(cls,
                audio_source: Union[torch.Tensor, np.ndarray, str, bytes],
                sample_rate: int = None):
        """
        Create a new CoreAudioSignal instance from various audio sources.

        Args:
            audio_source (Union[torch.Tensor, np.ndarray, str, bytes]): The audio data source.
            sample_rate (int, optional): The sample rate of the audio signal.

        Returns:
            CoreAudioSignal: An instance with audio data and sample rate.
        """

        if isinstance(audio_source, cls):
            return audio_source

        inferred_sr = None
        if isinstance(audio_source, (str, bytes)):
            audio_source, inferred_sr = ta.load(audio_source)

        elif isinstance(audio_source, np.ndarray):
            audio_source = torch.from_numpy(audio_source)

        elif isinstance(audio_source, torch.Tensor):
            if sample_rate is None and hasattr(audio_source, '_sample_rate'):
                sample_rate = audio_source._sample_rate
        else:
            raise TypeError(f'Unsupported input data type: {type(audio_source)}')

        if sample_rate is None:
            if inferred_sr is None:
                raise ValueError('Sample rate must be provided')
            else:
              sample_rate = inferred_sr

        obj = torch.Tensor(audio_source).as_subclass(cls)
        obj._sample_rate = sample_rate

        if inferred_sr is not None and inferred_sr!=sample_rate:
            warnings.warn(f'Inferred sample rate {inferred_sr} does not match provided sample rate {sample_rate}')
            obj = obj.resample(sample_rate)

        return obj

    @property
    def sample_rate(self):
        """
        Returns the sample rate of the audio signal.

        Returns:
            int: The sample rate.
        """
        return self._sample_rate


    def resample(self, new_sample_rate):
        """
        Resample the audio signal to a new sample rate.

        Args:
            new_sample_rate (int): The desired sample rate.

        Returns:
            CoreAudioSignal: A new instance with the resampled audio.
        """
        result =  ta.transforms.Resample(self.sample_rate, new_sample_rate)(self)
        return type(self)(result,new_sample_rate)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        result = super().__torch_function__(func, types, args, kwargs)

        # This is a tricky part - Based on the result of the function we
        # should decide whether to return a CoreAudioSignal or not.

        if not isinstance(result, CoreAudioSignal):
            return result

        if result.is_complex() or result.ndim not in (1,2):
            return result.as_subclass(torch.Tensor)

        if result.ndim == 2 and result.shape[0] not in (1,2):
            return result.as_subclass(torch.Tensor)

        # find a source with sample rate
        def find_sr(obj):
            return getattr(obj, '_sample_rate', None)

        # look in args
        sr = None
        all_args = list(args)
        if args and isinstance(args[0], (tuple, list)):
            all_args += list(args[0])

        for arg in all_args:
            if isinstance(arg, CoreAudioSignal):
                sr = find_sr(arg)
                if sr is not None:
                    break

        # propagate sample_rate if result is tensor-like
        if sr is None:
          return result.as_subclass(torch.Tensor)

        result._sample_rate = sr
        return result

    def _from_tensor(self, tensor):
        return type(self)(tensor,self._sample_rate)

    def clone(self):
        """
        Return a copy of the audio signal.

        Returns:
            CoreAudioSignal: A cloned instance.
        """
        return self._from_tensor(super().clone())

    def __repr__(self):
        return super().__repr__() + f', sample_rate={self.sample_rate}'

class AudioSignal(CoreAudioSignal):
    """ 
    A High-level audio signal class that inherits from CoreAudioSignal.
    Provides additional methods for common audio operations like playback, saving, and generating waveforms.
    """

    @classmethod
    def wave(cls, freq, time, sr):
        """
        Generate a sine wave AudioSignal.

        Args:
            freq (float): Frequency of the sine wave in Hz.
            time (float): Duration in seconds.
            sr (int): Sample rate.

        Returns:
            AudioSignal: Generated sine wave.
        """
        t = torch.linspace(0,time,sr*time)
        wf = torch.sin(2*np.pi*freq*t)
        return cls(wf.unsqueeze(0),sr)

    def play(self):
        """
        Play the audio signal in a Jupyter/IPython environment.

        Returns:
            IPython.display.Audio: Audio widget for playback.
        """
        return Audio(self,rate=self.sample_rate)

    def save(self, path):
        """
        Save the audio signal to a file.

        Args:
            path (str): Path to save the audio file.
        """
        ta.save(path,self,self.sample_rate)

    def speed(self, factor: float):
        """
        Change the playback speed by adjusting the sample rate.

        Args:
            factor (float): Speed factor (e.g., 2.0 is double speed).

        Returns:
            AudioSignal: The modified instance (in-place).
        """
        self._sample_rate = int(self._sample_rate*factor)
        return self

    def convolve(self, kernel):
        """
        Convolve the audio signal with a kernel.

        Args:
            kernel (torch.Tensor): The convolution kernel.

        Returns:
            AudioSignal: The convolved signal.
        """
        return self._from_tensor(ta.functional.convolve(self,kernel))
    
    def correlate(self, kernel):
        """
        Correlate the audio signal with a kernel.

        Args:
            kernel (torch.Tensor): The correlation kernel.

        Returns:
            AudioSignal: The correlated signal.
        """
        return self._from_tensor(ta.functional.corr(self,kernel))
