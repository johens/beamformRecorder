import os
import wave

import pyaudio
import numpy as np

from common.consts import RECORDS_PATH, WAV_FILE_NAME
from common.log import debug
from common.process_sync import should_stop
from audio.beamformer import DelayAndSumBeamformer

FRAME_RATE = 16000

SAMPLE_FORMAT = pyaudio.paInt16


class MicArray:
    def __init__(self, output_path=None, rate=FRAME_RATE, chunk_size=None, use_beamforming=False,
                 azimuth_deg=0, elevation_deg=0, mic_spacing=0.02, gain_db=0):
        self.pyaudio_instance = None
        self.stream = None
        self.channels = None
        self.sample_rate = rate
        self.chunk_size = chunk_size if chunk_size else None
        self.use_beamforming = use_beamforming
        self.beamformer = None
        self.azimuth_deg = azimuth_deg
        self.elevation_deg = elevation_deg
        self.mic_spacing = mic_spacing
        self.gain_db = gain_db
        self.gain_linear = 10 ** (gain_db / 20.0)  # Convert dB to linear
        os.makedirs(RECORDS_PATH, exist_ok=True)
        self._output_path = output_path if output_path else os.path.join(RECORDS_PATH, WAV_FILE_NAME)
        self.frames = []
        
        if gain_db != 0:
            debug('Audio gain', gain_db=gain_db, gain_linear=f'{self.gain_linear:.2f}x')

    def _select_mic_device_index(self):
        max_channels = 0
        max_channels_device_index = None
        for i in range(self.pyaudio_instance.get_device_count()):
            dev = self.pyaudio_instance.get_device_info_by_index(i)
            name = dev['name'].encode('utf-8')
            input_channels = dev['maxInputChannels']
            debug(
                'Listing audio device',
                index=i,
                name=name,
                in_channels=input_channels
            )
            if input_channels > max_channels:
                max_channels = input_channels
                max_channels_device_index = i
        if max_channels_device_index is None:
            raise Exception('can not find input device')
        self.channels = max_channels
        self.chunk_size = self.chunk_size if self.chunk_size else self.sample_rate // self.channels
        debug('Audio device', channels=max_channels)
        return max_channels_device_index

    def _apply_gain_and_clip(self, audio_data):
        """Apply gain and clip to prevent distortion."""
        if self.gain_linear == 1.0:
            return audio_data
        
        # Apply gain
        audio_float = audio_data.astype(np.float32) * self.gain_linear
        
        # Clip to int16 range to prevent distortion
        audio_float = np.clip(audio_float, -32768, 32767)
        
        return audio_float.astype(np.int16)

    def run(self):
        self.frames = []
        self.pyaudio_instance = pyaudio.PyAudio()
        device_index = self._select_mic_device_index()
        
        # Initialize beamformer if requested
        if self.use_beamforming:
            self.beamformer = DelayAndSumBeamformer(
                num_mics=self.channels,
                sample_rate=self.sample_rate,
                azimuth_deg=self.azimuth_deg,
                elevation_deg=self.elevation_deg,
                mic_spacing=self.mic_spacing
            )
            debug('Beamforming enabled', azimuth=self.azimuth_deg, elevation=self.elevation_deg)
        
        self.stream = self.pyaudio_instance.open(
            format=SAMPLE_FORMAT,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_size,
            input=True,
            input_device_index=device_index,
        )

        debug('Recording audio')

        frames = []  # Initialize array to store frames

        try:
            while True:
                data = self.stream.read(self.chunk_size)
                
                if self.use_beamforming:
                    # Convert byte data to numpy array
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    # Reshape to (chunk_size, channels)
                    audio_data = audio_data.reshape(-1, self.channels)
                    # Apply beamforming
                    beamformed = self.beamformer.process_frame(audio_data)
                    # Apply gain
                    beamformed = self._apply_gain_and_clip(beamformed)
                    # Convert back to bytes
                    frames.append(beamformed.tobytes())
                else:
                    # For multichannel audio, apply gain to each frame
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    audio_data = self._apply_gain_and_clip(audio_data)
                    frames.append(audio_data.tobytes())
                
                if should_stop():
                    raise KeyboardInterrupt()
        except KeyboardInterrupt:
            debug('Quitting audio recording')

        # Stop and close the stream
        self.stream.stop_stream()
        self.stream.close()
        # Terminate the PortAudio interface
        self.pyaudio_instance.terminate()

        debug('Finished recording audio')

        # Save the recorded data as a WAV file
        wf = wave.open(self._output_path, 'wb')
        
        # When beamforming, output is mono (1 channel)
        if self.use_beamforming:
            wf.setnchannels(1)
        else:
            wf.setnchannels(self.channels)
        
        wf.setsampwidth(self.pyaudio_instance.get_sample_size(SAMPLE_FORMAT))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()


def audio_capture(use_beamforming=False, azimuth_deg=0, elevation_deg=0, gain_db=0):
    mic = MicArray(use_beamforming=use_beamforming, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, gain_db=gain_db)
    mic.run()

