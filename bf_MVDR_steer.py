import argparse
import wave

import numpy as np
import pyaudio

from common.log import debug

SAMPLE_FORMAT = pyaudio.paInt16
SPEED_OF_SOUND = 343.0


def generate_square_positions(num_mics, spacing_m):
    """
    Generate UMA16 microphone positions.
    Channel-to-MIC mapping: Channel N corresponds to MIC(N+1)
    Physical layout (4x4 grid, 42mm spacing):
        Front (0°):        MIC8(0,3)   MIC7(1,3)   MIC10(2,3)  MIC9(3,3)
        Left (-90°):       MIC6(0,2)   MIC5(1,2)   MIC12(2,2)  MIC11(3,2)
        Right (+90°):      MIC4(0,1)   MIC3(1,1)   MIC14(2,1)  MIC13(3,1)
        Back (180°):       MIC2(0,0)   MIC1(1,0)   MIC16(2,0)  MIC15(3,0)
    """
    if num_mics != 16:
        raise ValueError('This function is designed for UMA16 (16 mics)')
    
    # Channel to (x_idx, y_idx) mapping based on confirmed 1:1 channel-to-MIC layout
    # Channel 0 = MIC1, Channel 1 = MIC2, etc.
    channel_positions_idx = [
        (1, 0),  # Channel 0 = MIC1
        (0, 0),  # Channel 1 = MIC2
        (1, 1),  # Channel 2 = MIC3
        (0, 1),  # Channel 3 = MIC4
        (1, 2),  # Channel 4 = MIC5
        (0, 2),  # Channel 5 = MIC6
        (1, 3),  # Channel 6 = MIC7
        (0, 3),  # Channel 7 = MIC8
        (3, 3),  # Channel 8 = MIC9
        (2, 3),  # Channel 9 = MIC10
        (3, 2),  # Channel 10 = MIC11
        (2, 2),  # Channel 11 = MIC12
        (3, 1),  # Channel 12 = MIC13
        (2, 1),  # Channel 13 = MIC14
        (3, 0),  # Channel 14 = MIC15
        (2, 0),  # Channel 15 = MIC16
    ]
    
    positions = []
    for x_idx, y_idx in channel_positions_idx:
        x = x_idx * spacing_m
        y = y_idx * spacing_m
        positions.append([x, y, 0.0])
    
    positions = np.array(positions, dtype=np.float32)
    positions -= positions.mean(axis=0)
    return positions


def steering_vector(freqs, positions, azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    # Coordinate system: 0°=front(+y), 90°=right(+x), 180°=back(-y), -90°=left(-x)
    direction = np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el)
    ], dtype=np.float32)
    delays = -np.dot(positions, direction) / SPEED_OF_SOUND

    sv = np.exp(-1j * 2 * np.pi * freqs[:, None] * delays[None, :])
    return sv


class SteeredMVDR:
    def __init__(self, n_channels, sample_rate, frame_len, hop, alpha, reg, ref_channel,
                 update_every, gain_db, positions, azimuth_deg, elevation_deg):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.hop = hop
        self.alpha = alpha
        self.reg = reg
        self.ref_channel = ref_channel
        self.update_every = update_every
        self.gain_linear = 10 ** (gain_db / 20.0)
        self.positions = positions
        self.azimuth_deg = azimuth_deg
        self.elevation_deg = elevation_deg

        self.n_fft = frame_len
        self.n_freqs = self.n_fft // 2 + 1
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

        self.window = np.hanning(frame_len).astype(np.float32)
        self.out_buffer = np.zeros(frame_len, dtype=np.float32)
        self.win_buffer = np.zeros(frame_len, dtype=np.float32)

        self.R = np.zeros((self.n_freqs, n_channels, n_channels), dtype=np.complex64)
        self.weights = np.zeros((self.n_freqs, n_channels), dtype=np.complex64)
        self.frame_idx = 0

        self.steering = steering_vector(self.freqs, self.positions, self.azimuth_deg, self.elevation_deg)

    def _update_weights(self):
        for f in range(self.n_freqs):
            Rf = self.R[f]
            trace = np.trace(Rf).real
            Rf = Rf + (self.reg * trace / self.n_channels + 1e-12) * np.eye(self.n_channels)

            d = self.steering[f]
            R_inv = np.linalg.pinv(Rf)
            denom = (d.conj().T @ R_inv @ d)
            if np.abs(denom) < 1e-12:
                self.weights[f] = np.zeros(self.n_channels, dtype=np.complex64)
            else:
                self.weights[f] = (R_inv @ d) / denom

    def process_frame(self, frame):
        frame_f = frame.astype(np.float32) * self.window[:, None]
        X = np.fft.rfft(frame_f, n=self.n_fft, axis=0)

        for f in range(self.n_freqs):
            x = X[f].reshape(-1, 1)
            self.R[f] = self.alpha * self.R[f] + (1.0 - self.alpha) * (x @ x.conj().T)

        if self.frame_idx % self.update_every == 0:
            self._update_weights()

        Y = np.sum(self.weights.conj() * X, axis=1)
        y_frame = np.fft.irfft(Y, n=self.n_fft).astype(np.float32)

        self.frame_idx += 1
        return y_frame

    def overlap_add(self, y_frame):
        self.out_buffer += y_frame * self.window
        self.win_buffer += self.window ** 2

        out_chunk = self.out_buffer[:self.hop].copy()
        win_chunk = self.win_buffer[:self.hop].copy()

        valid = win_chunk > 1e-6
        out_chunk[valid] /= win_chunk[valid]

        self.out_buffer = np.roll(self.out_buffer, -self.hop)
        self.win_buffer = np.roll(self.win_buffer, -self.hop)
        self.out_buffer[-self.hop:] = 0.0
        self.win_buffer[-self.hop:] = 0.0

        if self.gain_linear != 1.0:
            out_chunk *= self.gain_linear

        out_chunk = np.clip(out_chunk, -32768, 32767)
        return out_chunk.astype(np.int16)


def select_input_device(pyaudio_instance, device_index=None):
    if device_index is not None:
        dev = pyaudio_instance.get_device_info_by_index(device_index)
        if dev['maxInputChannels'] < 1:
            raise ValueError(f'Device index {device_index} has no input channels')
        debug('Using audio device', index=device_index, name=dev['name'], in_channels=dev['maxInputChannels'])
        return device_index, int(dev['maxInputChannels'])

    max_channels = 0
    max_channels_device_index = None
    for i in range(pyaudio_instance.get_device_count()):
        dev = pyaudio_instance.get_device_info_by_index(i)
        input_channels = dev['maxInputChannels']
        debug('Listing audio device', index=i, name=dev['name'], in_channels=input_channels)
        if input_channels > max_channels:
            max_channels = input_channels
            max_channels_device_index = i

    if max_channels_device_index is None:
        raise RuntimeError('No input device found')

    debug('Selected audio device', index=max_channels_device_index, in_channels=max_channels)
    return max_channels_device_index, max_channels


def main():
    parser = argparse.ArgumentParser(description='Real-time MVDR with explicit steering')
    parser.add_argument('--output', default='recorder_output/records/audio_mvdr_steer.wav', help='Output WAV file path')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate in Hz')
    parser.add_argument('--frame-len', type=int, default=1024, help='STFT frame length in samples')
    parser.add_argument('--hop', type=int, default=512, help='STFT hop size in samples')
    parser.add_argument('--alpha', type=float, default=0.9, help='Covariance smoothing factor (0-1)')
    parser.add_argument('--reg', type=float, default=1e-3, help='Diagonal loading regularization')
    parser.add_argument('--ref-channel', type=int, default=0, help='Reference channel index for phase')
    parser.add_argument('--update-every', type=int, default=2, help='Update MVDR weights every N frames')
    parser.add_argument('--device-index', type=int, default=None, help='Optional input device index')
    parser.add_argument('--gain', type=float, default=12.0, help='Output gain in dB')
    parser.add_argument('--mic-spacing', type=float, default=0.042, help='Mic spacing in meters for 4x4 grid')
    parser.add_argument('--azimuth', type=float, default=0.0, help='Steering azimuth in degrees')
    parser.add_argument('--elevation', type=float, default=0.0, help='Steering elevation in degrees')
    args = parser.parse_args()

    if args.hop <= 0 or args.hop > args.frame_len:
        raise ValueError('hop must be > 0 and <= frame-len')

    pa = pyaudio.PyAudio()
    device_index, n_channels = select_input_device(pa, args.device_index)

    positions = generate_square_positions(n_channels, args.mic_spacing)

    stream = pa.open(
        format=SAMPLE_FORMAT,
        channels=n_channels,
        rate=args.sample_rate,
        frames_per_buffer=args.hop,
        input=True,
        input_device_index=device_index,
    )

    beamformer = SteeredMVDR(
        n_channels=n_channels,
        sample_rate=args.sample_rate,
        frame_len=args.frame_len,
        hop=args.hop,
        alpha=args.alpha,
        reg=args.reg,
        ref_channel=args.ref_channel,
        update_every=args.update_every,
        gain_db=args.gain,
        positions=positions,
        azimuth_deg=args.azimuth,
        elevation_deg=args.elevation,
    )

    debug('MVDR steered capture started', channels=n_channels, sample_rate=args.sample_rate,
          azimuth=args.azimuth, elevation=args.elevation)

    with wave.open(args.output, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(args.sample_rate)

        try:
            buffer = np.zeros((0, n_channels), dtype=np.int16)
            while True:
                data = stream.read(args.hop, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16).reshape(-1, n_channels)
                buffer = np.vstack([buffer, samples])

                while buffer.shape[0] >= args.frame_len:
                    frame = buffer[:args.frame_len]
                    buffer = buffer[args.hop:]
                    y_frame = beamformer.process_frame(frame)
                    out_chunk = beamformer.overlap_add(y_frame)
                    wf.writeframes(out_chunk.tobytes())
        except KeyboardInterrupt:
            debug('Stopping MVDR steered capture')
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()


if __name__ == '__main__':
    main()
