# python bf_LCMV_realtime.py --output recorder_output\records\audio_lcmv.wav --target-azimuth 0 --null-azimuth 180 --gain 30  

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
    return np.exp(-1j * 2 * np.pi * freqs[:, None] * delays[None, :])


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


class LCMV:
    def __init__(self, n_channels, sample_rate, frame_len, hop, reg, gain_db, limiter,
                 positions, target_az, target_el, null_az, null_el):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.hop = hop
        self.reg = reg
        self.gain_linear = 10 ** (gain_db / 20.0)
        self.limiter = limiter
        self.positions = positions
        self.target_az = target_az
        self.target_el = target_el
        self.null_az = null_az
        self.null_el = null_el

        self.n_fft = frame_len
        self.n_freqs = self.n_fft // 2 + 1
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

        self.window = np.hanning(frame_len).astype(np.float32)
        self.out_buffer = np.zeros(frame_len, dtype=np.float32)
        self.win_buffer = np.zeros(frame_len, dtype=np.float32)

        # Steering vectors
        self.a_target = steering_vector(self.freqs, self.positions, self.target_az, self.target_el)
        self.a_null = steering_vector(self.freqs, self.positions, self.null_az, self.null_el)

        # Weights per frequency
        self.weights = np.zeros((self.n_freqs, n_channels), dtype=np.complex64)
        self._compute_weights()

        self.frame_idx = 0

    def _compute_weights(self):
        for f in range(self.n_freqs):
            # Target and null steering vectors
            a_t = self.a_target[f].reshape(-1, 1)
            a_n = self.a_null[f].reshape(-1, 1)

            # Normalize steering vectors
            a_t = a_t / (np.linalg.norm(a_t) + 1e-10)
            a_n = a_n / (np.linalg.norm(a_n) + 1e-10)

            # Constraint matrix: [a_target | a_null]
            C = np.hstack([a_t, a_n])
            
            # Desired response: [1, 0] (keep target, null constraint)
            d = np.array([[1.0], [0.0]], dtype=np.complex64)

            # Compute regularization adaptively based on frequency
            freq_dependent_reg = self.reg * (1.0 + f / float(self.n_freqs))
            
            # Identity + regularization
            I = np.eye(self.n_channels, dtype=np.complex64)
            R = I * (freq_dependent_reg + 1e-10)

            # Compute LCMV weights with better numerical stability
            try:
                # Solve: w = R^-1 * C * (C^H * R^-1 * C)^-1 * d
                R_inv = np.linalg.pinv(R, rcond=1e-6)
                G = C.conj().T @ R_inv @ C
                G_inv = np.linalg.pinv(G, rcond=1e-6)
                numerator = R_inv @ C @ G_inv @ d
                
                # Normalize weights to prevent extreme values
                w_norm = np.linalg.norm(numerator)
                if w_norm > 1e-10:
                    self.weights[f] = (numerator / w_norm).reshape(-1)
                else:
                    self.weights[f] = np.zeros(self.n_channels, dtype=np.complex64)
            except (np.linalg.LinAlgError, ValueError):
                self.weights[f] = np.zeros(self.n_channels, dtype=np.complex64)

    def process_frame(self, frame):
        frame_f = frame.astype(np.float32) * self.window[:, None]
        X = np.fft.rfft(frame_f, n=self.n_fft, axis=0)

        Y = np.zeros(self.n_freqs, dtype=np.complex64)
        for f in range(self.n_freqs):
            Y[f] = self.weights[f].conj().T @ X[f]

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

        peak = np.max(np.abs(out_chunk))
        if peak > 0:
            limit_val = self.limiter * 32767.0
            if peak > limit_val:
                out_chunk *= (limit_val / peak)

        out_chunk = np.clip(out_chunk, -32768, 32767)
        return out_chunk.astype(np.int16)


def main():
    parser = argparse.ArgumentParser(description='Real-time LCMV beamformer with explicit target+null constraints')
    parser.add_argument('--output', default='recorder_output/records/audio_lcmv.wav', help='Output WAV file path')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate in Hz')
    parser.add_argument('--frame-len', type=int, default=1024, help='STFT frame length in samples')
    parser.add_argument('--hop', type=int, default=512, help='STFT hop size in samples')
    parser.add_argument('--reg', type=float, default=1e-1, help='Regularization strength (try 0.05-0.2 for stability)')
    parser.add_argument('--gain', type=float, default=18.0, help='Output gain in dB')
    parser.add_argument('--limiter', type=float, default=0.95, help='Limiter ceiling (0-1 of full scale)')
    parser.add_argument('--device-index', type=int, default=None, help='Optional input device index')
    parser.add_argument('--mic-spacing', type=float, default=0.042, help='Mic spacing in meters for 4x4 grid')
    parser.add_argument('--target-azimuth', type=float, default=-90.0, help='Target source azimuth (degrees)')
    parser.add_argument('--target-elevation', type=float, default=0.0, help='Target source elevation (degrees)')
    parser.add_argument('--null-azimuth', type=float, default=90.0, help='Null constraint azimuth (degrees)')
    parser.add_argument('--null-elevation', type=float, default=0.0, help='Null constraint elevation (degrees)')
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

    beamformer = LCMV(
        n_channels=n_channels,
        sample_rate=args.sample_rate,
        frame_len=args.frame_len,
        hop=args.hop,
        reg=args.reg,
        gain_db=args.gain,
        limiter=args.limiter,
        positions=positions,
        target_az=args.target_azimuth,
        target_el=args.target_elevation,
        null_az=args.null_azimuth,
        null_el=args.null_elevation,
    )

    debug('LCMV capture started', channels=n_channels, sample_rate=args.sample_rate,
          target=f'{args.target_azimuth}°/{args.target_elevation}°',
          null=f'{args.null_azimuth}°/{args.null_elevation}°')

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
            debug('Stopping LCMV capture')
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()


if __name__ == '__main__':
    main()
