import argparse
import wave

import numpy as np
import pyaudio

from common.log import debug

SAMPLE_FORMAT = pyaudio.paInt16
SPEED_OF_SOUND = 343.0


def generate_square_positions(num_mics, spacing_m):
    side = int(np.sqrt(num_mics))
    if side * side != num_mics:
        raise ValueError('num_mics must be a perfect square for square geometry')
    positions = []
    for i in range(side):
        for j in range(side):
            x = i * spacing_m
            y = j * spacing_m
            positions.append([x, y, 0.0])
    positions = np.array(positions, dtype=np.float32)
    positions -= positions.mean(axis=0)
    return positions


def gcc_phat(sig, refsig, fs, max_tau, interp=16):
    n = sig.size + refsig.size
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    cc = np.fft.irfft(R / denom, n=interp * n)

    max_shift = int(interp * fs * max_tau)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    shift = np.argmax(cc) - max_shift
    tau = shift / float(interp * fs)
    return tau


def azimuth_from_tdoa(tau, mic_distance):
    if mic_distance <= 0:
        return 0.0
    val = tau * SPEED_OF_SOUND / mic_distance
    val = float(np.clip(val, -1.0, 1.0))
    return float(np.degrees(np.arcsin(val)))


def steering_vector(freqs, positions, azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    direction = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
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


class FrostGSC:
    def __init__(self, n_channels, sample_rate, frame_len, hop, mu, reg, gain_db, limiter,
                 positions, doa_pairs, doa_update_every, doa_smoothing, max_doa_step, doa_log_every):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.hop = hop
        self.mu = mu
        self.reg = reg
        self.gain_linear = 10 ** (gain_db / 20.0)
        self.limiter = limiter
        self.positions = positions
        self.doa_pairs = doa_pairs
        self.doa_update_every = doa_update_every
        self.doa_smoothing = doa_smoothing
        self.max_doa_step = max_doa_step
        self.doa_log_every = doa_log_every

        self.n_fft = frame_len
        self.n_freqs = self.n_fft // 2 + 1
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

        self.window = np.hanning(frame_len).astype(np.float32)
        self.out_buffer = np.zeros(frame_len, dtype=np.float32)
        self.win_buffer = np.zeros(frame_len, dtype=np.float32)

        self.azimuth_deg = 0.0
        self.elevation_deg = 0.0
        self.steering = steering_vector(self.freqs, self.positions, self.azimuth_deg, self.elevation_deg)

        # Adaptive filters per frequency (same dimension as channels)
        self.g = np.zeros((self.n_freqs, self.n_channels), dtype=np.complex64)
        self.frame_idx = 0

    def _estimate_doa(self, frame):
        azimuths = []
        for i, j in self.doa_pairs:
            sig = frame[:, i].astype(np.float32)
            ref = frame[:, j].astype(np.float32)

            mic_distance = np.linalg.norm(self.positions[i] - self.positions[j])
            if mic_distance <= 0:
                continue
            max_tau = mic_distance / SPEED_OF_SOUND
            tau = gcc_phat(sig, ref, self.sample_rate, max_tau)
            azimuths.append(azimuth_from_tdoa(tau, mic_distance))

        if not azimuths:
            return

        new_az = float(np.median(azimuths))
        if self.max_doa_step is not None and self.max_doa_step > 0:
            delta = new_az - self.azimuth_deg
            delta = np.clip(delta, -self.max_doa_step, self.max_doa_step)
            new_az = self.azimuth_deg + delta

        self.azimuth_deg = (1.0 - self.doa_smoothing) * self.azimuth_deg + (self.doa_smoothing * new_az)
        self.elevation_deg = 0.0
        self.steering = steering_vector(self.freqs, self.positions, self.azimuth_deg, self.elevation_deg)

        if self.doa_log_every and (self.frame_idx % (self.doa_log_every * self.doa_update_every) == 0):
            debug('Estimated DOA', azimuth=f'{self.azimuth_deg:.1f}', samples=len(azimuths))

    def process_frame(self, frame):
        if self.frame_idx % self.doa_update_every == 0:
            self._estimate_doa(frame)

        frame_f = frame.astype(np.float32) * self.window[:, None]
        X = np.fft.rfft(frame_f, n=self.n_fft, axis=0)

        Y = np.zeros(self.n_freqs, dtype=np.complex64)
        for f in range(self.n_freqs):
            a = self.steering[f].reshape(-1, 1)
            ah_a = (a.conj().T @ a).real + 1e-12
            wq = (a / ah_a).reshape(-1)
            P = np.eye(self.n_channels, dtype=np.complex64) - (a @ a.conj().T) / ah_a

            x = X[f]
            u = P.conj().T @ x
            y = wq.conj().T @ x - self.g[f].conj().T @ u

            denom = (u.conj().T @ u).real + self.reg
            self.g[f] = self.g[f] + (self.mu / denom) * u * np.conj(y)
            Y[f] = y

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
    parser = argparse.ArgumentParser(description='Real-time Frost (GSC) beamformer with GCC-PHAT steering')
    parser.add_argument('--output', default='recorder_output/records/audio_frost.wav', help='Output WAV file path')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate in Hz')
    parser.add_argument('--frame-len', type=int, default=1024, help='STFT frame length in samples')
    parser.add_argument('--hop', type=int, default=512, help='STFT hop size in samples')
    parser.add_argument('--mu', type=float, default=0.05, help='NLMS step size')
    parser.add_argument('--reg', type=float, default=1e-3, help='NLMS regularization')
    parser.add_argument('--gain', type=float, default=6.0, help='Output gain in dB')
    parser.add_argument('--limiter', type=float, default=0.95, help='Limiter ceiling (0-1 of full scale)')
    parser.add_argument('--device-index', type=int, default=None, help='Optional input device index')
    parser.add_argument('--mic-spacing', type=float, default=0.042, help='Mic spacing in meters for 4x4 grid')
    parser.add_argument('--pair', type=str, default='0,3', help='Single mic pair indices (e.g. 0,3)')
    parser.add_argument('--pairs', type=str, default=None, help='Multiple pairs (e.g. 0,3;12,15)')
    parser.add_argument('--doa-update-every', type=int, default=8, help='Update DOA every N frames')
    parser.add_argument('--doa-smoothing', type=float, default=0.2, help='EMA smoothing for DOA updates (0-1)')
    parser.add_argument('--max-doa-step', type=float, default=8.0, help='Max DOA change per update (degrees)')
    parser.add_argument('--doa-log-every', type=int, default=0, help='Log DOA every N updates (0=off)')
    args = parser.parse_args()

    if args.hop <= 0 or args.hop > args.frame_len:
        raise ValueError('hop must be > 0 and <= frame-len')

    if args.pairs:
        pairs = []
        for item in args.pairs.split(';'):
            vals = [v.strip() for v in item.split(',') if v.strip()]
            if len(vals) != 2:
                raise ValueError('pairs must be like 0,3;12,15')
            pairs.append((int(vals[0]), int(vals[1])))
    else:
        pair = tuple(int(x.strip()) for x in args.pair.split(','))
        if len(pair) != 2:
            raise ValueError('pair must be two comma-separated indices, e.g. 0,3')
        pairs = [pair]

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

    beamformer = FrostGSC(
        n_channels=n_channels,
        sample_rate=args.sample_rate,
        frame_len=args.frame_len,
        hop=args.hop,
        mu=args.mu,
        reg=args.reg,
        gain_db=args.gain,
        limiter=args.limiter,
        positions=positions,
        doa_pairs=pairs,
        doa_update_every=args.doa_update_every,
        doa_smoothing=args.doa_smoothing,
        max_doa_step=args.max_doa_step,
        doa_log_every=args.doa_log_every,
    )

    debug('Frost GSC capture started', channels=n_channels, sample_rate=args.sample_rate)

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
            debug('Stopping Frost capture')
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()


if __name__ == '__main__':
    main()
