import argparse
import wave

import numpy as np


def read_wav(path):
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError('Only 16-bit PCM WAV is supported')
    data = np.frombuffer(raw, dtype=np.int16).reshape(-1, n_channels)
    return data, sample_rate


def write_wav(path, data, sample_rate):
    data = np.clip(data, -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def principal_eigenvector(matrix):
    vals, vecs = np.linalg.eigh(matrix)
    return vecs[:, np.argmax(vals)]


def mvdr_beamform(data, sample_rate, frame_len=1024, hop=512, alpha=0.9, reg=1e-3, ref_channel=0, update_every=4):
    n_samples, n_channels = data.shape
    n_fft = frame_len
    n_freqs = n_fft // 2 + 1

    window = np.hanning(frame_len).astype(np.float32)
    out = np.zeros(n_samples + frame_len, dtype=np.float32)
    win_sum = np.zeros(n_samples + frame_len, dtype=np.float32)

    R = np.zeros((n_freqs, n_channels, n_channels), dtype=np.complex64)
    weights = np.zeros((n_freqs, n_channels), dtype=np.complex64)

    frame_idx = 0
    for start in range(0, n_samples, hop):
        frame = data[start:start + frame_len]
        if frame.shape[0] < frame_len:
            pad = np.zeros((frame_len - frame.shape[0], n_channels), dtype=frame.dtype)
            frame = np.vstack([frame, pad])

        frame_f = frame.astype(np.float32) * window[:, None]
        X = np.fft.rfft(frame_f, n=n_fft, axis=0)

        for f in range(n_freqs):
            x = X[f].reshape(-1, 1)
            R[f] = alpha * R[f] + (1.0 - alpha) * (x @ x.conj().T)

        if frame_idx % update_every == 0:
            for f in range(n_freqs):
                Rf = R[f]
                trace = np.trace(Rf).real
                Rf = Rf + (reg * trace / n_channels + 1e-12) * np.eye(n_channels)

                d = principal_eigenvector(Rf)
                phase = np.angle(d[ref_channel])
                d = d * np.exp(-1j * phase)

                R_inv = np.linalg.pinv(Rf)
                denom = (d.conj().T @ R_inv @ d)
                if np.abs(denom) < 1e-12:
                    weights[f] = np.zeros(n_channels, dtype=np.complex64)
                else:
                    w = (R_inv @ d) / denom
                    weights[f] = w

        Y = np.sum(weights.conj() * X, axis=1)
        y_frame = np.fft.irfft(Y, n=n_fft).astype(np.float32)

        out[start:start + frame_len] += y_frame * window
        win_sum[start:start + frame_len] += window ** 2
        frame_idx += 1

    valid = win_sum > 1e-6
    out[valid] /= win_sum[valid]
    out = out[:n_samples]
    return out


def main():
    parser = argparse.ArgumentParser(description='Blind MVDR beamforming for UMA-16 multichannel WAV')
    parser.add_argument('--input', required=True, help='Input multichannel WAV file')
    parser.add_argument('--output', required=True, help='Output mono WAV file')
    parser.add_argument('--frame-len', type=int, default=1024, help='Frame length in samples')
    parser.add_argument('--hop', type=int, default=512, help='Hop size in samples')
    parser.add_argument('--alpha', type=float, default=0.9, help='Covariance smoothing factor (0-1)')
    parser.add_argument('--reg', type=float, default=1e-3, help='Diagonal loading regularization')
    parser.add_argument('--ref-channel', type=int, default=0, help='Reference channel index for phase')
    parser.add_argument('--update-every', type=int, default=4, help='Update MVDR weights every N frames')
    parser.add_argument('--gain', type=float, default=12.0, help='Output gain in dB')
    args = parser.parse_args()

    data, sample_rate = read_wav(args.input)
    out = mvdr_beamform(
        data,
        sample_rate,
        frame_len=args.frame_len,
        hop=args.hop,
        alpha=args.alpha,
        reg=args.reg,
        ref_channel=args.ref_channel,
        update_every=args.update_every,
    )
    if args.gain != 0:
        gain_linear = 10 ** (args.gain / 20.0)
        out = np.clip(out * gain_linear, -32768, 32767)
    write_wav(args.output, out, sample_rate)


if __name__ == '__main__':
    main()
