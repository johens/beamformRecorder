# python bf_Frost_adaptive_3d.py --output recorder_output\records\audio_frost_adaptive_3d.wav --track-duration 5.0 --gain 42 --mu 0.05

import argparse
import threading
import wave

import numpy as np
import pyaudio

from common.log import debug

SAMPLE_FORMAT = pyaudio.paInt16
SPEED_OF_SOUND = 343.0

"""
         Front (0°)
              ↑
       MIC8  MIC7  MIC10  MIC9
       
Left ← MIC6  MIC5  MIC12  MIC11 → Right
(-90°) MIC4  MIC3  MIC14  MIC13  (90°)
       
       MIC2  MIC1  MIC16  MIC15
              ↓
         Back (180°)
         
Elevation: 0° = horizontal, +90° = directly above, -90° = directly below
"""

def generate_square_positions(num_mics, spacing_m):
    """Generate UMA16 microphone positions with confirmed 1:1 channel-to-MIC mapping."""
    if num_mics != 16:
        raise ValueError('This function is designed for UMA16 (16 mics)')
    
    channel_positions_idx = [
        (1, 0), (0, 0), (1, 1), (0, 1),  # Channels 0-3: MIC1-4
        (1, 2), (0, 2), (1, 3), (0, 3),  # Channels 4-7: MIC5-8
        (3, 3), (2, 3), (3, 2), (2, 2),  # Channels 8-11: MIC9-12
        (3, 1), (2, 1), (3, 0), (2, 0),  # Channels 12-15: MIC13-16
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

    if max_channels_device_index is None or max_channels < 1:
        raise RuntimeError('No input device found with at least 1 input channel')

    debug('Selected audio device', index=max_channels_device_index, in_channels=max_channels)
    return max_channels_device_index, max_channels


def normalize_angle_to_reference(angle, reference):
    """Normalize angle to [-180, 180] range, keeping it close to reference."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    
    diff = angle - reference
    if diff > 180:
        angle -= 360
    elif diff < -180:
        angle += 360
    
    return angle


def estimate_doa_grid_search_3d(R, positions, freqs, sample_rate, n_channels, prev_az=0.0, prev_el=0.0, 
                                 el_min=-90, el_max=90):
    """3D grid search DOA estimation using covariance power."""
    coarse_az_angles = np.arange(-180, 180, 30)
    coarse_el_angles = np.arange(el_min, el_max + 1, 45)
    
    best_az_coarse = 0.0
    best_el_coarse = 0.0
    best_power_coarse = -1e10
    
    for az_deg in coarse_az_angles:
        for el_deg in coarse_el_angles:
            power = 0.0
            try:
                for f in range(len(freqs)):
                    a = steering_vector(np.array([freqs[f]]), positions, az_deg, el_deg)
                    a_f = a[0].reshape(-1, 1)
                    a_f_norm = a_f / (np.linalg.norm(a_f) + 1e-10)
                    power += np.real((a_f_norm.conj().T @ R[f] @ a_f_norm)[0, 0])
            except Exception as e:
                continue
            
            if power > best_power_coarse:
                best_power_coarse = power
                best_az_coarse = az_deg
                best_el_coarse = el_deg
    
    # Fine search
    fine_angles = []
    for az_offset in np.arange(-18, 19, 5):
        for el_offset in np.arange(-28, 29, 5):
            az = np.clip(best_az_coarse + az_offset, -180, 179)
            el = np.clip(best_el_coarse + el_offset, el_min, el_max)
            fine_angles.append((az, el))
    
    best_az = 0.0
    best_el = 0.0
    best_power = -1e10
    
    for az_deg, el_deg in fine_angles:
        power = 0.0
        try:
            for f in range(len(freqs)):
                a = steering_vector(np.array([freqs[f]]), positions, float(az_deg), float(el_deg))
                a_f = a[0].reshape(-1, 1)
                a_f_norm = a_f / (np.linalg.norm(a_f) + 1e-10)
                power += np.real((a_f_norm.conj().T @ R[f] @ a_f_norm)[0, 0])
        except Exception as e:
            continue
        
        if power > best_power:
            best_power = power
            best_az = az_deg
            best_el = el_deg
    
    best_az = normalize_angle_to_reference(best_az, prev_az)
    best_el = np.clip(best_el, el_min, el_max)
    
    return float(best_az), float(best_el), best_power


class FrostAdaptive3D:
    """
    Frost (GSC) beamformer with adaptive 3D DOA tracking.
    Uses LMS gradient descent for smooth weight updates - no matrix inversion needed.
    """
    def __init__(self, n_channels, sample_rate, frame_len, hop, mu, reg, gain_db, limiter,
                 positions, cov_alpha, track_duration_sec, el_min, el_max):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.hop = hop
        self.mu = mu
        self.reg = reg
        self.gain_linear = 10 ** (gain_db / 20.0)
        self.limiter = limiter
        self.positions = positions
        self.cov_alpha = cov_alpha
        self.track_duration_frames = int(track_duration_sec * sample_rate / hop)
        self.el_min = el_min
        self.el_max = el_max

        self.n_fft = frame_len
        self.n_freqs = self.n_fft // 2 + 1
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate).astype(np.float32)

        self.window = np.hanning(frame_len).astype(np.float32)
        self.out_buffer = np.zeros(frame_len, dtype=np.float32)
        self.win_buffer = np.zeros(frame_len, dtype=np.float32)

        # Covariance matrix for DOA estimation
        self.R = np.zeros((self.n_freqs, n_channels, n_channels), dtype=np.complex64)
        
        # Frost adaptive filters (LMS weights) per frequency
        self.g = np.zeros((self.n_freqs, self.n_channels), dtype=np.complex64)
        
        # Target tracking
        self.current_target_az = 0.0
        self.current_target_el = 0.0
        self.frames_in_track = 0
        
        # Current steering vectors
        self.steering = steering_vector(self.freqs, self.positions, self.current_target_az, self.current_target_el)
        
        # Async DOA estimation
        self.doa_thread = None
        self.doa_result = None
        self.doa_lock = threading.Lock()
        
        self.frame_idx = 0

    def _estimate_doa_async(self):
        """Background thread for DOA estimation."""
        try:
            az, el, power = estimate_doa_grid_search_3d(
                self.R.copy(), self.positions, self.freqs, self.sample_rate, 
                self.n_channels, prev_az=self.current_target_az, prev_el=self.current_target_el,
                el_min=self.el_min, el_max=self.el_max
            )
            with self.doa_lock:
                self.doa_result = (az, el, power)
        except Exception as e:
            debug(f'DOA estimation failed: {e}')
            with self.doa_lock:
                self.doa_result = (self.current_target_az, self.current_target_el, 0.0)

    def process_frame(self, frame):
        # Detect frame dropout
        if frame.shape[0] != self.frame_len:
            debug(f'Frame size mismatch: expected {self.frame_len}, got {frame.shape[0]}')
        
        frame_f = frame.astype(np.float32) * self.window[:, None]
        X = np.fft.rfft(frame_f, n=self.n_fft, axis=0)
        
        # Update covariance matrix for DOA estimation
        for f in range(self.n_freqs):
            x = X[f].reshape(-1, 1)
            self.R[f] = self.cov_alpha * self.R[f] + (1.0 - self.cov_alpha) * (x @ x.conj().T)
        
        # Re-estimate DOA or maintain lock (async)
        if self.frames_in_track == 0:
            # Check if previous DOA thread completed
            if self.doa_thread is not None and not self.doa_thread.is_alive():
                with self.doa_lock:
                    if self.doa_result is not None:
                        new_az, new_el, eigenval = self.doa_result
                        doa_smoothing = 0.4
                        self.current_target_az = doa_smoothing * self.current_target_az + (1.0 - doa_smoothing) * new_az
                        self.current_target_el = doa_smoothing * self.current_target_el + (1.0 - doa_smoothing) * new_el
                        
                        # Update steering vectors
                        self.steering = steering_vector(self.freqs, self.positions, self.current_target_az, self.current_target_el)
                        
                        debug(f'Frost Adaptive 3D locked to source', 
                              azimuth=f'{self.current_target_az:.1f}°', 
                              elevation=f'{self.current_target_el:.1f}°', 
                              raw_az=f'{new_az:.1f}°', 
                              raw_el=f'{new_el:.1f}°',
                              power=f'{eigenval:.2e}')
                        self.doa_result = None
            
            # Start new DOA estimation thread (non-blocking)
            if self.doa_thread is None or not self.doa_thread.is_alive():
                self.doa_thread = threading.Thread(target=self._estimate_doa_async, daemon=True)
                self.doa_thread.start()
            
            self.frames_in_track = 0
        
        self.frames_in_track += 1
        if self.frames_in_track >= self.track_duration_frames:
            self.frames_in_track = 0
        
        # Frost (GSC) beamformer with LMS weight adaptation
        Y = np.zeros(self.n_freqs, dtype=np.complex64)
        for f in range(self.n_freqs):
            try:
                a = self.steering[f].reshape(-1, 1)
                ah_a = (a.conj().T @ a).real + 1e-12
                
                # Quiescent weight vector (delay-and-sum)
                wq = (a / ah_a).reshape(-1)
                
                # Blocking matrix (projects onto null space of steering vector)
                P = np.eye(self.n_channels, dtype=np.complex64) - (a @ a.conj().T) / ah_a
                
                x = X[f]
                u = P.conj().T @ x  # Blocked signal
                y = wq.conj().T @ x - self.g[f].conj().T @ u  # Output
                
                # Detect NaN/Inf in output
                if not np.isfinite(np.abs(y)):
                    debug(f'NaN/Inf output at freq {f}, frame {self.frame_idx}')
                    Y[f] = 0.0
                    continue
                
                # NLMS weight update (smooth gradient descent)
                denom = (u.conj().T @ u).real + self.reg
                self.g[f] = self.g[f] + (self.mu / denom) * u * np.conj(y)
                
                Y[f] = y
            except Exception as e:
                debug(f'Processing error at freq {f}: {e}')
                Y[f] = 0.0
        
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
        
        # Shift circular buffer using explicit indexing (more stable than np.roll)
        self.out_buffer = np.concatenate([self.out_buffer[self.hop:], np.zeros(self.hop, dtype=np.float32)])
        self.win_buffer = np.concatenate([self.win_buffer[self.hop:], np.zeros(self.hop, dtype=np.float32)])
        
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
    parser = argparse.ArgumentParser(description='Adaptive Frost (GSC) 3D: auto-track strongest source with elevation')
    parser.add_argument('--output', default='recorder_output/records/audio_frost_adaptive_3d.wav', help='Output WAV file path')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate in Hz')
    parser.add_argument('--frame-len', type=int, default=1024, help='STFT frame length in samples')
    parser.add_argument('--hop', type=int, default=512, help='STFT hop size in samples')
    parser.add_argument('--mu', type=float, default=0.05, help='NLMS step size for weight adaptation')
    parser.add_argument('--reg', type=float, default=1e-3, help='NLMS regularization')
    parser.add_argument('--gain', type=float, default=36, help='Output gain in dB')
    parser.add_argument('--limiter', type=float, default=0.95, help='Limiter ceiling (0-1)')
    parser.add_argument('--device-index', type=int, default=None, help='Optional input device index')
    parser.add_argument('--cov-alpha', type=float, default=0.96, help='Covariance smoothing (0-1)')
    parser.add_argument('--track-duration', type=float, default=5.0, help='Lock duration in seconds before re-estimating DOA')
    parser.add_argument('--el-min', type=float, default=-90, help='Minimum elevation angle (degrees)')
    parser.add_argument('--el-max', type=float, default=90, help='Maximum elevation angle (degrees)')

    args = parser.parse_args()

    n_mics = 16
    spacing_m = 0.042
    positions = generate_square_positions(n_mics, spacing_m)

    p = pyaudio.PyAudio()
    device_index, n_channels = select_input_device(p, args.device_index)

    if n_channels < n_mics:
        raise ValueError(f'Selected device has {n_channels} channels but need {n_mics}')

    beamformer = FrostAdaptive3D(
        n_channels=n_mics,
        sample_rate=args.sample_rate,
        frame_len=args.frame_len,
        hop=args.hop,
        mu=args.mu,
        reg=args.reg,
        gain_db=args.gain,
        limiter=args.limiter,
        positions=positions,
        cov_alpha=args.cov_alpha,
        track_duration_sec=args.track_duration,
        el_min=args.el_min,
        el_max=args.el_max
    )

    stream = p.open(
        format=SAMPLE_FORMAT,
        channels=n_channels,
        rate=args.sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=args.hop
    )

    wf = wave.open(args.output, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
    wf.setframerate(args.sample_rate)

    debug('Frost Adaptive 3D capture started',
          channels=n_channels,
          sample_rate=args.sample_rate,
          track_duration=f'{args.track_duration}s',
          mu=args.mu,
          cov_alpha=args.cov_alpha,
          elevation_range=f'{args.el_min}° to {args.el_max}°')

    try:
        buffer = np.zeros((0, n_channels), dtype=np.int16)
        frame_count = 0
        while True:
            try:
                data = stream.read(args.hop, exception_on_overflow=False)
                if not data or len(data) == 0:
                    debug('Empty read from audio stream')
                    continue
                
                samples = np.frombuffer(data, dtype=np.int16).reshape(-1, n_channels)
                
                # Warn if frame size mismatch
                if samples.shape[0] != args.hop:
                    debug(f'Frame size mismatch at read #{frame_count}: expected {args.hop}, got {samples.shape[0]}')
                
                buffer = np.vstack([buffer, samples])

                # Process complete frames
                while buffer.shape[0] >= args.frame_len:
                    frame = buffer[:args.frame_len].copy()
                    buffer = buffer[args.hop:]
                    
                    try:
                        y_frame = beamformer.process_frame(frame)
                        out_chunk = beamformer.overlap_add(y_frame)
                        wf.writeframes(out_chunk.tobytes())
                    except Exception as e:
                        debug(f'Frame {frame_count} processing error: {e}')
                        silence = np.zeros(args.hop, dtype=np.int16)
                        wf.writeframes(silence.tobytes())
                    
                    frame_count += 1
            except Exception as e:
                debug(f'Audio read error: {e}')
                continue
    except KeyboardInterrupt:
        debug('Stopping Frost Adaptive 3D capture')

    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == '__main__':
    main()
