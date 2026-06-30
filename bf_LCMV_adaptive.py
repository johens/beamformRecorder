# python bf_LCMV_adaptive.py `  --output recorder_output\records\audio_lcmv_adaptive.wav `  --track-duration 5.0 `  --gain 30 `  --reg 0.05 `  --diag-load 0.15
# python bf_LCMV_adaptive.py --output recorder_output\records\audio_lcmv_adaptive_test3.wav --track-duration 5.0 --gain 42 --reg 0.05 --diag-load 0.15

import argparse
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
"""

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


def normalize_angle_to_reference(angle, reference):
    """Normalize angle to [-180, 180] while staying close to reference angle."""
    # Wrap to [-180, 180]
    angle = ((angle + 180) % 360) - 180
    # If distance to reference is > 180°, flip to the equivalent angle
    if abs(angle - reference) > 180:
        angle = angle + 360 if angle < reference else angle - 360
    return angle


def estimate_doa_grid_search(R, positions, freqs, sample_rate, n_channels, prev_az=0.0, angles_to_test=None):
    """
    Estimate DOA using 2-stage grid search with focus on region around previous angle.
    Coarse stage: 10° resolution across all angles
    Fine stage: 1° resolution around best coarse peak
    Angles are normalized to stay close to previous estimate to avoid ±180° ambiguity.
    """
    # Stage 1: Coarse search across all angles
    coarse_angles = np.arange(-180, 180, 10)
    best_az_coarse = 0.0
    best_power_coarse = -1e10
    
    for az_deg in coarse_angles:
        power = 0.0
        for f in range(len(freqs)):
            a = steering_vector(np.array([freqs[f]]), positions, az_deg, 0.0)
            a_f = a[0].reshape(-1, 1)
            a_f_norm = a_f / (np.linalg.norm(a_f) + 1e-10)
            power += np.real((a_f_norm.conj().T @ R[f] @ a_f_norm)[0, 0])
        
        if power > best_power_coarse:
            best_power_coarse = power
            best_az_coarse = az_deg
    
    # Stage 2: Fine search around coarse peak ± 10°
    fine_angles = set()
    for offset in np.arange(-9, 10, 1):
        fine_angles.add(np.clip(best_az_coarse + offset, -180, 179))
    
    best_az = 0.0
    best_power = -1e10
    
    for az_deg in fine_angles:
        power = 0.0
        for f in range(len(freqs)):
            a = steering_vector(np.array([freqs[f]]), positions, float(az_deg), 0.0)
            a_f = a[0].reshape(-1, 1)
            a_f_norm = a_f / (np.linalg.norm(a_f) + 1e-10)
            power += np.real((a_f_norm.conj().T @ R[f] @ a_f_norm)[0, 0])
        
        if power > best_power:
            best_power = power
            best_az = az_deg
    
    # Normalize result to stay close to previous angle (avoids -180°/180° flipping)
    best_az = normalize_angle_to_reference(best_az, prev_az)
    
    return float(best_az), 0.0, best_power


class LCMVAdaptive:
    def __init__(self, n_channels, sample_rate, frame_len, hop, reg, diag_load, gain_db, limiter,
                 positions, freqs, null_elevation, cov_alpha, track_duration_sec):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.frame_len = frame_len
        self.hop = hop
        self.reg = reg
        self.diag_load = diag_load
        self.gain_linear = 10 ** (gain_db / 20.0)
        self.limiter = limiter
        self.positions = positions
        self.freqs = freqs
        self.null_elevation = null_elevation
        self.cov_alpha = cov_alpha
        self.track_duration_frames = int(track_duration_sec * sample_rate / hop)

        self.n_fft = frame_len
        self.n_freqs = self.n_fft // 2 + 1
        
        self.window = np.hanning(frame_len).astype(np.float32)
        self.out_buffer = np.zeros(frame_len, dtype=np.float32)
        self.win_buffer = np.zeros(frame_len, dtype=np.float32)

        # Covariance matrix (estimated from signal)
        self.R = np.zeros((self.n_freqs, n_channels, n_channels), dtype=np.complex64)
        
        # Weights per frequency
        self.weights = np.zeros((self.n_freqs, n_channels), dtype=np.complex64)
        self.weights_prev = np.zeros((self.n_freqs, n_channels), dtype=np.complex64)
        
        # Target tracking
        self.current_target_az = 0.0
        self.current_target_el = 0.0
        self.frames_in_track = 0
        self.frames_since_doa_update = 0  # Track frames since last DOA update for smooth transitions
        self.prev_target_az = 0.0  # Previous target for smooth steering interpolation
        self.prev_target_el = 0.0

        self.frame_idx = 0

    def _estimate_doa(self):
        """Estimate DOA using 2-stage grid search with hysteresis toward previous angle."""
        try:
            # Pass previous angle to enable hysteresis
            az, el, power = estimate_doa_grid_search(
                self.R, self.positions, self.freqs, self.sample_rate, 
                self.n_channels, prev_az=self.current_target_az
            )
            return az, el, power
        except Exception as e:
            debug(f'DOA estimation failed: {e}')
            return self.current_target_az, self.current_target_el, 0.0

    def _compute_weights(self, target_az, target_el, null_az):
        """Compute LCMV weights for given target and null directions."""
        # Target steering vector
        a_target = steering_vector(self.freqs, self.positions, target_az, target_el)
        a_null = steering_vector(self.freqs, self.positions, null_az, self.null_elevation)
        
        for f in range(self.n_freqs):
            a_t = a_target[f].reshape(-1, 1)
            a_t = a_t / (np.linalg.norm(a_t) + 1e-10)
            
            a_n = a_null[f].reshape(-1, 1)
            a_n = a_n / (np.linalg.norm(a_n) + 1e-10)
            
            # Constraint matrix
            C = np.hstack([a_t, a_n])
            d = np.array([[1.0], [0.0]], dtype=np.complex64)
            
            # Covariance with diagonal loading
            Rf = self.R[f]
            trace = np.trace(Rf).real
            # Increased diagonal loading for better numerical stability
            Rf = Rf + (self.diag_load * trace / self.n_channels) * np.eye(self.n_channels, dtype=np.complex64)
            Rf = Rf + (self.reg * np.eye(self.n_channels, dtype=np.complex64))
            
            # LCMV: w = Rf^-1 * C * (C^H * Rf^-1 * C)^-1 * d
            try:
                # More conservative pseudoinverse for better numerical stability
                Rf_inv = np.linalg.pinv(Rf, rcond=1e-5)  # Increased from 1e-6
                G = C.conj().T @ Rf_inv @ C
                # Add small regularization to G for extra stability
                G = G + 1e-8 * np.eye(2, dtype=np.complex64)
                G_inv = np.linalg.pinv(G, rcond=1e-5)  # Increased from 1e-6
                numerator = Rf_inv @ C @ G_inv @ d
                
                # Normalize weights with bounds checking
                w_norm = np.linalg.norm(numerator)
                if w_norm > 1e-10 and np.isfinite(w_norm):
                    w_new = (numerator / w_norm).reshape(-1)
                else:
                    w_new = self.weights_prev[f]  # Use previous if normalization fails
                
                # Adaptive weight smoothing: very smooth (0.99) during DOA transitions,
                # then normal (0.9) during lock period to minimize clicks
                if self.frames_since_doa_update < 10:
                    weight_smoothing = 0.99  # Very smooth for first 10 frames after DOA update
                else:
                    weight_smoothing = 0.9   # Normal smoothing during lock
                self.weights[f] = weight_smoothing * self.weights_prev[f] + (1.0 - weight_smoothing) * w_new
                self.weights_prev[f] = self.weights[f]
            except (np.linalg.LinAlgError, ValueError) as e:
                debug(f'Weight computation error at freq {f}: {e}')
                self.weights[f] = self.weights_prev[f]  # Keep previous weights on error

    def process_frame(self, frame):
        frame_f = frame.astype(np.float32) * self.window[:, None]
        X = np.fft.rfft(frame_f, n=self.n_fft, axis=0)
        
        # Update covariance matrix
        for f in range(self.n_freqs):
            x = X[f].reshape(-1, 1)
            self.R[f] = self.cov_alpha * self.R[f] + (1.0 - self.cov_alpha) * (x @ x.conj().T)
        
        # Re-estimate DOA or maintain lock
        if self.frames_in_track == 0:
            # Time to find new source
            new_az, new_el, eigenval = self._estimate_doa()
            # Moderate DOA smoothing (0.4) for faster convergence while reducing jitter
            # Uses exponential moving average: 40% previous + 60% new estimate
            doa_smoothing = 0.4
            self.prev_target_az = self.current_target_az  # Save previous for smooth steering interpolation
            self.prev_target_el = self.current_target_el
            self.current_target_az = doa_smoothing * self.current_target_az + (1.0 - doa_smoothing) * new_az
            self.current_target_el = new_el
            self.frames_in_track = 0
            self.frames_since_doa_update = 0  # Reset transition counter for smooth weight convergence
            debug(f'LCMV Adaptive locked to source', azimuth=f'{self.current_target_az:.1f}°', elevation=f'{self.current_target_el:.1f}°', raw_az=f'{new_az:.1f}°', power=f'{eigenval:.2e}')
        
        self.frames_in_track += 1
        self.frames_since_doa_update += 1
        if self.frames_in_track >= self.track_duration_frames:
            self.frames_in_track = 0
        
        # Smooth steering angle over first 15 frames after DOA update to create gradual phase transition
        # This prevents audible clicks from instantaneous steering vector changes
        if self.frames_since_doa_update < 15:
            # Interpolate from previous to current steering direction
            interp_factor = self.frames_since_doa_update / 15.0  # 0.0 → 1.0 over 15 frames
            steering_az = (1.0 - interp_factor) * self.prev_target_az + interp_factor * self.current_target_az
            steering_el = (1.0 - interp_factor) * self.prev_target_el + interp_factor * self.current_target_el
        else:
            # After transition, use current target
            steering_az = self.current_target_az
            steering_el = self.current_target_el
        
        # Compute weights using smoothly-interpolated steering direction
        null_az = (steering_az + 180) % 360 - 180
        
        # During steering transitions (first 15 frames after DOA update), compute weights EVERY frame
        # for smooth interpolation. After that, compute every 2 frames for performance.
        if self.frames_since_doa_update < 15 or self.frame_idx % 2 == 0:
            self._compute_weights(steering_az, steering_el, null_az)
        
        # Apply beamformer
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
                debug(f'Audio clipped: peak {peak:.0f} > limit {limit_val:.0f}')
                out_chunk *= (limit_val / peak)
        
        out_chunk = np.clip(out_chunk, -32768, 32767)
        return out_chunk.astype(np.int16)


def main():
    parser = argparse.ArgumentParser(description='Adaptive LCMV: auto-track strongest source, lock for N seconds')
    parser.add_argument('--output', default='recorder_output/records/audio_lcmv_adaptive.wav', help='Output WAV file path')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate in Hz')
    parser.add_argument('--frame-len', type=int, default=1024, help='STFT frame length in samples')
    parser.add_argument('--hop', type=int, default=512, help='STFT hop size in samples')
    parser.add_argument('--reg', type=float, default=0.05, help='Regularization strength')
    parser.add_argument('--diag-load', type=float, default=0.15, help='Diagonal loading factor (0-1)')
    parser.add_argument('--cov-alpha', type=float, default=0.96, help='Covariance smoothing (0-1) - higher = cleaner but slower adapting')
    parser.add_argument('--gain', type=float, default=35.0, help='Output gain in dB')
    parser.add_argument('--limiter', type=float, default=0.95, help='Limiter ceiling (0-1)')
    parser.add_argument('--device-index', type=int, default=None, help='Optional input device index')
    parser.add_argument('--mic-spacing', type=float, default=0.042, help='Mic spacing in meters')
    parser.add_argument('--null-elevation', type=float, default=0.0, help='Null constraint elevation (degrees)')
    parser.add_argument('--track-duration', type=float, default=5.0, help='Lock onto source for N seconds before re-estimating DOA')
    args = parser.parse_args()

    if args.hop <= 0 or args.hop > args.frame_len:
        raise ValueError('hop must be > 0 and <= frame-len')

    pa = pyaudio.PyAudio()
    device_index, n_channels = select_input_device(pa, args.device_index)

    positions = generate_square_positions(n_channels, args.mic_spacing)
    freqs = np.fft.rfftfreq(args.frame_len, d=1.0 / args.sample_rate)

    stream = pa.open(
        format=SAMPLE_FORMAT,
        channels=n_channels,
        rate=args.sample_rate,
        frames_per_buffer=args.hop,
        input=True,
        input_device_index=device_index,
    )

    beamformer = LCMVAdaptive(
        n_channels=n_channels,
        sample_rate=args.sample_rate,
        frame_len=args.frame_len,
        hop=args.hop,
        reg=args.reg,
        diag_load=args.diag_load,
        gain_db=args.gain,
        limiter=args.limiter,
        positions=positions,
        freqs=freqs,
        null_elevation=args.null_elevation,
        cov_alpha=args.cov_alpha,
        track_duration_sec=args.track_duration,
    )

    debug('LCMV Adaptive capture started', channels=n_channels, sample_rate=args.sample_rate,
          track_duration=f'{args.track_duration}s', cov_alpha=f'{args.cov_alpha}')

    with wave.open(args.output, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(args.sample_rate)

        try:
            buffer = np.zeros((0, n_channels), dtype=np.int16)
            while True:
                data = stream.read(args.hop, exception_on_overflow=False)
                # Check for frame loss
                if not data:
                    debug('Empty frame received - possible buffer underrun')
                    continue
                samples = np.frombuffer(data, dtype=np.int16).reshape(-1, n_channels)
                buffer = np.vstack([buffer, samples])
            if samples.shape[0] != args.hop:
                debug(f'Frame dropout detected: expected {args.hop}, got {samples.shape[0]} samples')

                while buffer.shape[0] >= args.frame_len:
                    frame = buffer[:args.frame_len]
                    buffer = buffer[args.hop:]
                    y_frame = beamformer.process_frame(frame)
                    out_chunk = beamformer.overlap_add(y_frame)
                    wf.writeframes(out_chunk.tobytes())
        except KeyboardInterrupt:
            debug('Stopping LCMV Adaptive capture')
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()


if __name__ == '__main__':
    main()
