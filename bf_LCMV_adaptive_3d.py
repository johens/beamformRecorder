# python bf_LCMV_adaptive_3d.py --output recorder_output\records\audio_lcmv_adaptive_3d.wav --track-duration 5.0 --gain 42 --reg 0.05 --diag-load 0.15
# python bf_LCMV_adaptive_3d.py --output recorder_output\records\audio_lcmv_opt.wav --track-duration 10.0 --gain 42 --reg 0.05 --diag-load 0.15 2>&1 | Tee-Object -FilePath debug_lcmv_opt.log
# python bf_LCMV_adaptive_3d.py --output recorder_output\records\audio_lcmv_fast.wav --track-duration 10.0 --gain 42 --reg 0.05 --diag-load 0.15 2>&1 | Tee-Object -FilePath debug_lcmv_fast.log

import argparse
import threading
import time
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
    # Elevation: 0°=horizontal, 90°=up, -90°=down
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


def estimate_doa_grid_search_3d(R, positions, freqs, sample_rate, n_channels, prev_az=0.0, prev_el=0.0, 
                                 el_min=-90, el_max=90):
    """
    Estimate DOA using 3D 2-stage grid search over azimuth and elevation.
    Coarse stage: 30° azimuth, 45° elevation resolution (reduced for speed)
    Fine stage: 5° resolution around best coarse peak (to stay real-time)
    """
    # Stage 1: Coarse search across azimuth and elevation - REDUCED RESOLUTION FOR SPEED
    coarse_az_angles = np.arange(-180, 180, 30)  # Changed from 20° to 30°
    coarse_el_angles = np.arange(el_min, el_max + 1, 45)  # Changed from 30° to 45°
    
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
                debug(f'Error in coarse search at az={az_deg}, el={el_deg}: {e}')
                continue
            
            if power > best_power_coarse:
                best_power_coarse = power
                best_az_coarse = az_deg
                best_el_coarse = el_deg
    
    # Stage 2: Fine search around coarse peak ± 20° azimuth, ± 30° elevation
    fine_angles = []
    for az_offset in np.arange(-18, 19, 5):  # 5° steps instead of 2°
        for el_offset in np.arange(-28, 29, 5):  # 5° steps instead of 2°
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
            debug(f'Error in fine search at az={az_deg}, el={el_deg}: {e}')
            continue
        
        if power > best_power:
            best_power = power
            best_az = az_deg
            best_el = el_deg
    
    # Normalize azimuth to stay close to previous angle (avoids -180°/180° flipping)
    best_az = normalize_angle_to_reference(best_az, prev_az)
    
    # Clamp elevation to valid range
    best_el = np.clip(best_el, el_min, el_max)
    
    return float(best_az), float(best_el), best_power


class LCMVAdaptive3D:
    def __init__(self, n_channels, sample_rate, frame_len, hop, reg, diag_load, gain_db, limiter,
                 positions, freqs, null_elevation, cov_alpha, track_duration_sec, el_min, el_max,
                 cov_update_every, weight_update_every, weight_bins_per_update):
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
        self.el_min = el_min
        self.el_max = el_max
        self.cov_update_every = max(1, int(cov_update_every))
        self.weight_update_every = max(1, int(weight_update_every))

        self.n_fft = frame_len
        self.n_freqs = self.n_fft // 2 + 1
        
        # Frequency masking: only compute weights for speech band (150-6000 Hz)
        self.freq_mask = (self.freqs >= 150) & (self.freqs <= 6000)
        self.freq_bins = np.where(self.freq_mask)[0]
        self.weight_bins_per_update = max(1, int(weight_bins_per_update))
        self.freq_update_cursor = 0
        
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

    def _compute_weights(self, target_az, target_el, null_az, null_el, bins_to_update):
        """Compute LCMV weights for given target and null directions (only in speech band)."""
        # Target steering vector
        a_target = steering_vector(self.freqs, self.positions, target_az, target_el)
        a_null = steering_vector(self.freqs, self.positions, null_az, null_el)
        
        # Adaptive weight smoothing based on DOA update recency
        weight_smoothing = 0.99 if self.frames_since_doa_update < 10 else 0.9
        
        # Compute weights only for a chunk of useful frequency bins (150-6000 Hz)
        for f in bins_to_update:
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
            Rf_loaded = Rf + (self.diag_load * trace / self.n_channels) * np.eye(self.n_channels, dtype=np.complex64)
            Rf_loaded = Rf_loaded + (self.reg * np.eye(self.n_channels, dtype=np.complex64))
            
            # LCMV: w = Rf^-1 * C * (C^H * Rf^-1 * C)^-1 * d
            # Use solve() instead of pinv() for speed
            try:
                Rf_inv_C = np.linalg.solve(Rf_loaded, C)  # Solves Rf * x = C
                G = C.conj().T @ Rf_inv_C
                G_inv = np.linalg.pinv(G, rcond=1e-6)  # G is small (2x2), so pinv is ok
                numerator = Rf_inv_C @ G_inv @ d
                
                # Normalize weights
                w_norm = np.linalg.norm(numerator)
                if w_norm > 1e-10:
                    w_new = (numerator / w_norm).reshape(-1)
                else:
                    w_new = np.zeros(self.n_channels, dtype=np.complex64)
                
                self.weights[f] = weight_smoothing * self.weights_prev[f] + (1.0 - weight_smoothing) * w_new
                self.weights_prev[f] = self.weights[f]
            except np.linalg.LinAlgError:
                # If solve fails, keep previous weights
                self.weights[f] = self.weights_prev[f]

    def _next_weight_bins(self):
        """Return next chunk of frequency bins to update to avoid compute bursts."""
        n_bins = len(self.freq_bins)
        if n_bins == 0:
            return np.array([], dtype=np.int32)
        chunk = min(self.weight_bins_per_update, n_bins)
        start = self.freq_update_cursor
        end = min(start + chunk, n_bins)
        bins = self.freq_bins[start:end]
        self.freq_update_cursor = 0 if end >= n_bins else end
        return bins

    def process_frame(self, frame):
        frame_f = frame.astype(np.float32) * self.window[:, None]
        X = np.fft.rfft(frame_f, n=self.n_fft, axis=0)
        
        # Update covariance matrix (optionally decimated for speed)
        if self.frame_idx % self.cov_update_every == 0:
            outer = X[:, :, None] * X[:, None, :].conj()
            self.R = self.cov_alpha * self.R + (1.0 - self.cov_alpha) * outer
        
        # Re-estimate DOA or maintain lock (async)
        if self.frames_in_track == 0:
            # Check if previous DOA thread completed
            if self.doa_thread is not None and not self.doa_thread.is_alive():
                with self.doa_lock:
                    if self.doa_result is not None:
                        new_az, new_el, eigenval = self.doa_result
                        # DOA smoothing for both azimuth and elevation
                        # 0.4 = 40% previous + 60% new estimate for faster convergence
                        doa_smoothing = 0.75
                        self.prev_target_az = self.current_target_az  # Save previous for smooth steering interpolation
                        self.prev_target_el = self.current_target_el
                        self.current_target_az = doa_smoothing * self.current_target_az + (1.0 - doa_smoothing) * new_az
                        self.current_target_el = doa_smoothing * self.current_target_el + (1.0 - doa_smoothing) * new_el
                        self.frames_since_doa_update = 0  # Reset transition counter for smooth weight convergence
                        self.freq_update_cursor = 0
                        debug(f'LCMV Adaptive 3D locked to source', 
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
        self.frames_since_doa_update += 1
        if self.frames_in_track >= self.track_duration_frames:
            self.frames_in_track = 0
        
        # Smooth steering angle over first 15 frames after DOA update to create gradual phase transition
        # This prevents audible clicks from instantaneous steering vector changes
        transition_frames = 40
        if self.frames_since_doa_update < transition_frames:
            # Interpolate from previous to current steering direction
            interp_factor = self.frames_since_doa_update / float(transition_frames)
            steering_az = (1.0 - interp_factor) * self.prev_target_az + interp_factor * self.current_target_az
            steering_el = (1.0 - interp_factor) * self.prev_target_el + interp_factor * self.current_target_el
        else:
            # After transition, use current target
            steering_az = self.current_target_az
            steering_el = self.current_target_el
        
        # Compute weights using smoothly-interpolated steering direction
        null_az = (steering_az + 180) % 360 - 180
        null_el = steering_el  # Keep same elevation for null
        if self.frame_idx % self.weight_update_every == 0:
            bins_to_update = self._next_weight_bins()
            self._compute_weights(steering_az, steering_el, null_az, null_el, bins_to_update)
        
        # Apply beamformer (vectorized over frequencies)
        Y = np.sum(self.weights.conj() * X, axis=1).astype(np.complex64)
        
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
                debug(f'Audio clipped: peak {peak:.0f} > limit {limit_val:.0f}')
                out_chunk *= (limit_val / peak)
        
        out_chunk = np.clip(out_chunk, -32768, 32767)
        return out_chunk.astype(np.int16)


def main():
    parser = argparse.ArgumentParser(description='Adaptive LCMV 3D: auto-track strongest source with elevation')
    parser.add_argument('--output', default='recorder_output/records/audio_lcmv_adaptive_3d.wav', help='Output WAV file path')
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
    parser.add_argument('--el-min', type=float, default=-90, help='Minimum elevation angle to search (degrees)')
    parser.add_argument('--el-max', type=float, default=90, help='Maximum elevation angle to search (degrees)')
    parser.add_argument('--cov-update-every', type=int, default=2, help='Update covariance every N frames (speed vs. accuracy)')
    parser.add_argument('--weight-update-every', type=int, default=8, help='Update LCMV weights every N frames (speed vs. accuracy)')
    parser.add_argument('--weight-bins-per-update', type=int, default=24, help='Number of frequency bins updated per weight cycle')
    args = parser.parse_args()

    if args.hop <= 0 or args.hop > args.frame_len:
        raise ValueError('hop must be > 0 and <= frame-len')

    pa = pyaudio.PyAudio()
    device_index, n_channels = select_input_device(pa, args.device_index)

    positions = generate_square_positions(n_channels, args.mic_spacing)
    freqs = np.fft.rfftfreq(args.frame_len, d=1.0 / args.sample_rate)

    stream = pa.open(
        format=SAMPLE_FORMAT,
        channels=16,  # Force 16 channels
        rate=args.sample_rate,
        frames_per_buffer=args.hop,
        input=True,
        input_device_index=device_index,
    )

    beamformer = LCMVAdaptive3D(
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
        el_min=args.el_min,
        el_max=args.el_max,
        cov_update_every=args.cov_update_every,
        weight_update_every=args.weight_update_every,
        weight_bins_per_update=args.weight_bins_per_update,
    )

    debug('LCMV Adaptive 3D capture started', channels=n_channels, sample_rate=args.sample_rate,
          track_duration=f'{args.track_duration}s', cov_alpha=f'{args.cov_alpha}',
          elevation_range=f'{args.el_min}° to {args.el_max}°')
    frame_budget = args.hop / args.sample_rate

    with wave.open(args.output, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(args.sample_rate)

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
                    if samples.shape[0] != args.hop:
                        debug(f'Frame size mismatch at read #{frame_count}: expected {args.hop}, got {samples.shape[0]}')
                    buffer = np.vstack([buffer, samples])

                    while buffer.shape[0] >= args.frame_len:
                        frame = buffer[:args.frame_len].copy()
                        buffer = buffer[args.hop:]
                        try:
                            start_time = time.perf_counter()
                            y_frame = beamformer.process_frame(frame)
                            out_chunk = beamformer.overlap_add(y_frame)
                            elapsed = time.perf_counter() - start_time
                            if elapsed > frame_budget:
                                debug(f'Frame {frame_count} processing overrun: {elapsed * 1000:.1f} ms > {frame_budget * 1000:.1f} ms')
                            wf.writeframes(out_chunk.tobytes())
                        except Exception as e:
                            debug(f'Frame {frame_count} processing error: {e}')
                            silence = np.zeros(args.hop, dtype=np.int16)
                            wf.writeframes(silence.tobytes())
                        frame_count += 1
                except Exception as e:
                    debug(f'Audio read error: {e}')
                    continue
        except Exception as e:
            debug(f'Main loop error: {e}')
        except KeyboardInterrupt:
            debug('Stopping LCMV Adaptive 3D capture')
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()


if __name__ == '__main__':
    main()
