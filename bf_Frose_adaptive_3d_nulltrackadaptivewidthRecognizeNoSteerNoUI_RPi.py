"""
Self-contained adaptive-width Frost 3D tracker (recognize but do not steer into reject range), headless (no UI).

How to run:

    python3 bf_Frose_adaptive_3d_nulltrackadaptivewidthRecognizeNoSteerNoUI_RPi.py

RPi ultralight preset:

    python3 bf_Frose_adaptive_3d_nulltrackadaptivewidthRecognizeNoSteerNoUI_RPi.py --rpi-ultralight
"""

import argparse
import logging
import threading
import wave

import numpy as np
import pyaudio


def _setup_logger():
    logger = logging.getLogger('beamformer')
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s;%(levelname)s;%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


_LOG = _setup_logger()


def debug(message, **kwargs):
    if kwargs:
        payload = ', '.join(f'{key}: {value}' for key, value in kwargs.items())
        _LOG.debug(f'{message} {payload}')
    else:
        _LOG.debug(message)


SAMPLE_FORMAT = pyaudio.paInt16
SPEED_OF_SOUND = 343.0
DEFAULT_REJECT_AZ_MIN = -50.0
DEFAULT_REJECT_AZ_MAX = 50.0
DEFAULT_REJECT_AZ_GUARD = 3.0


def generate_square_positions(num_mics, spacing_m):
    if num_mics != 16:
        raise ValueError('This function is designed for UMA16 (16 mics)')

    channel_positions_idx = [
        (1, 0), (0, 0), (1, 1), (0, 1),
        (1, 2), (0, 2), (1, 3), (0, 3),
        (3, 3), (2, 3), (3, 2), (2, 2),
        (3, 1), (2, 1), (3, 0), (2, 0),
    ]

    positions = []
    for x_idx, y_idx in channel_positions_idx:
        positions.append([x_idx * spacing_m, y_idx * spacing_m, 0.0])

    positions = np.array(positions, dtype=np.float32)
    positions -= positions.mean(axis=0)
    return positions


def wrap_angle_deg(angle_deg):
    return ((angle_deg + 180) % 360) - 180


def angular_distance_deg(angle_a, angle_b):
    return wrap_angle_deg(angle_a - angle_b)


def is_azimuth_in_reject_range(azimuth_deg, reject_az_min, reject_az_max):
    az = wrap_angle_deg(azimuth_deg)
    az_min = wrap_angle_deg(reject_az_min)
    az_max = wrap_angle_deg(reject_az_max)

    if az_min <= az_max:
        return az_min <= az <= az_max
    return az >= az_min or az <= az_max


def is_azimuth_blocked_with_guard(azimuth_deg, reject_az_min, reject_az_max, guard_deg):
    az = wrap_angle_deg(azimuth_deg)
    az_min = wrap_angle_deg(reject_az_min)
    az_max = wrap_angle_deg(reject_az_max)

    if is_azimuth_in_reject_range(az, az_min, az_max):
        return True

    if guard_deg <= 0:
        return False

    if az_min <= az_max:
        return (az_min - guard_deg) <= az <= (az_max + guard_deg)

    return az >= (az_min - guard_deg) or az <= (az_max + guard_deg)


def snap_azimuth_outside_reject(azimuth_deg, reject_az_min, reject_az_max, margin_deg=0.5):
    az = wrap_angle_deg(azimuth_deg)
    az_min = wrap_angle_deg(reject_az_min)
    az_max = wrap_angle_deg(reject_az_max)

    if not is_azimuth_in_reject_range(az, az_min, az_max):
        return az

    if az_min <= az_max:
        dist_to_min = abs(az - az_min)
        dist_to_max = abs(az - az_max)
        if dist_to_min <= dist_to_max:
            return wrap_angle_deg(az_min - margin_deg)
        return wrap_angle_deg(az_max + margin_deg)

    dist_to_min = abs(wrap_angle_deg(az - az_min))
    dist_to_max = abs(wrap_angle_deg(az - az_max))
    if dist_to_min <= dist_to_max:
        return wrap_angle_deg(az_min + margin_deg)
    return wrap_angle_deg(az_max - margin_deg)


def steering_vector(freqs, positions, azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    direction = np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el),
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

    preferred_match = None
    max_channels = 0
    max_channels_device_index = None

    for i in range(pyaudio_instance.get_device_count()):
        dev = pyaudio_instance.get_device_info_by_index(i)
        input_channels = int(dev['maxInputChannels'])
        device_name = str(dev.get('name', ''))
        device_name_lower = device_name.lower()
        debug('Listing audio device', index=i, name=device_name, in_channels=input_channels)

        if input_channels >= 16 and ('uma16v2' in device_name_lower or 'uma16' in device_name_lower):
            preferred_match = (i, input_channels, device_name)

        if input_channels > max_channels:
            max_channels = input_channels
            max_channels_device_index = i

    if preferred_match is not None:
        preferred_index, preferred_channels, preferred_name = preferred_match
        debug('Selected preferred UMA16 device', index=preferred_index, name=preferred_name, in_channels=preferred_channels)
        return preferred_index, preferred_channels

    if max_channels_device_index is None or max_channels < 1:
        raise RuntimeError('No input device found with at least 1 input channel')

    debug('Selected audio device', index=max_channels_device_index, in_channels=max_channels)
    return max_channels_device_index, max_channels


def compute_power_for_direction(R, positions, freqs, freq_indices, az_deg, el_deg):
    total_power = 0.0
    for f in freq_indices:
        a = steering_vector(np.array([freqs[f]]), positions, float(az_deg), float(el_deg))
        a_f = a[0].reshape(-1, 1)
        a_f_norm = a_f / (np.linalg.norm(a_f) + 1e-10)
        total_power += np.real((a_f_norm.conj().T @ R[f] @ a_f_norm)[0, 0])
    return float(total_power)


def estimate_doa_adaptive_width(
    R,
    positions,
    freqs,
    freq_indices,
    prev_az,
    prev_el,
    el_min,
    el_max,
    reject_az_min,
    reject_az_max,
    mode,
    wide_az_span,
    wide_el_span,
    narrow_az_span,
    narrow_el_span,
    fine_step,
    prev_locked_az,
    prev_locked_el,
):
    coarse_az_angles = np.arange(-180, 180, 30)
    coarse_el_angles = np.arange(el_min, el_max + 1, 30)

    best_coarse_power = -1e12
    best_coarse_az = prev_az
    best_coarse_el = prev_el

    for az in coarse_az_angles:
        if reject_az_min is not None and reject_az_max is not None:
            if is_azimuth_in_reject_range(az, reject_az_min, reject_az_max):
                continue

        for el in coarse_el_angles:
            try:
                p = compute_power_for_direction(R, positions, freqs, freq_indices, az, el)
            except Exception:
                continue
            if p > best_coarse_power:
                best_coarse_power = p
                best_coarse_az = az
                best_coarse_el = el

    if mode == 'NARROW':
        az_span = float(narrow_az_span)
        el_span = float(narrow_el_span)
    else:
        az_span = float(wide_az_span)
        el_span = float(wide_el_span)

    candidates = []
    az_offsets = np.arange(-az_span, az_span + 1e-6, fine_step)
    el_offsets = np.arange(-el_span, el_span + 1e-6, fine_step)

    for az_off in az_offsets:
        az = wrap_angle_deg(best_coarse_az + az_off)
        if reject_az_min is not None and reject_az_max is not None:
            if is_azimuth_in_reject_range(az, reject_az_min, reject_az_max):
                continue

        for el_off in el_offsets:
            el = np.clip(best_coarse_el + el_off, el_min, el_max)
            try:
                p = compute_power_for_direction(R, positions, freqs, freq_indices, az, el)
                candidates.append((p, float(az), float(el)))
            except Exception:
                continue

    if not candidates:
        return float(prev_az), float(prev_el), 0.0, 0.0, 0.0, 0.0, 0.0

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_power, best_az, best_el = candidates[0]
    second_power = candidates[1][0] if len(candidates) > 1 else (best_power - 1e-9)

    powers = np.array([item[0] for item in candidates], dtype=np.float64)
    p_mean = float(np.mean(powers))
    p_std = float(np.std(powers) + 1e-9)

    margin_conf = float(np.clip((best_power - second_power) / (abs(best_power) + 1e-9), 0.0, 1.0))
    z_score = float((best_power - p_mean) / p_std)
    sharpness_conf = float(np.clip((z_score - 1.0) / 3.0, 0.0, 1.0))

    az_delta = abs(angular_distance_deg(best_az, prev_locked_az))
    el_delta = abs(float(best_el - prev_locked_el))
    motion_penalty = float(np.clip((az_delta / 60.0) + (el_delta / 35.0), 0.0, 1.0))
    stability_conf = 1.0 - motion_penalty

    confidence = 0.25 * margin_conf + 0.45 * sharpness_conf + 0.30 * stability_conf
    confidence = float(np.clip(confidence, 0.0, 1.0))

    best_az = wrap_angle_deg(best_az)
    best_el = float(np.clip(best_el, el_min, el_max))
    return best_az, best_el, float(best_power), confidence, margin_conf, sharpness_conf, stability_conf


def estimate_doa_adaptive_width_recognize_nosteer(
    R,
    positions,
    freqs,
    freq_indices,
    prev_az,
    prev_el,
    el_min,
    el_max,
    reject_az_min,
    reject_az_max,
    mode,
    wide_az_span,
    wide_el_span,
    narrow_az_span,
    narrow_el_span,
    fine_step,
    prev_locked_az,
    prev_locked_el,
):
    coarse_az_angles = np.arange(-180, 180, 30)
    coarse_el_angles = np.arange(el_min, el_max + 1, 30)

    best_coarse_power = -1e12
    best_coarse_az = prev_az
    best_coarse_el = prev_el

    for az in coarse_az_angles:
        for el in coarse_el_angles:
            try:
                p = compute_power_for_direction(R, positions, freqs, freq_indices, az, el)
            except Exception:
                continue
            if p > best_coarse_power:
                best_coarse_power = p
                best_coarse_az = az
                best_coarse_el = el

    if mode == 'NARROW':
        az_span = float(narrow_az_span)
        el_span = float(narrow_el_span)
    else:
        az_span = float(wide_az_span)
        el_span = float(wide_el_span)

    candidates = []
    az_offsets = np.arange(-az_span, az_span + 1e-6, fine_step)
    el_offsets = np.arange(-el_span, el_span + 1e-6, fine_step)

    for az_off in az_offsets:
        az = wrap_angle_deg(best_coarse_az + az_off)
        for el_off in el_offsets:
            el = np.clip(best_coarse_el + el_off, el_min, el_max)
            try:
                p = compute_power_for_direction(R, positions, freqs, freq_indices, az, el)
                candidates.append((p, float(az), float(el)))
            except Exception:
                continue

    if not candidates:
        return float(prev_az), float(prev_el), 0.0, 0.0, 0.0, 0.0, 0.0

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_power, best_az, best_el = candidates[0]
    second_power = candidates[1][0] if len(candidates) > 1 else (best_power - 1e-9)

    powers = np.array([item[0] for item in candidates], dtype=np.float64)
    p_mean = float(np.mean(powers))
    p_std = float(np.std(powers) + 1e-9)

    margin_conf = float(np.clip((best_power - second_power) / (abs(best_power) + 1e-9), 0.0, 1.0))
    z_score = float((best_power - p_mean) / p_std)
    sharpness_conf = float(np.clip((z_score - 1.0) / 3.0, 0.0, 1.0))

    az_delta = abs(angular_distance_deg(best_az, prev_locked_az))
    el_delta = abs(float(best_el - prev_locked_el))
    motion_penalty = float(np.clip((az_delta / 60.0) + (el_delta / 35.0), 0.0, 1.0))
    stability_conf = 1.0 - motion_penalty

    confidence = 0.25 * margin_conf + 0.45 * sharpness_conf + 0.30 * stability_conf
    confidence = float(np.clip(confidence, 0.0, 1.0))

    best_az = wrap_angle_deg(best_az)
    best_el = float(np.clip(best_el, el_min, el_max))
    return best_az, best_el, float(best_power), confidence, margin_conf, sharpness_conf, stability_conf


class FrostAdaptive3DNullAdaptiveWidth:
    def __init__(
        self,
        n_channels,
        sample_rate,
        frame_len,
        hop,
        mu,
        reg,
        gain_db,
        limiter,
        positions,
        cov_alpha,
        track_duration_sec,
        el_min,
        el_max,
        reject_az_min,
        reject_az_max,
        reject_az_guard,
        doa_fmin,
        doa_fmax,
        doa_bin_stride,
        wide_smoothing,
        narrow_smoothing,
        fast_jump_az,
        fast_jump_el,
        lock_confidence_thresh,
        release_confidence_thresh,
        stable_locks_required,
        wide_az_span,
        wide_el_span,
        narrow_az_span,
        narrow_el_span,
        fine_step,
    ):
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
        self.track_duration_frames = max(1, int(track_duration_sec * sample_rate / hop))
        self.el_min = el_min
        self.el_max = el_max
        self.reject_az_min = reject_az_min
        self.reject_az_max = reject_az_max
        self.reject_az_guard = max(0.0, float(reject_az_guard))

        self.wide_smoothing = float(np.clip(wide_smoothing, 0.0, 1.0))
        self.narrow_smoothing = float(np.clip(narrow_smoothing, 0.0, 1.0))
        self.fast_jump_az = max(0.0, float(fast_jump_az))
        self.fast_jump_el = max(0.0, float(fast_jump_el))

        self.lock_confidence_thresh = float(np.clip(lock_confidence_thresh, 0.0, 1.0))
        self.release_confidence_thresh = float(np.clip(release_confidence_thresh, 0.0, 1.0))
        self.stable_locks_required = max(1, int(stable_locks_required))

        self.wide_az_span = max(5.0, float(wide_az_span))
        self.wide_el_span = max(5.0, float(wide_el_span))
        self.narrow_az_span = max(3.0, float(narrow_az_span))
        self.narrow_el_span = max(3.0, float(narrow_el_span))
        self.fine_step = max(1.0, float(fine_step))

        self.n_fft = frame_len
        self.n_freqs = self.n_fft // 2 + 1
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate).astype(np.float32)

        doa_fmin = max(0.0, float(doa_fmin))
        doa_fmax = max(doa_fmin, float(doa_fmax))
        doa_bin_stride = max(1, int(doa_bin_stride))

        doa_mask = (self.freqs >= doa_fmin) & (self.freqs <= doa_fmax)
        doa_indices = np.where(doa_mask)[0]
        if doa_indices.size == 0:
            doa_indices = np.arange(self.n_freqs)
        self.doa_freq_indices = doa_indices[::doa_bin_stride]
        if self.doa_freq_indices.size == 0:
            self.doa_freq_indices = doa_indices

        self.window = np.hanning(frame_len).astype(np.float32)
        self.out_buffer = np.zeros(frame_len, dtype=np.float32)
        self.win_buffer = np.zeros(frame_len, dtype=np.float32)

        self.R = np.zeros((self.n_freqs, n_channels, n_channels), dtype=np.complex64)
        self.g = np.zeros((self.n_freqs, self.n_channels), dtype=np.complex64)

        self.current_target_az = 0.0
        self.current_target_el = 0.0
        self.steering = steering_vector(self.freqs, self.positions, self.current_target_az, self.current_target_el)

        self.track_mode = 'WIDE'
        self.stable_lock_count = 0
        self.last_confidence = 0.0
        self.last_confidence_components = (0.0, 0.0, 0.0)

        self.last_lock_state = 'INIT'
        self.last_lock_note = 'Waiting for first DOA update'

        self.frames_in_track = 0
        self.frame_idx = 0

        self.doa_thread = None
        self.doa_result = None
        self.doa_lock = threading.Lock()

    def _estimate_doa_async(self):
        try:
            az, el, power, conf, c_margin, c_sharp, c_stable = estimate_doa_adaptive_width(
                self.R.copy(),
                self.positions,
                self.freqs,
                self.doa_freq_indices,
                prev_az=self.current_target_az,
                prev_el=self.current_target_el,
                el_min=self.el_min,
                el_max=self.el_max,
                reject_az_min=self.reject_az_min,
                reject_az_max=self.reject_az_max,
                mode=self.track_mode,
                wide_az_span=self.wide_az_span,
                wide_el_span=self.wide_el_span,
                narrow_az_span=self.narrow_az_span,
                narrow_el_span=self.narrow_el_span,
                fine_step=self.fine_step,
                prev_locked_az=self.current_target_az,
                prev_locked_el=self.current_target_el,
            )
            with self.doa_lock:
                self.doa_result = (az, el, power, conf, c_margin, c_sharp, c_stable)
        except Exception as e:
            debug(f'Adaptive DOA estimation failed: {e}')
            with self.doa_lock:
                self.doa_result = (self.current_target_az, self.current_target_el, 0.0, 0.0, 0.0, 0.0, 0.0)

    def process_frame(self, frame):
        frame_f = frame.astype(np.float32) * self.window[:, None]
        X = np.fft.rfft(frame_f, n=self.n_fft, axis=0)

        for f in range(self.n_freqs):
            x = X[f].reshape(-1, 1)
            self.R[f] = self.cov_alpha * self.R[f] + (1.0 - self.cov_alpha) * (x @ x.conj().T)

        if self.frames_in_track == 0:
            if self.doa_thread is not None and not self.doa_thread.is_alive():
                with self.doa_lock:
                    if self.doa_result is not None:
                        new_az, new_el, eigenval, conf, c_margin, c_sharp, c_stable = self.doa_result
                        new_az = wrap_angle_deg(new_az)
                        self.last_confidence = float(conf)
                        self.last_confidence_components = (float(c_margin), float(c_sharp), float(c_stable))

                        if is_azimuth_blocked_with_guard(
                            new_az,
                            self.reject_az_min,
                            self.reject_az_max,
                            self.reject_az_guard,
                        ):
                            self.last_lock_state = 'REJECTED'
                            self.last_lock_note = (
                                f'Rejected az {new_az:.1f} deg (blocked range {self.reject_az_min:.1f}..{self.reject_az_max:.1f})'
                            )
                            self.track_mode = 'WIDE'
                            self.stable_lock_count = 0
                            debug(
                                'Adaptive-width DOA rejected',
                                rejected_az=f'{new_az:.1f}°',
                                mode=self.track_mode,
                                confidence=f'{conf:.3f}',
                                kept_locked_az=f'{self.current_target_az:.1f}°',
                            )
                        else:
                            prev_az_wrapped = wrap_angle_deg(self.current_target_az)
                            az_delta = abs(angular_distance_deg(new_az, prev_az_wrapped))
                            el_delta = abs(float(new_el - self.current_target_el))
                            fast_jump = (az_delta >= self.fast_jump_az) or (el_delta >= self.fast_jump_el)

                            if conf >= self.lock_confidence_thresh:
                                self.stable_lock_count += 1
                            else:
                                self.stable_lock_count = max(0, self.stable_lock_count - 1)

                            if self.track_mode == 'WIDE' and self.stable_lock_count >= self.stable_locks_required:
                                self.track_mode = 'NARROW'
                            elif self.track_mode == 'NARROW' and conf < self.release_confidence_thresh:
                                self.track_mode = 'WIDE'
                                self.stable_lock_count = 0

                            if fast_jump:
                                self.track_mode = 'WIDE'
                                self.stable_lock_count = 0

                            smoothing = self.narrow_smoothing if self.track_mode == 'NARROW' else self.wide_smoothing
                            if fast_jump:
                                smoothing = 0.0

                            self.current_target_az = smoothing * prev_az_wrapped + (1.0 - smoothing) * new_az
                            self.current_target_az = wrap_angle_deg(self.current_target_az)
                            self.current_target_az = snap_azimuth_outside_reject(
                                self.current_target_az,
                                self.reject_az_min,
                                self.reject_az_max,
                            )
                            self.current_target_el = smoothing * self.current_target_el + (1.0 - smoothing) * new_el

                            self.steering = steering_vector(
                                self.freqs,
                                self.positions,
                                self.current_target_az,
                                self.current_target_el,
                            )

                            self.last_lock_state = 'LOCKED'
                            self.last_lock_note = (
                                f'{self.track_mode} conf={conf:.3f} smooth={smoothing:.2f} fast_jump={fast_jump}'
                            )

                            debug(
                                'Adaptive-width lock update',
                                azimuth=f'{self.current_target_az:.1f}°',
                                elevation=f'{self.current_target_el:.1f}°',
                                raw_az=f'{new_az:.1f}°',
                                raw_el=f'{new_el:.1f}°',
                                confidence=f'{conf:.3f}',
                                conf_margin=f'{c_margin:.3f}',
                                conf_sharp=f'{c_sharp:.3f}',
                                conf_stable=f'{c_stable:.3f}',
                                mode=self.track_mode,
                                smooth=f'{smoothing:.2f}',
                                fast_jump=fast_jump,
                                power=f'{eigenval:.2e}',
                            )

                        self.doa_result = None

            if self.doa_thread is None or not self.doa_thread.is_alive():
                self.doa_thread = threading.Thread(target=self._estimate_doa_async, daemon=True)
                self.doa_thread.start()

            self.frames_in_track = 0

        self.frames_in_track += 1
        if self.frames_in_track >= self.track_duration_frames:
            self.frames_in_track = 0

        Y = np.zeros(self.n_freqs, dtype=np.complex64)
        for f in range(self.n_freqs):
            try:
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
            except Exception:
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


class FrostAdaptive3DNullAdaptiveWidthRecognizeNoSteer(FrostAdaptive3DNullAdaptiveWidth):
    def _estimate_doa_async(self):
        try:
            az, el, power, conf, c_margin, c_sharp, c_stable = estimate_doa_adaptive_width_recognize_nosteer(
                self.R.copy(),
                self.positions,
                self.freqs,
                self.doa_freq_indices,
                prev_az=self.current_target_az,
                prev_el=self.current_target_el,
                el_min=self.el_min,
                el_max=self.el_max,
                reject_az_min=self.reject_az_min,
                reject_az_max=self.reject_az_max,
                mode=self.track_mode,
                wide_az_span=self.wide_az_span,
                wide_el_span=self.wide_el_span,
                narrow_az_span=self.narrow_az_span,
                narrow_el_span=self.narrow_el_span,
                fine_step=self.fine_step,
                prev_locked_az=self.current_target_az,
                prev_locked_el=self.current_target_el,
            )
            with self.doa_lock:
                self.doa_result = (az, el, power, conf, c_margin, c_sharp, c_stable)
        except Exception as e:
            debug(f'Adaptive DOA estimation failed (recognize-nosteer): {e}')
            with self.doa_lock:
                self.doa_result = (self.current_target_az, self.current_target_el, 0.0, 0.0, 0.0, 0.0, 0.0)


def main():
    parser = argparse.ArgumentParser(
        description='Self-contained adaptive Frost 3D tracker with recognize-but-no-steer reject behavior (headless)'
    )
    parser.add_argument('--output', default='recorder_output/records/audio_frose_adaptive_3d_nulltrackadaptivewidthRecognizeNoSteerNoUI_RPi.wav', help='Output WAV file path')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate in Hz')
    parser.add_argument('--frame-len', type=int, default=1024, help='STFT frame length in samples')
    parser.add_argument('--hop', type=int, default=512, help='STFT hop size in samples')
    parser.add_argument('--mu', type=float, default=0.05, help='NLMS step size')
    parser.add_argument('--reg', type=float, default=1e-3, help='NLMS regularization')
    parser.add_argument('--gain', type=float, default=36, help='Output gain in dB')
    parser.add_argument('--limiter', type=float, default=0.95, help='Limiter ceiling (0-1)')
    parser.add_argument('--device-index', type=int, default=None, help='Optional input device index')
    parser.add_argument('--cov-alpha', type=float, default=0.96, help='Covariance smoothing (0-1)')
    parser.add_argument('--track-duration', type=float, default=1.0, help='DOA refresh interval in seconds')

    parser.add_argument('--el-min', type=float, default=0, help='Minimum elevation angle (degrees)')
    parser.add_argument('--el-max', type=float, default=90, help='Maximum elevation angle (degrees)')
    parser.add_argument('--reject-az-min', type=float, default=DEFAULT_REJECT_AZ_MIN, help='Reject-zone azimuth min')
    parser.add_argument('--reject-az-max', type=float, default=DEFAULT_REJECT_AZ_MAX, help='Reject-zone azimuth max')
    parser.add_argument('--reject-az-guard', type=float, default=DEFAULT_REJECT_AZ_GUARD, help='Reject-zone guard margin')

    parser.add_argument('--doa-fmin', type=float, default=300.0, help='DOA min frequency (Hz)')
    parser.add_argument('--doa-fmax', type=float, default=3200.0, help='DOA max frequency (Hz)')
    parser.add_argument('--doa-bin-stride', type=int, default=6, help='Use every Nth bin for DOA')

    parser.add_argument('--wide-smoothing', type=float, default=0.06, help='Smoothing in WIDE mode')
    parser.add_argument('--narrow-smoothing', type=float, default=0.22, help='Smoothing in NARROW mode')
    parser.add_argument('--fast-jump-az', type=float, default=22.0, help='Force instant jump if azimuth change exceeds threshold')
    parser.add_argument('--fast-jump-el', type=float, default=18.0, help='Force instant jump if elevation change exceeds threshold')

    parser.add_argument('--lock-confidence-thresh', type=float, default=0.20, help='Enter/keep narrow mode above this confidence')
    parser.add_argument('--release-confidence-thresh', type=float, default=0.12, help='Drop back to wide mode below this confidence')
    parser.add_argument('--stable-locks-required', type=int, default=3, help='Consecutive confident locks before narrowing')

    parser.add_argument('--wide-az-span', type=float, default=50.0, help='Fine search az span in WIDE mode (deg)')
    parser.add_argument('--wide-el-span', type=float, default=35.0, help='Fine search el span in WIDE mode (deg)')
    parser.add_argument('--narrow-az-span', type=float, default=12.0, help='Fine search az span in NARROW mode (deg)')
    parser.add_argument('--narrow-el-span', type=float, default=9.0, help='Fine search el span in NARROW mode (deg)')
    parser.add_argument('--fine-step', type=float, default=6.0, help='Fine search grid step (deg)')
    parser.add_argument('--rpi-ultralight', action='store_true', help='Apply low-compute preset tuned for Raspberry Pi real-time stability')

    args = parser.parse_args()

    if args.rpi_ultralight:
        args.frame_len = 1024
        args.hop = 512
        args.track_duration = 1.0
        args.doa_bin_stride = 24
        args.doa_fmin = 700.0
        args.doa_fmax = 2200.0
        args.fine_step = 12.0
        args.wide_az_span = 24.0
        args.wide_el_span = 18.0
        args.narrow_az_span = 6.0
        args.narrow_el_span = 4.0
        args.mu = 0.01
        debug(
            'Applied RPi ultralight preset',
            frame_len=args.frame_len,
            hop=args.hop,
            track_duration=args.track_duration,
            doa_bin_stride=args.doa_bin_stride,
            doa_fmin=args.doa_fmin,
            doa_fmax=args.doa_fmax,
            fine_step=args.fine_step,
            wide_az_span=args.wide_az_span,
            wide_el_span=args.wide_el_span,
            narrow_az_span=args.narrow_az_span,
            narrow_el_span=args.narrow_el_span,
            mu=args.mu,
        )

    if args.release_confidence_thresh >= args.lock_confidence_thresh:
        args.release_confidence_thresh = max(0.0, args.lock_confidence_thresh - 0.03)
        debug(
            'Adjusted confidence hysteresis',
            lock_confidence_thresh=args.lock_confidence_thresh,
            release_confidence_thresh=args.release_confidence_thresh,
        )

    n_mics = 16
    spacing_m = 0.042
    positions = generate_square_positions(n_mics, spacing_m)

    p = pyaudio.PyAudio()
    device_index, n_channels = select_input_device(p, args.device_index)
    n_channels = int(n_channels)

    if args.device_index is not None and n_channels < n_mics:
        debug(
            'Provided device index is not UMA16-capable; retrying auto device selection',
            provided_index=args.device_index,
            provided_channels=n_channels,
        )
        device_index, n_channels = select_input_device(p, None)
        n_channels = int(n_channels)

    if n_channels < n_mics:
        raise ValueError(f'Selected device has {n_channels} channels but need {n_mics}')

    beamformer = FrostAdaptive3DNullAdaptiveWidthRecognizeNoSteer(
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
        el_max=args.el_max,
        reject_az_min=args.reject_az_min,
        reject_az_max=args.reject_az_max,
        reject_az_guard=args.reject_az_guard,
        doa_fmin=args.doa_fmin,
        doa_fmax=args.doa_fmax,
        doa_bin_stride=args.doa_bin_stride,
        wide_smoothing=args.wide_smoothing,
        narrow_smoothing=args.narrow_smoothing,
        fast_jump_az=args.fast_jump_az,
        fast_jump_el=args.fast_jump_el,
        lock_confidence_thresh=args.lock_confidence_thresh,
        release_confidence_thresh=args.release_confidence_thresh,
        stable_locks_required=args.stable_locks_required,
        wide_az_span=args.wide_az_span,
        wide_el_span=args.wide_el_span,
        narrow_az_span=args.narrow_az_span,
        narrow_el_span=args.narrow_el_span,
        fine_step=args.fine_step,
    )

    stream = p.open(
        format=SAMPLE_FORMAT,
        channels=n_channels,
        rate=args.sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=args.hop,
    )

    wf = wave.open(args.output, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
    wf.setframerate(args.sample_rate)

    debug(
        'Self-contained adaptive-width Frost tracker (recognize-no-steer, headless) started',
        track_duration=f'{args.track_duration}s',
        reject_azimuth=f'{args.reject_az_min}° to {args.reject_az_max}°',
        reject_az_guard=f'{args.reject_az_guard}°',
        lock_confidence_thresh=args.lock_confidence_thresh,
        release_confidence_thresh=args.release_confidence_thresh,
        wide_span=f'az±{args.wide_az_span} el±{args.wide_el_span}',
        narrow_span=f'az±{args.narrow_az_span} el±{args.narrow_el_span}',
        doa_band=f'{args.doa_fmin:.0f}-{args.doa_fmax:.0f}Hz',
        doa_bin_stride=args.doa_bin_stride,
        doa_bins=int(beamformer.doa_freq_indices.size),
    )

    try:
        buffer = np.zeros((0, n_channels), dtype=np.int16)

        while True:
            data = stream.read(args.hop, exception_on_overflow=False)
            if not data:
                continue

            samples = np.frombuffer(data, dtype=np.int16).reshape(-1, n_channels)
            buffer = np.vstack([buffer, samples])

            while buffer.shape[0] >= args.frame_len:
                frame = buffer[:args.frame_len].copy()
                buffer = buffer[args.hop:]
                try:
                    y_frame = beamformer.process_frame(frame)
                    out_chunk = beamformer.overlap_add(y_frame)
                    wf.writeframes(out_chunk.tobytes())
                except Exception as e:
                    debug(f'Recognize-no-steer frame processing error: {e}')
                    silence = np.zeros(args.hop, dtype=np.int16)
                    wf.writeframes(silence.tobytes())

    except KeyboardInterrupt:
        debug('Stopping self-contained adaptive-width Frost tracker (recognize-no-steer, headless)')
    finally:
        wf.close()
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == '__main__':
    main()
