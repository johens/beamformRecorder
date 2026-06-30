"""
Adaptive-width Frost 3D tracker (recognize but do not steer into reject range), headless (no UI).

How to run:

    python bf_Frost_adaptive_3d_null_track_adaptive_width_recognize_nosteer_no_ui.py

Stable profile:

    python3 bf_Frost_adaptive_3d_null_track_adaptive_width_recognize_nosteer_no_ui.py --track-duration 1.0 --lock-confidence-thresh 0.20 --release-confidence-thresh 0.12 --doa-bin-stride 6

Ultralight RPi profile:

    python3 bf_Frost_adaptive_3d_null_track_adaptive_width_recognize_nosteer_no_ui.py --frame-len 1024 --hop 512 --track-duration 1.0 --doa-bin-stride 24 --doa-fmin 700 --doa-fmax 2200 --fine-step 12 --wide-az-span 24 --wide-el-span 18 --narrow-az-span 6 --narrow-el-span 4 --mu 0.01 --lock-confidence-thresh 0.20 --release-confidence-thresh 0.12
    python3 bf_Frost_adaptive_3d_null_track_adaptive_width_recognize_nosteer_no_ui.py --rpi-ultralight
    
"""

import argparse
import wave

import numpy as np
import pyaudio

from common.log import debug
from bf_Frost_adaptive_3d_null import (
    SAMPLE_FORMAT,
    DEFAULT_REJECT_AZ_MIN,
    DEFAULT_REJECT_AZ_MAX,
    DEFAULT_REJECT_AZ_GUARD,
    generate_square_positions,
    select_input_device,
    wrap_angle_deg,
)
from bf_Frost_adaptive_3d_null_track_adaptive_width import (
    FrostAdaptive3DNullAdaptiveWidth,
    compute_power_for_direction,
    angular_distance_deg,
)


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


class FrostAdaptive3DNullAdaptiveWidthRecognizeNoSteer(FrostAdaptive3DNullAdaptiveWidth):
    """
    Recognize DOA everywhere (including blocked range), but keep hard no-steer behavior.
    Steering still does not update when detected azimuth is inside blocked+guard range.
    """

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
    parser = argparse.ArgumentParser(description='Adaptive Frost 3D tracker with recognize-but-no-steer reject behavior (headless)')
    parser.add_argument('--output', default='recorder_output/records/audio_frost_adaptive_3d_null_track_adaptive_width_recognize_nosteer_no_ui.wav', help='Output WAV file path')
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
        'Adaptive-width Frost tracker (recognize-no-steer, headless) started',
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
        debug('Stopping adaptive-width Frost tracker (recognize-no-steer, headless)')
    finally:
        wf.close()
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == '__main__':
    main()
