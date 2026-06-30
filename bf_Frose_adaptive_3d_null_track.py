"""
How to run:

    python bf_Frose_adaptive_3d_null_track.py

Example with explicit output and tracking params:

    python bf_Frose_adaptive_3d_null_track.py --output recorder_output\records\audio_frose_adaptive_3d_null_track.wav --track-duration 5.0 --gain 42 --mu 0.05

Faster update for demonstration of tracking response to fast movement (not recommended for regular use due to higher CPU load and less stable lock):

    python bf_Frose_adaptive_3d_null_track.py --output recorder_output\records\audio_frose_adaptive_3d_null_track.wav --track-duration 0.5 --gain 42 --mu 0.05

Optional (lighter UI refresh for smaller devices like Raspberry Pi 5):

    python bf_Frose_adaptive_3d_null_track.py --ui-fps 12 --device-index 1
"""

import argparse
import time
import wave

import cv2
import numpy as np
import pyaudio

from common.log import debug
from bf_Frost_adaptive_3d_null import (
    SAMPLE_FORMAT,
    DEFAULT_REJECT_AZ_MIN,
    DEFAULT_REJECT_AZ_MAX,
    DEFAULT_REJECT_AZ_GUARD,
    FrostAdaptive3DNull,
    generate_square_positions,
    select_input_device,
    wrap_angle_deg,
)


class Frost3DTrackerUI:
    def __init__(self, positions, window_name='Frost 3D Tracker', width=1280, height=760):
        self.positions = positions
        self.window_name = window_name
        self.width = width
        self.height = height

        self.bg_color = (20, 22, 28)
        self.panel_color = (30, 34, 44)
        self.grid_color = (60, 68, 88)
        self.text_color = (220, 228, 245)
        self.mic_color = (90, 200, 255)
        self.source_color = (65, 225, 125)
        self.reject_color = (120, 120, 255)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

        self.left_panel = (20, 70, self.width // 2 - 30, self.height - 130)
        self.right_panel = (self.width // 2 + 10, 70, self.width // 2 - 30, self.height - 130)

        self._base_canvas = None
        self._base_reject_min = None
        self._base_reject_max = None

        self._top_center = (0, 0)
        self._top_radius = 0
        self._view3d_center = (0, 0)
        self._view3d_scale = 0

        self._prev_lock_status = 'INIT'
        self._reject_flash_until = 0.0

    def _draw_panel(self, canvas, x, y, w, h, title):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.panel_color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 76, 98), 1)
        cv2.putText(canvas, title, (x + 14, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.text_color, 2, cv2.LINE_AA)

    def _draw_top_view_static(self, canvas, x, y, w, h, reject_az_min, reject_az_max):
        inner_margin = 48
        x0 = x + inner_margin
        y0 = y + inner_margin
        x1 = x + w - inner_margin
        y1 = y + h - inner_margin

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (78, 85, 108), 1)

        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        radius = int(0.44 * min(x1 - x0, y1 - y0))

        cv2.circle(canvas, (center_x, center_y), radius, self.grid_color, 1)
        cv2.line(canvas, (center_x, y0), (center_x, y1), self.grid_color, 1)
        cv2.line(canvas, (x0, center_y), (x1, center_y), self.grid_color, 1)

        if reject_az_min <= reject_az_max:
            sweep_start = reject_az_min
            sweep_end = reject_az_max
            angles = np.linspace(sweep_start, sweep_end, 80)
        else:
            angles_a = np.linspace(reject_az_min, 180, 40)
            angles_b = np.linspace(-180, reject_az_max, 40)
            angles = np.concatenate([angles_a, angles_b])

        pts = [(center_x, center_y)]
        for angle in angles:
            angle_rad = np.deg2rad(angle)
            px = int(center_x + radius * np.sin(angle_rad))
            py = int(center_y - radius * np.cos(angle_rad))
            pts.append((px, py))

        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(canvas, [pts], (40, 45, 95))

        pos_xy = self.positions[:, :2]
        max_abs = np.max(np.abs(pos_xy)) + 1e-9
        scale = (0.78 * radius) / max_abs

        for idx, (px, py) in enumerate(pos_xy):
            sx = int(center_x + px * scale)
            sy = int(center_y - py * scale)
            cv2.circle(canvas, (sx, sy), 8, self.mic_color, -1)
            cv2.circle(canvas, (sx, sy), 10, (35, 35, 35), 1)

        cv2.circle(canvas, (center_x, center_y), 12, (255, 215, 80), 2)
        cv2.circle(canvas, (center_x, center_y), 3, (255, 215, 80), -1)
        cv2.putText(canvas, 'Camera', (center_x + 16, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 120), 1, cv2.LINE_AA)

        cv2.putText(canvas, 'Front (0 deg)', (center_x - 72, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.text_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Left (-90 deg)', (x0 - 2, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.text_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Right (90 deg)', (x1 - 96, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.text_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Back (180 deg)', (center_x - 72, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.text_color, 1, cv2.LINE_AA)

        self._top_center = (center_x, center_y)
        self._top_radius = radius

    def _project_3d(self, vec3):
        vx, vy, vz = float(vec3[0]), float(vec3[1]), float(vec3[2])

        yaw = np.deg2rad(-90.0)
        pitch = np.deg2rad(25.0)

        cy = np.cos(yaw)
        sy = np.sin(yaw)
        cp = np.cos(pitch)
        sp = np.sin(pitch)

        x1 = cy * vx - sy * vy
        y1 = sy * vx + cy * vy
        z1 = vz

        x2 = x1
        y2 = cp * y1 - sp * z1
        z2 = sp * y1 + cp * z1

        depth = y2
        persp = 1.0 / (1.0 + 0.35 * depth)

        cx, cy_screen = self._view3d_center
        px = int(cx + self._view3d_scale * persp * x2)
        py = int(cy_screen - self._view3d_scale * persp * z2)
        return px, py

    def _draw_3d_view_static(self, canvas, x, y, w, h):
        inner_margin = 48
        x0 = x + inner_margin
        y0 = y + inner_margin
        x1 = x + w - inner_margin
        y1 = y + h - inner_margin

        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2

        scale = int(0.34 * min(x1 - x0, y1 - y0))
        self._view3d_center = (center_x, center_y)
        self._view3d_scale = scale

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (78, 85, 108), 1)

        origin = self._project_3d((0.0, 0.0, 0.0))
        x_axis = self._project_3d((1.0, 0.0, 0.0))
        y_axis = self._project_3d((0.0, 1.0, 0.0))
        z_axis = self._project_3d((0.0, 0.0, 1.0))

        cv2.line(canvas, origin, x_axis, (95, 195, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, origin, y_axis, (120, 255, 150), 2, cv2.LINE_AA)
        cv2.line(canvas, origin, z_axis, (255, 170, 120), 2, cv2.LINE_AA)

        cv2.putText(canvas, '+X Right', (x_axis[0] + 4, x_axis[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (95, 195, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, '+Y Front', (y_axis[0] + 4, y_axis[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 255, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, '+Z Up', (z_axis[0] + 4, z_axis[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 170, 120), 1, cv2.LINE_AA)

        ring_points = []
        for deg in np.linspace(-180, 180, 72):
            a = np.deg2rad(deg)
            ring_points.append(self._project_3d((np.sin(a), np.cos(a), 0.0)))
        ring = np.array(ring_points, dtype=np.int32)
        cv2.polylines(canvas, [ring], isClosed=True, color=self.grid_color, thickness=1, lineType=cv2.LINE_AA)

        cv2.circle(canvas, origin, 9, (255, 215, 80), 2)
        cv2.circle(canvas, origin, 3, (255, 215, 80), -1)
        cv2.putText(canvas, 'Camera/Array Origin', (origin[0] + 12, origin[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 120), 1, cv2.LINE_AA)

        cv2.putText(canvas, 'Beam arrow shows 3D DOA direction', (x0 + 8, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.text_color, 1, cv2.LINE_AA)

    def _ensure_base_canvas(self, reject_az_min, reject_az_max):
        if (
            self._base_canvas is not None
            and self._base_reject_min == reject_az_min
            and self._base_reject_max == reject_az_max
        ):
            return

        canvas = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

        title = 'Frost Adaptive 3D Null Tracker'
        cv2.putText(canvas, title, (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (236, 240, 255), 2, cv2.LINE_AA)

        lx, ly, lw, lh = self.left_panel
        rx, ry, rw, rh = self.right_panel
        self._draw_panel(canvas, lx, ly, lw, lh, 'Top View (Azimuth)')
        self._draw_panel(canvas, rx, ry, rw, rh, '3D Beam View')

        self._draw_top_view_static(canvas, lx, ly, lw, lh, reject_az_min, reject_az_max)
        self._draw_3d_view_static(canvas, rx, ry, rw, rh)

        self._base_canvas = canvas
        self._base_reject_min = reject_az_min
        self._base_reject_max = reject_az_max

    def _draw_dynamic_overlays(
        self,
        canvas,
        azimuth_deg,
        elevation_deg,
        top_beam_half_width_deg=14.0,
        beam_half_angle_deg=12.0,
        beam_mode_label='',
    ):
        top_cx, top_cy = self._top_center
        az = np.deg2rad(azimuth_deg)
        top_beam_half_width_deg = float(np.clip(top_beam_half_width_deg, 3.0, 70.0))
        beam_half_angle_deg = float(np.clip(beam_half_angle_deg, 2.0, 45.0))
        beam_half_width_deg = top_beam_half_width_deg
        beam_half_width = np.deg2rad(beam_half_width_deg)

        beam_angles = np.linspace(az - beam_half_width, az + beam_half_width, 26)
        beam_pts = [(top_cx, top_cy)]
        for ang in beam_angles:
            px = int(top_cx + self._top_radius * np.sin(ang))
            py = int(top_cy - self._top_radius * np.cos(ang))
            beam_pts.append((px, py))

        beam_pts = np.array(beam_pts, dtype=np.int32)
        cv2.fillPoly(canvas, [beam_pts], (58, 170, 102))

        center_tip_x = int(top_cx + self._top_radius * np.sin(az))
        center_tip_y = int(top_cy - self._top_radius * np.cos(az))
        cv2.line(canvas, (top_cx, top_cy), (center_tip_x, center_tip_y), self.source_color, 2, cv2.LINE_AA)
        cv2.polylines(canvas, [beam_pts], isClosed=True, color=self.source_color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (top_cx, top_cy), 5, (255, 255, 255), -1)

        mode_label = str(beam_mode_label).upper().strip()
        if mode_label:
            if mode_label == 'NARROW':
                label_bg = (40, 150, 70)
                label_fg = (235, 255, 235)
            elif mode_label == 'WIDE':
                label_bg = (155, 95, 35)
                label_fg = (255, 245, 225)
            else:
                label_bg = (95, 100, 120)
                label_fg = (240, 242, 248)

            text_x = int(center_tip_x + 10)
            text_y = int(center_tip_y - 8)
            text_w = 94
            text_h = 24
            cv2.rectangle(canvas, (text_x, text_y - text_h + 4), (text_x + text_w, text_y + 4), label_bg, -1)
            cv2.rectangle(canvas, (text_x, text_y - text_h + 4), (text_x + text_w, text_y + 4), (22, 22, 22), 1)
            cv2.putText(canvas, mode_label, (text_x + 8, text_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_fg, 1, cv2.LINE_AA)

        az = np.deg2rad(azimuth_deg)
        el = np.deg2rad(elevation_deg)
        dir_x = np.cos(el) * np.sin(az)
        dir_y = np.cos(el) * np.cos(az)
        dir_z = np.sin(el)

        origin = self._project_3d((0.0, 0.0, 0.0))

        direction = np.array([dir_x, dir_y, dir_z], dtype=np.float32)
        norm = float(np.linalg.norm(direction)) + 1e-9
        direction = direction / norm

        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        side = np.cross(direction, up_hint)
        if np.linalg.norm(side) < 1e-6:
            up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            side = np.cross(direction, up_hint)
        side = side / (np.linalg.norm(side) + 1e-9)

        beam_half_angle = np.deg2rad(beam_half_angle_deg)
        tan_half = np.tan(beam_half_angle)
        beam_length = 1.0
        base_radius = beam_length * tan_half

        beam_tip = direction * beam_length
        edge_a = beam_tip + side * base_radius
        edge_b = beam_tip - side * base_radius

        p0 = origin
        p1 = self._project_3d(tuple(edge_a.tolist()))
        p2 = self._project_3d(tuple(edge_b.tolist()))

        beam_tri = np.array([p0, p1, p2], dtype=np.int32)
        cv2.fillConvexPoly(canvas, beam_tri, (58, 170, 102))
        cv2.polylines(canvas, [beam_tri], isClosed=True, color=self.source_color, thickness=2, lineType=cv2.LINE_AA)

        tip = self._project_3d(tuple(beam_tip.tolist()))
        cv2.line(canvas, origin, tip, self.source_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, origin, 5, (255, 255, 255), -1)

        if mode_label:
            if mode_label == 'NARROW':
                label_bg = (40, 150, 70)
                label_fg = (235, 255, 235)
            elif mode_label == 'WIDE':
                label_bg = (155, 95, 35)
                label_fg = (255, 245, 225)
            else:
                label_bg = (95, 100, 120)
                label_fg = (240, 242, 248)

            text_x = int(tip[0] + 10)
            text_y = int(tip[1] - 8)
            text_w = 94
            text_h = 24
            cv2.rectangle(canvas, (text_x, text_y - text_h + 4), (text_x + text_w, text_y + 4), label_bg, -1)
            cv2.rectangle(canvas, (text_x, text_y - text_h + 4), (text_x + text_w, text_y + 4), (22, 22, 22), 1)
            cv2.putText(canvas, mode_label, (text_x + 8, text_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_fg, 1, cv2.LINE_AA)

    def render(
        self,
        azimuth_deg,
        elevation_deg,
        track_duration_sec,
        reject_az_min,
        reject_az_max,
        lock_state='INIT',
        lock_note='',
        top_beam_half_width_deg=14.0,
        beam_half_angle_deg=12.0,
        beam_mode_label='',
    ):
        self._ensure_base_canvas(reject_az_min, reject_az_max)
        if self._base_canvas is None:
            return
        canvas = self._base_canvas.copy()
        self._draw_dynamic_overlays(
            canvas,
            azimuth_deg,
            elevation_deg,
            top_beam_half_width_deg=top_beam_half_width_deg,
            beam_half_angle_deg=beam_half_angle_deg,
            beam_mode_label=beam_mode_label,
        )

        stat_x = 22
        stat_y = self.height - 24
        cv2.putText(canvas, f'Azimuth: {azimuth_deg:6.1f} deg', (stat_x, stat_y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, self.text_color, 2, cv2.LINE_8)
        cv2.putText(canvas, f'Elevation: {elevation_deg:6.1f} deg', (stat_x + 280, stat_y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, self.text_color, 2, cv2.LINE_8)
        cv2.putText(canvas, f'Track refresh: {track_duration_sec:.1f}s', (stat_x + 510, stat_y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, self.text_color, 2, cv2.LINE_8)

        badge_x = stat_x + 760
        badge_y = stat_y - 20
        badge_w = 250
        badge_h = 28

        status_upper = str(lock_state).upper()
        if status_upper == 'REJECTED':
            badge_bg = (50, 55, 180)
            badge_text = 'LOCK REJECTED'
            badge_fg = (235, 235, 255)
        elif status_upper == 'LOCKED':
            badge_bg = (45, 145, 70)
            badge_text = 'LOCK ACCEPTED'
            badge_fg = (235, 255, 235)
        else:
            badge_bg = (85, 88, 110)
            badge_text = 'LOCK PENDING'
            badge_fg = (235, 238, 245)

        cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), badge_bg, -1)
        cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (30, 30, 30), 1)
        cv2.putText(canvas, badge_text, (badge_x + 10, badge_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, badge_fg, 1, cv2.LINE_AA)

        note_text = lock_note if lock_note else 'No lock event yet'
        if len(note_text) > 90:
            note_text = note_text[:87] + '...'
        cv2.putText(canvas, note_text, (stat_x, stat_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 210, 230), 1, cv2.LINE_AA)

        cv2.putText(canvas, 'Press Q to quit', (self.width - 190, stat_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 176, 205), 1, cv2.LINE_AA)

        now = time.perf_counter()
        if status_upper == 'REJECTED' and self._prev_lock_status != 'REJECTED':
            self._reject_flash_until = now + 0.9

        if now <= self._reject_flash_until:
            remaining = max(0.0, self._reject_flash_until - now)
            strength = float(np.clip(remaining / 0.9, 0.0, 1.0))
            alpha = 0.10 + 0.35 * strength

            overlay = np.full_like(canvas, (25, 25, 210), dtype=np.uint8)
            canvas = cv2.addWeighted(canvas, 1.0 - alpha, overlay, alpha, 0.0)

            border_color = (40, 40, 255)
            border_thickness = 3 + int(3 * strength)
            cv2.rectangle(canvas, (8, 8), (self.width - 8, self.height - 8), border_color, border_thickness)

            warn_text = 'REJECTED: TARGET IN BLOCKED AZIMUTH RANGE'
            text_size = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)[0]
            wx = max(12, (self.width - text_size[0]) // 2)
            wy = max(48, self.height // 2)
            cv2.putText(canvas, warn_text, (wx, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (235, 235, 255), 2, cv2.LINE_AA)

        self._prev_lock_status = status_upper

        cv2.imshow(self.window_name, canvas)

    def should_close(self):
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                return True
        except Exception:
            return True

        key = cv2.waitKey(1) & 0xFF
        return key in (ord('q'), 27)

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Adaptive Frost (GSC) 3D with azimuth reject zone + live tracker UI')
    parser.add_argument('--output', default='recorder_output/records/audio_frose_adaptive_3d_null_track.wav', help='Output WAV file path')
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
    parser.add_argument('--el-min', type=float, default=0, help='Minimum elevation angle (degrees). For planar arrays, use 0 for above-plane tracking')
    parser.add_argument('--el-max', type=float, default=90, help='Maximum elevation angle (degrees)')
    parser.add_argument('--reject-az-min', type=float, default=DEFAULT_REJECT_AZ_MIN, help='Reject-zone azimuth min (degrees)')
    parser.add_argument('--reject-az-max', type=float, default=DEFAULT_REJECT_AZ_MAX, help='Reject-zone azimuth max (degrees)')
    parser.add_argument('--reject-az-guard', type=float, default=DEFAULT_REJECT_AZ_GUARD, help='Guard margin around reject zone (degrees)')
    parser.add_argument('--doa-smoothing', type=float, default=0.12, help='DOA smoothing weight on previous lock (0=instant jump, 1=no movement)')
    parser.add_argument('--doa-fast-jump-az', type=float, default=25.0, help='Instant jump if azimuth change exceeds this threshold (degrees)')
    parser.add_argument('--doa-fast-jump-el', type=float, default=20.0, help='Instant jump if elevation change exceeds this threshold (degrees)')
    parser.add_argument('--doa-fmin', type=float, default=300.0, help='Minimum frequency for DOA search (Hz)')
    parser.add_argument('--doa-fmax', type=float, default=3400.0, help='Maximum frequency for DOA search (Hz)')
    parser.add_argument('--doa-bin-stride', type=int, default=4, help='Use every Nth frequency bin for DOA search to reduce CPU')
    parser.add_argument('--ui-fps', type=float, default=4.0, help='Max GUI refresh rate')
    parser.add_argument('--no-ui', action='store_true', help='Disable UI to maximize audio stability')

    args = parser.parse_args()

    n_mics = 16
    spacing_m = 0.042
    positions = generate_square_positions(n_mics, spacing_m)

    is_planar_array = float(np.ptp(positions[:, 2])) < 1e-9
    if is_planar_array and args.el_min < 0 < args.el_max:
        debug(
            'Planar array elevation-sign ambiguity detected',
            note='All microphone z coordinates are identical, so above/below sign is not observable',
            recommendation='Use --el-min 0 --el-max 90 for above-plane sources',
            current_range=f'{args.el_min}° to {args.el_max}°',
        )

    p = pyaudio.PyAudio()
    device_index, n_channels = select_input_device(p, args.device_index)
    n_channels = int(n_channels)

    if n_channels < n_mics:
        raise ValueError(f'Selected device has {n_channels} channels but need {n_mics}')

    beamformer = FrostAdaptive3DNull(
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
        doa_smoothing=args.doa_smoothing,
        doa_fast_jump_az=args.doa_fast_jump_az,
        doa_fast_jump_el=args.doa_fast_jump_el,
        doa_fmin=args.doa_fmin,
        doa_fmax=args.doa_fmax,
        doa_bin_stride=args.doa_bin_stride,
    )

    stream = p.open(
        format=SAMPLE_FORMAT,
        channels=n_channels,
        rate=args.sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=args.hop,
    )

    ui = None
    if not args.no_ui:
        ui = Frost3DTrackerUI(positions=positions)

    wf = wave.open(args.output, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
    wf.setframerate(args.sample_rate)

    debug(
        'Frose Adaptive 3D Null + Tracker capture started',
        channels=n_channels,
        sample_rate=args.sample_rate,
        track_duration=f'{args.track_duration}s',
        mu=args.mu,
        cov_alpha=args.cov_alpha,
        elevation_range=f'{args.el_min}° to {args.el_max}°',
        reject_azimuth=f'{args.reject_az_min}° to {args.reject_az_max}°',
        reject_az_guard=f'{args.reject_az_guard}°',
        doa_smoothing=args.doa_smoothing,
        doa_fast_jump_az=f'{args.doa_fast_jump_az}°',
        doa_fast_jump_el=f'{args.doa_fast_jump_el}°',
        doa_band=f'{args.doa_fmin:.0f}-{args.doa_fmax:.0f}Hz',
        doa_bin_stride=args.doa_bin_stride,
        doa_bins=int(beamformer.doa_freq_indices.size),
        ui_enabled=not args.no_ui,
        ui_fps=args.ui_fps,
    )

    try:
        buffer = np.zeros((0, n_channels), dtype=np.int16)
        frame_count = 0
        ui_interval = max(1.0 / max(1.0, float(args.ui_fps)), 0.03)
        last_ui_t = 0.0

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
                    debug(f'Frame {frame_count} processing error: {e}')
                    silence = np.zeros(args.hop, dtype=np.int16)
                    wf.writeframes(silence.tobytes())

                frame_count += 1

            if ui is not None:
                az = wrap_angle_deg(beamformer.current_target_az)
                el = float(np.clip(beamformer.current_target_el, args.el_min, args.el_max))

                now = time.perf_counter()
                if now - last_ui_t >= ui_interval:
                    ui.render(
                        azimuth_deg=az,
                        elevation_deg=el,
                        track_duration_sec=args.track_duration,
                        reject_az_min=args.reject_az_min,
                        reject_az_max=args.reject_az_max,
                        lock_state=beamformer.last_lock_state,
                        lock_note=beamformer.last_lock_note,
                    )
                    last_ui_t = now

                if ui.should_close():
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        debug('Stopping Frose Adaptive 3D Null + Tracker capture')
    finally:
        wf.close()
        stream.stop_stream()
        stream.close()
        p.terminate()
        if ui is not None:
            ui.close()


if __name__ == '__main__':
    main()
