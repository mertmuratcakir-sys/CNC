import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import io
import json
import os
import re
import pandas as pd
from moviepy.editor import ImageSequenceClip

CHANNEL_COLORS = ['#0072B2', '#D55E00', '#0072B2']

demo_gcode = [
    "G1 X0 Y0 F800\nG1 X40 Y0\nG41\nG1 X40 Y30\nG2 X20 Y50 R20\nG1 X0 Y50\nG40\nWAIT\nG1 X0 Y0",
    "G1 X60 Y60 F600\nWAIT\nG1 X10 Y10\nG3 X30 Y10 R-15\nG1 X10 Y10",
    "G1 X-20 Y-20 F1000\nG4 P1\nWAIT\nG1 X-20 Y40",
]

def parse_gcode(text):
    commands = []
    for raw_line in text.strip().split('\n'):
        line = raw_line.split(';')[0].strip()
        if not line:
            continue
        if line.upper().startswith('WAIT'):
            commands.append({'type': 'WAIT'})
            continue
        tokens = re.findall(r'([A-Za-z])\s*(-?\d+\.?\d*)', line)
        if not tokens:
            continue
        cmd = {'type': 'CMD'}
        for letter, value in tokens:
            cmd[letter.upper()] = float(value)
        commands.append(cmd)
    return commands

def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    return (v[0] / n, v[1] / n) if n > 1e-12 else (0.0, 0.0)

def _rotate90(v, sign):
    return (-sign * v[1], sign * v[0])

def _resample_arc(center, r, a0, a1, n_min=3):
    n_pts = max(int(abs(a1 - a0) / 0.04), n_min)
    angs = np.linspace(a0, a1, n_pts)
    return [(center[0] + r * np.cos(a), center[1] + r * np.sin(a)) for a in angs]

def _tangent_for(seg, ang):
    if seg['clockwise']:
        return _unit((np.sin(ang), -np.cos(ang)))
    return _unit((-np.sin(ang), np.cos(ang)))

def simulate_channel(commands, start_x, start_y, tool_radius, default_feed=1000.0, y_offset=0.0):
    x, y = start_x, start_y
    motion_mode = 'G0'
    comp_mode = 'G40'
    feed = default_feed
    segments = []
    move_counter = 0

    for cmd in commands:
        if cmd['type'] == 'WAIT':
            segments.append({'type': 'wait'})
            continue
        if 'G' in cmd:
            g = int(cmd['G'])
            if g in (0, 1, 2, 3):
                motion_mode = f'G{g}'
            elif g == 4:
                motion_mode = 'G4'
            elif g in (40, 41, 42):
                comp_mode = f'G{g}'
        if 'F' in cmd:
            feed = cmd['F']
        if motion_mode == 'G4':
            dwell = cmd.get('P', 0.0)
            segments.append({'type': 'dwell', 'point': (x, y), 'duration': dwell})
            continue

        target_x = cmd.get('X', x)
        target_y = cmd.get('Y', y - y_offset) + y_offset
        if np.isclose(x, target_x) and np.isclose(y, target_y):
            continue

        active_comp = comp_mode if comp_mode in ('G41', 'G42') else 'G40'
        seg_color_idx = move_counter
        move_counter += 1

        if motion_mode in ('G0', 'G1'):
            p0 = (x, y)
            p1 = (target_x, target_y)
            direction = _unit((p1[0] - p0[0], p1[1] - p0[1]))
            if active_comp != 'G40':
                sign = +1 if active_comp == 'G41' else -1
                normal = _rotate90(direction, sign)
                p0c = (p0[0] + normal[0] * tool_radius, p0[1] + normal[1] * tool_radius)
                p1c = (p1[0] + normal[0] * tool_radius, p1[1] + normal[1] * tool_radius)
            else:
                p0c, p1c = p0, p1
            dist = float(np.hypot(p1c[0] - p0c[0], p1c[1] - p0c[1]))
            eff_feed = feed if motion_mode == 'G1' else max(feed * 3, 3000)
            dur = (dist / eff_feed) * 60 if eff_feed > 0 else 0.0
            segments.append({
                'type': 'move', 'kind': 'line',
                'points': [p0c, p1c], 'duration': dur, 'feed': eff_feed,
                'comp_mode': active_comp,
                'tangent_start': direction, 'tangent_end': direction,
                'orig_start': p0, 'orig_end': p1,
                'orig_points': [p0, p1],
                'color_idx': seg_color_idx
            })
        elif motion_mode in ('G2', 'G3'):
            clockwise = (motion_mode == 'G2')
            if 'R' in cmd:
                r_signed = cmd['R']
                r_abs = abs(r_signed)
                dx, dy = target_x - x, target_y - y
                d = float(np.hypot(dx, dy))
                if d < 1e-9:
                    x, y = target_x, target_y
                    continue
                if r_abs < d / 2:
                    r_abs = d / 2
                h = float(np.sqrt(max(r_abs ** 2 - (d / 2) ** 2, 0.0)))
                mx, my = (x + target_x) / 2, (y + target_y) / 2
                ux, uy = dx / d, dy / d
                n_left = (-uy, ux)
                n_right = (uy, -ux)
                if clockwise:
                    normal = n_right if r_signed >= 0 else n_left
                else:
                    normal = n_left if r_signed >= 0 else n_right
                cx = mx + h * normal[0]
                cy = my + h * normal[1]
            else:
                i_off = cmd.get('I', 0.0)
                j_off = cmd.get('J', 0.0)
                cx, cy = x + i_off, y + j_off

            r = float(np.hypot(x - cx, y - cy))
            start_ang = np.arctan2(y - cy, x - cx)
            end_ang = np.arctan2(target_y - cy, target_x - cx)
            if clockwise and end_ang > start_ang:
                end_ang -= 2 * np.pi
            if (not clockwise) and end_ang < start_ang:
                end_ang += 2 * np.pi
            if np.isclose(start_ang, end_ang):
                end_ang += (-2 * np.pi if clockwise else 2 * np.pi)

            r_comp = r
            if active_comp == 'G41':
                r_comp = (r + tool_radius) if clockwise else (r - tool_radius)
            elif active_comp == 'G42':
                r_comp = (r - tool_radius) if clockwise else (r + tool_radius)
            r_comp = max(r_comp, 0.01)

            pts = _resample_arc((cx, cy), r_comp, start_ang, end_ang)
            orig_pts = _resample_arc((cx, cy), r, start_ang, end_ang)
            arc_len = abs(end_ang - start_ang) * r_comp
            dur = (arc_len / feed) * 60 if feed > 0 else 0.0
            seg = {
                'type': 'move', 'kind': 'arc',
                'points': pts, 'duration': dur, 'feed': feed,
                'comp_mode': active_comp,
                'center': (cx, cy), 'radius': r_comp, 'orig_radius': r,
                'start_ang': start_ang, 'end_ang': end_ang, 'clockwise': clockwise,
                'orig_start': (x, y), 'orig_end': (target_x, target_y),
                'orig_points': orig_pts,
                'color_idx': seg_color_idx
            }
            seg['tangent_start'] = _tangent_for(seg, start_ang)
            seg['tangent_end'] = _tangent_for(seg, end_ang)
            segments.append(seg)
        x, y = target_x, target_y
    return segments

def _line_line_intersection(p0, d0, p1, d1):
    cross_d = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(cross_d) < 1e-9:
        return None
    ex, ey = p1[0] - p0[0], p1[1] - p0[1]
    t = (ex * d1[1] - ey * d1[0]) / cross_d
    return (p0[0] + t * d0[0], p0[1] + t * d0[1])

def _line_circle_intersection(p0, d, center, r, near_point):
    fx, fy = p0[0] - center[0], p0[1] - center[1]
    a = d[0] ** 2 + d[1] ** 2
    b = 2 * (fx * d[0] + fy * d[1])
    c = fx ** 2 + fy ** 2 - r ** 2
    disc = b ** 2 - 4 * a * c
    if disc < 0 or a < 1e-12:
        return None
    sq = np.sqrt(disc)
    t1, t2 = (-b + sq) / (2 * a), (-b - sq) / (2 * a)
    cand1 = (p0[0] + t1 * d[0], p0[1] + t1 * d[1])
    cand2 = (p0[0] + t2 * d[0], p0[1] + t2 * d[1])
    d1_ = (cand1[0] - near_point[0]) ** 2 + (cand1[1] - near_point[1]) ** 2
    d2_ = (cand2[0] - near_point[0]) ** 2 + (cand2[1] - near_point[1]) ** 2
    return cand1 if d1_ <= d2_ else cand2

def _circle_circle_intersection(c1, r1, c2, r2, near_point):
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    d = float(np.hypot(dx, dy))
    if d < 1e-9 or d > r1 + r2 or d < abs(r1 - r2):
        return None
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h_sq = r1 ** 2 - a ** 2
    if h_sq < 0:
        return None
    h = float(np.sqrt(h_sq))
    px, py = c1[0] + a * dx / d, c1[1] + a * dy / d
    perp = (-dy / d, dx / d)
    cand1 = (px + h * perp[0], py + h * perp[1])
    cand2 = (px - h * perp[0], py - h * perp[1])
    d1_ = (cand1[0] - near_point[0]) ** 2 + (cand1[1] - near_point[1]) ** 2
    d2_ = (cand2[0] - near_point[0]) ** 2 + (cand2[1] - near_point[1]) ** 2
    return cand1 if d1_ <= d2_ else cand2

def _trim_arc(seg, new_point, at_start):
    ang = np.arctan2(new_point[1] - seg['center'][1], new_point[0] - seg['center'][0])
    ref = seg['end_ang'] if at_start else seg['start_ang']
    while ang - ref > np.pi: ang -= 2 * np.pi
    while ang - ref < -np.pi: ang += 2 * np.pi
    if at_start:
        seg['start_ang'] = ang
        seg['tangent_start'] = _tangent_for(seg, ang)
    else:
        seg['end_ang'] = ang
        seg['tangent_end'] = _tangent_for(seg, ang)
    seg['points'] = _resample_arc(seg['center'], seg['radius'], seg['start_ang'], seg['end_ang'])
    seg['orig_points'] = _resample_arc(seg['center'], seg['orig_radius'], seg['start_ang'], seg['end_ang'])
    arc_len = abs(seg['end_ang'] - seg['start_ang']) * seg['radius']
    seg['duration'] = (arc_len / seg['feed']) * 60 if seg['feed'] > 0 else 0.0

def _trim_line(seg, new_point, at_start, tool_radius):
    sign = +1 if seg['comp_mode'] == 'G41' else -1
    normal = _rotate90(seg['tangent_end'], sign)
    orig_new_point = (new_point[0] - normal[0] * tool_radius, new_point[1] - normal[1] * tool_radius)
    if at_start:
        seg['points'] = [new_point, seg['points'][-1]]
        seg['orig_points'] = [orig_new_point, seg['orig_points'][-1]]
    else:
        seg['points'] = [seg['points'][0], new_point]
        seg['orig_points'] = [seg['orig_points'][0], orig_new_point]
    p0, p1 = seg['points']
    dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    seg['duration'] = (dist / seg['feed']) * 60 if seg['feed'] > 0 else 0.0

def apply_corner_compensation(segments, tool_radius):
    result = []
    n = len(segments)
    i = 0
    while i < n:
        seg = segments[i]
        result.append(seg)
        if seg['type'] != 'move':
            i += 1
            continue
        j = i + 1
        while j < n and segments[j]['type'] != 'move':
            j += 1
        if j >= n:
            i += 1
            continue
        segA, segB = seg, segments[j]
        if segA['comp_mode'] == segB['comp_mode'] and segA['comp_mode'] != 'G40':
            tin, tout = segA['tangent_end'], segB['tangent_start']
            cross = tin[0] * tout[1] - tin[1] * tout[0]
            if abs(cross) > 1e-9:
                offset_sign = +1 if segA['comp_mode'] == 'G41' else -1
                turn_sign = 1 if cross > 0 else -1
                V = segA['orig_end']
                if turn_sign != offset_sign:
                    Ei, Sj = segA['points'][-1], segB['points'][0]
                    a0 = np.arctan2(Ei[1] - V[1], Ei[0] - V[0])
                    a1 = np.arctan2(Sj[1] - V[1], Sj[0] - V[0])
                    if turn_sign > 0 and a1 < a0: a1 += 2 * np.pi
                    if turn_sign < 0 and a1 > a0: a1 -= 2 * np.pi
                    fillet_pts = _resample_arc(V, tool_radius, a0, a1, n_min=2)
                    arc_len = abs(a1 - a0) * tool_radius
                    feed_ref = segB.get('feed', segA.get('feed', 1000.0))
                    dur = (arc_len / feed_ref) * 60 if feed_ref > 0 else 0.0
                    result.append({
                        'type': 'move', 'kind': 'fillet',
                        'points': fillet_pts, 'duration': dur, 'feed': feed_ref,
                        'comp_mode': segA['comp_mode'],
                        'tangent_start': tin, 'tangent_end': tout,
                        'orig_start': V, 'orig_end': V,
                        'orig_points': [V, V],
                        'color_idx': segA.get('color_idx', 0)
                    })
                else:
                    if segA['kind'] == 'line' and segB['kind'] == 'line':
                        P = _line_line_intersection(segA['points'][0], segA['tangent_end'], segB['points'][0], segB['tangent_start'])
                    elif segA['kind'] == 'line' and segB['kind'] == 'arc':
                        P = _line_circle_intersection(segA['points'][0], segA['tangent_end'], segB['center'], segB['radius'], V)
                    elif segA['kind'] == 'arc' and segB['kind'] == 'line':
                        P = _line_circle_intersection(segB['points'][0], segB['tangent_start'], segA['center'], segA['radius'], V)
                    else:
                        P = _circle_circle_intersection(segA['center'], segA['radius'], segB['center'], segB['radius'], V)
                    if P is not None:
                        if segA['kind'] == 'arc': _trim_arc(segA, P, False)
                        else: _trim_line(segA, P, False, tool_radius)
                        if segB['kind'] == 'arc': _trim_arc(segB, P, True)
                        else: _trim_line(segB, P, True, tool_radius)
        i += 1
    return result

def split_by_wait(segments):
    stages = [[]]
    for seg in segments:
        if seg['type'] == 'wait': stages.append([])
        else: stages[-1].append(seg)
    return stages

def synchronize(channels_segments):
    stages_per_channel = [split_by_wait(segs) for segs in channels_segments]
    max_stage_count = max(len(s) for s in stages_per_channel)
    timelines = [[] for _ in channels_segments]
    current_time = [0.0] * len(channels_segments)
    for k in range(max_stage_count):
        stage_end_times = []
        for ci, stages in enumerate(stages_per_channel):
            if k < len(stages):
                t = current_time[ci]
                for seg in stages[k]:
                    dur = seg.get('duration', 0.0)
                    timelines[ci].append((t, t + dur, seg))
                    t += dur
                current_time[ci] = t
                stage_end_times.append(t)
        if stage_end_times:
            sync_t = max(stage_end_times)
            for ci, stages in enumerate(stages_per_channel):
                if k < len(stages): current_time[ci] = sync_t
    total_time = max(current_time) if current_time else 0.0
    return timelines, total_time

def _dwell_point(seg, key):
    if key == 'orig_points': return seg.get('orig_point', seg['point'])
    return seg['point']

def get_position(timeline, t, key='points'):
    if not timeline: return (0.0, 0.0)
    if t <= timeline[0][0]:
        first = timeline[0][2]
        return _dwell_point(first, key) if first['type'] == 'dwell' else first[key][0]
    for (t0, t1, seg) in timeline:
        if t0 <= t <= t1:
            if seg['type'] == 'dwell': return _dwell_point(seg, key)
            pts = seg[key]
            frac = 1.0 if t1 == t0 else (t - t0) / (t1 - t0)
            idx = frac * (len(pts) - 1)
            i0 = int(np.floor(idx))
            i1 = min(i0 + 1, len(pts) - 1)
            local_frac = idx - i0
            p0 = np.array(pts[i0]); p1 = np.array(pts[i1])
            return tuple(p0 + (p1 - p0) * local_frac)
    last = timeline[-1][2]
    return _dwell_point(last, key) if last['type'] == 'dwell' else last[key][-1]

def _seg_points(seg, key, n=None):
    pts = seg[key]
    limit = len(pts) if n is None else n
    brk = seg.get('orig_breaks') if key == 'orig_points' else None
    if not brk: return list(pts[:limit])
    brk = set(brk)
    out = []
    for k in range(limit):
        out.append(pts[k])
        if k in brk and k + 1 < limit: out.append((float('nan'), float('nan')))
    return out

def get_trail(timeline, t, key='points'):
    trail = []
    for (t0, t1, seg) in timeline:
        if seg['type'] == 'dwell':
            if t0 <= t: trail.append(_dwell_point(seg, key))
            continue
        pts = seg[key]
        if t >= t1: trail.extend(_seg_points(seg, key))
        elif t0 <= t < t1:
            frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            idx = frac * (len(pts) - 1)
            i0 = int(np.floor(idx))
            local_frac = idx - i0
            trail.extend(_seg_points(seg, key, i0 + 1))
            if (local_frac > 1e-9 and i0 + 1 < len(pts) and not (key == 'orig_points' and i0 in seg.get('orig_breaks', ()))):
                p0 = np.array(pts[i0]); p1 = np.array(pts[i0 + 1])
                trail.append(tuple(p0 + (p1 - p0) * local_frac))
            break
        else:
            break
    return trail

def _densify(points, max_step):
    out = [tuple(points[0])]
    for a, b in zip(points[:-1], points[1:]):
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(int(np.ceil(d / max_step)), 1)
        for k in range(1, n + 1): out.append((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
    return out

def _tool_contact_points(centers, mold, tool_radius, chunk=256):
    C = np.asarray(centers, float)
    A = mold[:-1]
    AB = mold[1:] - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-18)
    out = np.empty_like(C)
    dist = np.empty(len(C))
    for s in range(0, len(C), chunk):
        Pc = C[s:s + chunk]
        AP = Pc[:, None, :] - A[None, :, :]
        t = np.clip((AP * AB[None]).sum(2) / L2[None], 0.0, 1.0)
        proj = A[None] + t[..., None] * AB[None]
        diff = proj - Pc[:, None, :]
        d2 = (diff ** 2).sum(2)
        idx = d2.argmin(1)
        rows = np.arange(len(Pc))
        v = diff[rows, idx]
        d = np.sqrt(d2[rows, idx])
        safe = np.where(d > 1e-9, d, 1.0)
        out[s:s + chunk] = Pc + tool_radius * v / safe[:, None]
        out[s:s + chunk][d <= 1e-9] = Pc[d <= 1e-9]
        dist[s:s + chunk] = d
    return out, dist

def apply_geometric_contact(segments, mold_points, tool_radius, start_xy):
    mold = np.asarray(mold_points, float)
    step = max(0.2, tool_radius / 15.0)
    last_center = (float(start_xy[0]), float(start_xy[1]))
    out = []
    for seg in segments:
        if seg['type'] == 'move':
            seg = dict(seg)
            dense = _densify(seg['points'], step)
            P, d = _tool_contact_points(np.asarray(dense), mold, tool_radius)
            seg['points'] = dense
            seg['orig_points'] = [tuple(p) for p in P]
            seg['gap_min'] = float(d.min() - tool_radius)
            Cn = np.asarray(dense)
            d_c = np.hypot(*(Cn[1:] - Cn[:-1]).T)
            d_p = np.hypot(*(P[1:] - P[:-1]).T)
            jump = (d_p > 0.5 * tool_radius) & (d_p > 5.0 * np.maximum(d_c, 1e-9))
            seg['orig_breaks'] = [int(k) for k in np.where(jump)[0]]
            last_center = dense[-1]
        elif seg['type'] == 'dwell':
            seg = dict(seg)
            seg['point'] = last_center
            P, d = _tool_contact_points(np.asarray([last_center]), mold, tool_radius)
            seg['orig_point'] = tuple(P[0])
            seg['gap_min'] = float(d[0] - tool_radius)
        out.append(seg)
    return out

def _contact_chunks(seg):
    pts = seg['orig_points']
    breaks = set(seg.get('orig_breaks', []))
    chunks = []
    current = []
    for k, p in enumerate(pts):
        current.append(p)
        if k in breaks:
            if current: chunks.append(current)
            current = []
    if current: chunks.append(current)
    return chunks

def _run_simulation_data(x1, y1, tr1, fd1, code1,
                         x2, y2, tr2, fd2, code2,
                         x3, y3, tr3, fd3, code3,
                         m_x0, m_y0, m_code, contact_mode):
    mold_cmds = parse_gcode(m_code)
    mold_segs = simulate_channel(mold_cmds, m_x0, m_y0, 0.0, 1000.0)
    mold_points = [(m_x0, m_y0)]
    for seg in mold_segs:
        if seg['type'] == 'move':
            mold_points.extend(seg['points'])

    use_geo = (contact_mode == 'geometric')
    note = ""
    if use_geo and len(mold_points) < 2:
        use_geo = False
        note = "Uyarı: Kalıp G-kodu boş - programlanan yol kullanılıyor.\n"

    inputs = [
        {'x0': x1, 'y0': y1, 'tr': tr1, 'fd': fd1, 'code': code1},
        {'x0': x2, 'y0': y2, 'tr': tr2, 'fd': fd2, 'code': code2},
        {'x0': x3, 'y0': y3, 'tr': tr3, 'fd': fd3, 'code': code3}
    ]

    channels_segments = []
    start_points = []
    radii = []

    for idx, inp in enumerate(inputs):
        cmds = parse_gcode(inp['code'])
        segs = simulate_channel(cmds, inp['x0'], inp['y0'], inp['tr'], inp['fd'], y_offset=0.0)
        segs = apply_corner_compensation(segs, inp['tr'])
        if use_geo:
            segs = apply_geometric_contact(segs, mold_points, inp['tr'], (inp['x0'], inp['y0']))
            gaps = [s['gap_min'] for s in segs if 'gap_min' in s]
            if gaps and min(gaps) < -0.05:
                note += f"Uyarı: Kanal {idx + 1} takım dairesi kalıba giriyor (en küçük boşluk = {min(gaps):.2f}).\n"
        channels_segments.append(segs)
        start_points.append((inp['x0'], inp['y0']))
        radii.append(inp['tr'])

    return channels_segments, start_points, radii, mold_points, note

def run_all_gradio_analyses(x1, y1, tr1, fd1, code1,
                            x2, y2, tr2, fd2, code2,
                            x3, y3, tr3, fd3, code3,
                            m_x0, m_y0, m_code, contact_mode,
                            ac_start, ac_end, ac_step):
    try:
        channels_segments, start_points, radii, mold_points, note = _run_simulation_data(
            x1, y1, tr1, fd1, code1,
            x2, y2, tr2, fd2, code2,
            x3, y3, tr3, fd3, code3,
            m_x0, m_y0, m_code, contact_mode
        )

        fig_comp, ax_comp = plt.subplots(figsize=(10, 8), dpi=100)
        ax_comp.grid(True, linestyle='--', alpha=0.5)
        ax_comp.set_aspect('equal')
        ax_comp.set_title('Genel Kontur Karşılaştırması', fontsize=12)
        if len(mold_points) > 1:
            ax_comp.plot([p[0] for p in mold_points], [p[1] for p in mold_points], color='black', linewidth=1.5, label='Kalıp Konturu')
        for i in range(3):
            for seg in channels_segments[i]:
                is_g0 = seg.get('feed', 1000.0) >= 3000.0
                if seg['type'] == 'move' and not is_g0 and seg['orig_points']:
                    c_idx = seg.get('color_idx', 0)
                    for pts in _contact_chunks(seg):
                        ax_comp.plot([p[0] for p in pts], [p[1] for p in pts], color=CHANNEL_COLORS[c_idx % 2], linewidth=2.0)
            ax_comp.plot(start_points[i][0], start_points[i][1], 'o', color=CHANNEL_COLORS[0], markersize=6)
        ax_comp.legend(loc='upper right', fontsize=8)
        plt.tight_layout()

        svg_outputs = []
        mold = np.asarray(mold_points, float)
        A = mold[:-1]
        AB = mold[1:] - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-18)

        def get_closest_mold_point_and_dist(point):
            p = np.array(point, float)
            ap = p - A
            t = np.clip((ap * AB).sum(1) / L2, 0.0, 1.0)
            proj = A + t[:, None] * AB
            dists = np.sqrt(((proj - p) ** 2).sum(1))
            min_idx = dists.argmin()
            return tuple(proj[min_idx]), float(dists[min_idx])

        for i in range(3):
            fig_ind, ax_ind = plt.subplots(figsize=(8, 6), dpi=100)
            ax_ind.grid(True, linestyle='--', alpha=0.5)
            ax_ind.set_aspect('equal')
            ax_ind.set_title(f'Kanal {i+1} Detaylı Kontur ve Kalıp Mesafesi', fontsize=12)
            if len(mold_points) > 1:
                ax_ind.plot([p[0] for p in mold_points], [p[1] for p in mold_points], color='black', linewidth=1.5, label='Kalıp Konturu')

            for seg in channels_segments[i]:
                is_g0 = seg.get('feed', 1000.0) >= 3000.0
                if seg['type'] == 'move' and not is_g0 and seg['orig_points']:
                    c_idx = seg.get('color_idx', 0)
                    for pts in _contact_chunks(seg):
                        ax_ind.plot([p[0] for p in pts], [p[1] for p in pts], color=CHANNEL_COLORS[c_idx % 2], linewidth=2.5)
                    pts = seg['orig_points']
                    n = len(pts)
                    key_points = [pts[0], pts[n // 2], pts[-1]]
                    for kp in key_points:
                        closest_p, d = get_closest_mold_point_and_dist(kp)
                        ax_ind.plot([kp[0], closest_p[0]], [kp[1], closest_p[1]], color='red', linestyle=':', linewidth=1.0)
                        ax_ind.plot(kp[0], kp[1], 'o', color='red', markersize=5)
                        ax_ind.text(kp[0] + 1.0, kp[1] + 1.0, f"{d:.2f}mm", color='darkred', fontsize=7)
            ax_ind.plot(start_points[i][0], start_points[i][1], 'o', color=CHANNEL_COLORS[0], markersize=6, label='Başlangıç')
            ax_ind.legend(loc='upper right', fontsize=8)
            plt.tight_layout()

            svg_io = io.StringIO()
            fig_ind.savefig(svg_io, format='svg', bbox_inches='tight')
            plt.close(fig_ind)
            svg_data = svg_io.getvalue()

            svg_container = f"""
            <div id="svg-wrapper-{i}" class="custom-fullscreen-container" style="width:100%; display:flex; flex-direction:column; align-items:center; padding:10px; background-color:white; border-radius:8px; border: 1px solid #ddd; position:relative;">
                <button onclick=\"var elem = document.getElementById('svg-wrapper-{i}'); if (!document.fullscreenElement) { elem.requestFullscreen().catch(err => { alert('Error: ' + err.message); }); } else { document.exitFullscreen(); }\" style="margin-bottom:10px; padding:8px 16px; font-size:12px; font-weight:bold; cursor:pointer; background-color:#0072B2; color:white; border:none; border-radius:4px;">
                    🔍 Tam Ekran Yap / Çık
                </button>
                <div style="width:100%; display:flex; justify-content:center; align-items:center; background-color:white;">
                    {svg_data}
                </div>
            </div>
            """
            svg_outputs.append(svg_container)

        distance_rows = []
        for i in range(3):
            for s_idx, seg in enumerate(channels_segments[i]):
                is_g0 = seg.get('feed', 1000.0) >= 3000.0
                if seg['type'] == 'move' and not is_g0 and seg['orig_points']:
                    pts = seg['orig_points']
                    n = len(pts)
                    d_start = get_closest_mold_point_and_dist(pts[0])[1]
                    d_mid = get_closest_mold_point_and_dist(pts[n // 2])[1]
                    d_end = get_closest_mold_point_and_dist(pts[-1])[1]
                    distance_rows.append({
                        'Kanal': f"Kanal {i+1}",
                        'Segment No': f"Kod: {seg.get('color_idx', s_idx)}",
                        'Başlangıç (mm)': round(d_start, 4),
                        'Orta (mm)': round(d_mid, 4),
                        'Bitiş (mm)': round(d_end, 4)
                    })
        df_dist = pd.DataFrame(distance_rows)

        timelines, total_time = synchronize(channels_segments)
        ac_circles = [f"; Kanal Çember Kodları (Zaman: {ac_start}s - {ac_end}s, Adım: {ac_step}s)"]
        for t in np.arange(ac_start, ac_end + ac_step, ac_step):
            t_clamped = max(0.0, min(t, total_time))
            for i in range(3):
                center_pos = get_position(timelines[i], t_clamped, 'points')
                ac_circles.append(f"_CIRCLE {center_pos[0]:.4f},{center_pos[1]:.4f} {radii[i]:.4f}")
        ac_circles_txt = "\n".join(ac_circles)

        ac_paths = []
        for i in range(3):
            ac_paths.append(f"; Kanal {i+1} İşlenmiş Kontur Kodları")
            for seg in channels_segments[i]:
                if seg['type'] == 'move' and seg['orig_points']:
                    for pts in _contact_chunks(seg):
                        points_str = " ".join([f"{q[0]:.4f},{q[1]:.4f}" for q in pts])
                        ac_paths.append(f"_PLINE {points_str}")
            ac_paths.append("")
        ac_paths_txt = "\n".join(ac_paths)

        fps = 20
        n_frames = max(int(total_time * fps), 2)
        video_filename = "gradio_temp_simulation.mp4"

        all_x, all_y = [p[0] for p in mold_points], [p[1] for p in mold_points]
        for segs, (sx, sy) in zip(channels_segments, start_points):
            all_x.append(sx); all_y.append(sy)
            for seg in segs:
                if seg['type'] == 'move':
                    all_x.extend(p[0] for p in seg['orig_points'])
                    all_y.extend(p[1] for p in seg['orig_points'])
        margin = max(10.0, max(radii) * 2)
        xmin, xmax = min(all_x) - margin, max(all_x) + margin
        ymin, ymax = min(all_y) - margin, max(all_y) + margin

        fig_v, ax_v = plt.subplots(figsize=(8.0, 6.0), dpi=90)
        fig_v.subplots_adjust(right=0.75)
        ax_v.set_xlim(xmin, xmax)
        ax_v.set_ylim(ymin, ymax)
        ax_v.set_aspect('equal')
        ax_v.grid(True, alpha=0.3)

        if len(mold_points) > 1:
            ax_v.plot([p[0] for p in mold_points], [p[1] for p in mold_points], color='black', linewidth=1.0, linestyle='-')

        contact_lines, tool_circles, spoke_lines, contact_dots, tool_centers = [], [], [], [], []
        for k in range(3):
            (cline,) = ax_v.plot([], [], color=CHANNEL_COLORS[k], linewidth=2.2, label=f'Kanal {k+1}')
            contact_lines.append(cline)
            circ = plt.Circle((0, 0), radii[k], facecolor=CHANNEL_COLORS[k], alpha=0.22, edgecolor=CHANNEL_COLORS[k], linewidth=1.8)
            ax_v.add_patch(circ)
            tool_circles.append(circ)
            (spoke,) = ax_v.plot([], [], color=CHANNEL_COLORS[k], linewidth=1.2)
            spoke_lines.append(spoke)
            (dot,) = ax_v.plot([], [], 'o', color=CHANNEL_COLORS[k], markersize=5)
            contact_dots.append(dot)
            (center,) = ax_v.plot([], [], '+', color=CHANNEL_COLORS[k], markersize=7, markeredgewidth=1.5)
            tool_centers.append(center)

        ax_v.legend(loc='upper left', bbox_to_anchor=(1.04, 1.0), borderaxespad=0, fontsize=8)
        time_text = ax_v.text(0.02, 0.98, '', transform=ax_v.transAxes, va='top', fontsize=9)

        def render_mp4_frame(frame):
            t_val = (frame / n_frames) * total_time
            for k in range(3):
                center_pos = get_position(timelines[k], t_val, 'points')
                contact_pos = get_position(timelines[k], t_val, 'orig_points')
                contact_trail = get_trail(timelines[k], t_val, 'orig_points')
                if contact_trail:
                    contact_lines[k].set_data([p[0] for p in contact_trail], [p[1] for p in contact_trail])
                tool_circles[k].center = center_pos
                tool_centers[k].set_data([center_pos[0]], [center_pos[1]])
                contact_dots[k].set_data([contact_pos[0]], [contact_pos[1]])
                spoke_lines[k].set_data([center_pos[0], contact_pos[0]], [center_pos[1], contact_pos[1]])
            time_text.set_text(f't = {t_val:.2f} / {total_time:.2f} s')
            fig_v.canvas.draw()
            img_argb = np.frombuffer(fig_v.canvas.tostring_argb(), dtype=np.uint8)
            img = img_argb.reshape(fig_v.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            return img

        frames = [render_mp4_frame(f_idx) for f_idx in range(n_frames)]
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(video_filename, fps=fps, codec='libx264', logger=None)
        plt.close(fig_v)

        if not note:
            note = "Simülasyon başarıyla tamamlandı! Herhangi bir kritik çakışma tespit edilmedi."

        return (
            fig_comp,
            svg_outputs[0], svg_outputs[1], svg_outputs[2],
            df_dist,
            ac_circles_txt, ac_paths_txt,
            video_filename,
            note
        )
    except Exception as e:
        import traceback
        err_str = f"Hata Detayı:\n{traceback.format_exc()}"
        dummy_fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Hata Oluştu", ha='center', va='center')
        return (dummy_fig, err_str, err_str, err_str, pd.DataFrame(), err_str, err_str, None, err_str)

def load_gradio_json_config(file_obj):
    if file_obj is None: return [gr.update()]*17
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        updates = []
        channels = config_data.get('channels', [])
        ch1 = channels[0] if len(channels) > 0 else {}
        ch2 = channels[1] if len(channels) > 1 else {}
        ch3 = channels[2] if len(channels) > 2 else {}
        updates.extend([
            gr.update(value=ch1.get('x0', -275.0)), gr.update(value=ch1.get('y0', -100.0)),
            gr.update(value=ch1.get('tr', 15.0)), gr.update(value=ch1.get('fd', 800.0)), gr.update(value=ch1.get('code', ''))
        ])
        updates.extend([
            gr.update(value=ch2.get('x0', -286.075)), gr.update(value=ch2.get('y0', -99.5)),
            gr.update(value=ch2.get('tr', 15.0)), gr.update(value=ch2.get('fd', 800.0)), gr.update(value=ch2.get('code', ''))
        ])
        updates.extend([
            gr.update(value=ch3.get('x0', -275.0)), gr.update(value=ch3.get('y0', -100.0)),
            gr.update(value=ch3.get('tr', 15.0)), gr.update(value=ch3.get('fd', 800.0)), gr.update(value=ch3.get('code', ''))
        ])
        mold = config_data.get('mold', {})
        updates.extend([
            gr.update(value=mold.get('x0', -30.0)), gr.update(value=mold.get('y0', -30.0)), gr.update(value=mold.get('code', ''))
        ])
        settings = config_data.get('settings', {})
        updates.append(gr.update(value=settings.get('contact_mode', 'geometric')))
        return updates
    except Exception as e:
         raise gr.Error(f"JSON yüklenirken hata oluştu: {str(e)}")

with gr.Blocks(title="Tam Entegre CNC Simülatörü") as demo:
    gr.Markdown("# 🚀 CNC Simülatör Web Kontrol ve Analiz Merkezi")
    gr.Markdown("Notebook'ta bulunan tüm bağımsız grafik, analiz ve video üretici hücrelerini tek bir panelde birleştirir.\n\n*(Doğrudan gömülü SVG render mimarisi aktiftir.)*\n\nGrafikleri **🔍 Tam Ekran Yap / Çık** butonu ile tarayıcı sınırlarında maksimum büyüklükte inceleyebilirsiniz.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Girdi ve Yapılandırma Paneli")
            with gr.Tabs():
                with gr.TabItem("Kanal 1"):
                    x1 = gr.Number(value=-275.0, label="Başlangıç X")
                    y1 = gr.Number(value=-100.0, label="Başlangıç Y")
                    tr1 = gr.Number(value=15.0, label="Takım Yarıçapı")
                    fd1 = gr.Number(value=800.0, label="Varsayılan F")
                    code1 = gr.Textbox(value=demo_gcode[0], lines=5, label="G-Kodu")
                with gr.TabItem("Kanal 2"):
                    x2 = gr.Number(value=-286.075, label="Başlangıç X")
                    y2 = gr.Number(value=-99.5, label="Başlangıç Y")
                    tr2 = gr.Number(value=15.0, label="Takım Yarıçapı")
                    fd2 = gr.Number(value=800.0, label="Varsayılan F")
                    code2 = gr.Textbox(value=demo_gcode[1], lines=5, label="G-Kodu")
                with gr.TabItem("Kanal 3"):
                    x3 = gr.Number(value=-275.0, label="Başlangıç X")
                    y3 = gr.Number(value=-100.0, label="Başlangıç Y")
                    tr3 = gr.Number(value=15.0, label="Takım Yarıçapı")
                    fd3 = gr.Number(value=800.0, label="Varsayılan F")
                    code3 = gr.Textbox(value=demo_gcode[2], lines=5, label="G-Kodu")
            with gr.Group():
                gr.Markdown("**Kalıp (Sabit Path) Ayarları**")
                m_x0 = gr.Number(value=-30.0, label="Kalıp Başlangıç X")
                m_y0 = gr.Number(value=-30.0, label="Kalıp Başlangıç Y")
                m_code = gr.Textbox(value="G1 X30 Y-30 F1000\nG1 X30 Y30\nG2 X-30 Y30 I-30 J0\nG1 X-30 Y-30", lines=3, label="Kalıp G-Kodu")
                contact_mode = gr.Dropdown(choices=["geometric", "programmed"], value="geometric", label="Çizilen Yol Modu")
            with gr.Group():
                gr.Markdown("**AutoCAD Ayarları**")
                ac_start = gr.Number(value=0.0, label="AutoCAD Başlangıç Zamanı (sn)")
                ac_end = gr.Number(value=10.0, label="AutoCAD Bitiş Zamanı (sn)")
                ac_step = gr.Number(value=0.1, label="AutoCAD Zaman Adımı (sn)")
            with gr.Group():
                gr.Markdown("**Konfigürasyon Yükle (JSON)**")
                json_upload = gr.File(file_count="single", file_types=[".json"], label="JSON Ayar Dosyasını Sürükleyin")
            run_btn = gr.Button("Simülasyon ve Analizleri Çalıştır", variant="primary")
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Tüm Analizler ve Çıktı Sekmeleri")
            with gr.Tabs() as out_tabs:
                with gr.TabItem("🎥 Simülasyon Video Oynatıcı & MP4"):
                    gr.Markdown("Simülasyonun hareketli oynatıcısı. Sağ alt köşedeki üç noktadan videoyu MP4 formatında indirebilirsiniz.")
                    video_out = gr.Video(label="Simülasyon Oynatıcı (MP4)", format="mp4")
                with gr.TabItem("🔍 Genel Karşılaştırma"):
                    plot_comp_out = gr.Plot(label="Genel Karşılaştırma Grafiği")
                with gr.TabItem("📐 Bireysel Kanal Analizi"):
                    gr.Markdown("Kanalların detaylandırılmış hassas ölçüm grafikleri (Doğrudan entegre vektörel SVG formatında anında yüklenir).")
                    with gr.Tabs():
                        with gr.TabItem("Kanal 1"):
                            plot_ind_1 = gr.HTML()
                        with gr.TabItem("Kanal 2"):
                            plot_ind_2 = gr.HTML()
                        with gr.TabItem("Kanal 3"):
                            plot_ind_3 = gr.HTML()
                with gr.TabItem("📏 Mesafe Tablosu"):
                    gr.Markdown("Segmentlerin Kalıp Konturuna Olan Başlangıç, Orta ve Bitiş Mesafeleri")
                    table_out = gr.DataFrame()
                with gr.TabItem("💻 AutoCAD Çember Kodları"):
                    gr.Markdown("Aşağıdaki AutoCAD çember komutlarını kopyalayarak AutoCAD komut satırına doğrudan yapıştırabilirsiniz.")
                    ac_circles_out = gr.TextArea(label="Çember Kodları (Kanal Bazlı)", lines=12)
                with gr.TabItem("🔀 AutoCAD Path Kodları"):
                    gr.Markdown("Aşağıdaki Çoklu Çizgi (_PLINE) kodlarını kopyalayarak AutoCAD'e aktarabilirsiniz.")
                    ac_paths_out = gr.TextArea(label="Çizgi Yolu Kodları (PLINE)", lines=12)
            log_output = gr.Textbox(label="Sistem Durum Raporu / Uyarılar", lines=5)

    json_upload.change(
        fn=load_gradio_json_config,
        inputs=[json_upload],
        outputs=[x1, y1, tr1, fd1, code1,
                 x2, y2, tr2, fd2, code2,
                 x3, y3, tr3, fd3, code3,
                 m_x0, m_y0, m_code, contact_mode]
    )

    run_btn.click(
        fn=run_all_gradio_analyses,
        inputs=[x1, y1, tr1, fd1, code1,
                x2, y2, tr2, fd2, code2,
                x3, y3, tr3, fd3, code3,
                m_x0, m_y0, m_code, contact_mode,
                ac_start, ac_end, ac_step],
        outputs=[plot_comp_out,
                 plot_ind_1, plot_ind_2, plot_ind_3,
                 table_out,
                 ac_circles_out, ac_paths_out,
                 video_out,
                 log_output]
    )

demo.launch()
