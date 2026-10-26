import os
import matplotlib
matplotlib.use("Agg")   # sunucuda ekransız çizim

# ===== Hücre 2 =====
# [render] kaldırıldı: !pip install gradio moviepy -q


# ===== Hücre 4 =====
import numpy as np
import re
import io


# -------------------------------------------------------------
def parse_gcode(text):
    """G-kodu metnini satır satır ayrıştırır, komut sözlükleri üretir."""
    commands = []
    for raw_line in text.strip().split('\n'):
        line = raw_line.split(';')[0].strip()  # yorumları at
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
    """sign=+1 -> yön vektörünü +90° (CCW), sign=-1 -> -90° (CW) döndürür."""
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
    """
    Bir kanalın G-kodu komutlarını işleyip hareket segmentleri listesi üretir.
    y_offset: G-kodundaki tüm Y değerlerine eklenecek ofset miktarı.
    """
    x, y = start_x, start_y
    motion_mode = 'G0'
    comp_mode = 'G40'
    feed = default_feed
    segments = []

    # Renklendirme için hareket içeren komutları sayacağız
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
        # Hedef Y değerine kullanıcı tarafından girilen offset dinamik olarak ekleniyor
        target_y = cmd.get('Y', y - y_offset) + y_offset

        if np.isclose(x, target_x) and np.isclose(y, target_y):
            continue

        active_comp = comp_mode if comp_mode in ('G41', 'G42') else 'G40'

        # Her geçerli geometrik harekete sırayla artan bir indeks veriyoruz
        seg_color_idx = move_counter
        move_counter += 1

        # --- Doğrusal hareket (G0 / G1) ---
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

        # --- Dairesel hareket (G2/G3) ---
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
                # Dairesel yay merkez koordinatı da Y offsete göre dinamik kaydırılıyor
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


# ===== Hücre 6 =====
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
    while ang - ref > np.pi:
        ang -= 2 * np.pi
    while ang - ref < -np.pi:
        ang += 2 * np.pi
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
                    # --- DIŞBÜKEY: fillet yayı ekle ---
                    Ei, Sj = segA['points'][-1], segB['points'][0]
                    a0 = np.arctan2(Ei[1] - V[1], Ei[0] - V[0])
                    a1 = np.arctan2(Sj[1] - V[1], Sj[0] - V[0])
                    if turn_sign > 0 and a1 < a0:
                        a1 += 2 * np.pi
                    if turn_sign < 0 and a1 > a0:
                        a1 -= 2 * np.pi
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
                        'color_idx': segA.get('color_idx', 0) # Bir önceki hareketin rengini sürdür
                    })
                else:
                    # --- İÇBÜKEY ---
                    if segA['kind'] == 'line' and segB['kind'] == 'line':
                        P = _line_line_intersection(segA['points'][0], segA['tangent_end'],
                                                     segB['points'][0], segB['tangent_start'])
                    elif segA['kind'] == 'line' and segB['kind'] == 'arc':
                        P = _line_circle_intersection(segA['points'][0], segA['tangent_end'],
                                                       segB['center'], segB['radius'], V)
                    elif segA['kind'] == 'arc' and segB['kind'] == 'line':
                        P = _line_circle_intersection(segB['points'][0], segB['tangent_start'],
                                                       segA['center'], segA['radius'], V)
                    else:
                        P = _circle_circle_intersection(segA['center'], segA['radius'],
                                                          segB['center'], segB['radius'], V)
                    if P is not None:
                        if segA['kind'] == 'arc':
                            _trim_arc(segA, P, False)
                        else:
                            _trim_line(segA, P, False, tool_radius)
                        if segB['kind'] == 'arc':
                            _trim_arc(segB, P, True)
                        else:
                            _trim_line(segB, P, True, tool_radius)
        i += 1

    return result


# ===== Hücre 8 =====
def split_by_wait(segments):
    """Segment listesini WAIT komutlarına göre aşamalara (stage) böler."""
    stages = [[]]
    for seg in segments:
        if seg['type'] == 'wait':
            stages.append([])
        else:
            stages[-1].append(seg)
    return stages


def synchronize(channels_segments):
    """
    3 kanalın segment listelerini alır, WAIT noktalarında birbirini
    bekleyecek şekilde mutlak zaman çizelgesi (timeline) üretir.
    Döndürür: (timelines, toplam_sure)
      timelines[i] = [(t_baslangic, t_bitis, segment), ...]
    """
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
            sync_t = max(stage_end_times)  # en yavaş kanalı bekle
            for ci, stages in enumerate(stages_per_channel):
                if k < len(stages):
                    current_time[ci] = sync_t

    total_time = max(current_time) if current_time else 0.0
    return timelines, total_time


def _dwell_point(seg, key):
    """G4 (bekleme) segmentinde konum: takım merkezi için 'point',
    temas noktası için (varsa) 'orig_point'."""
    if key == 'orig_points':
        return seg.get('orig_point', seg['point'])
    return seg['point']


def get_position(timeline, t, key='points'):
    """Verilen zamanda kanalın konumunu (x,y) döndürür (ara değerleme ile).
    key='points' -> takım merkezi yolu, key='orig_points' -> takım kenarının
    (kalıp konturuna değdiği) temas noktası yolu."""
    if not timeline:
        return (0.0, 0.0)
    if t <= timeline[0][0]:
        first = timeline[0][2]
        return _dwell_point(first, key) if first['type'] == 'dwell' else first[key][0]
    for (t0, t1, seg) in timeline:
        if t0 <= t <= t1:
            if seg['type'] == 'dwell':
                return _dwell_point(seg, key)
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
    """Segmentin noktalarını döndürür; temas noktası izinde (orig_points)
    sıçrama (orig_breaks) varsa araya NaN koyar (çizimde çizgi kopuk görünür).
    n verilirse sadece ilk n nokta alınır."""
    pts = seg[key]
    limit = len(pts) if n is None else n
    brk = seg.get('orig_breaks') if key == 'orig_points' else None
    if not brk:
        return list(pts[:limit])
    brk = set(brk)
    out = []
    for k in range(limit):
        out.append(pts[k])
        if k in brk and k + 1 < limit:
            out.append((float('nan'), float('nan')))
    return out


def get_trail(timeline, t, key='points'):
    """Verilen zamana kadar kat edilen tüm noktaları (iz) döndürür.
    key='points' -> takım merkezi izi, key='orig_points' -> temas noktası izi."""
    trail = []
    for (t0, t1, seg) in timeline:
        if seg['type'] == 'dwell':
            if t0 <= t:
                trail.append(_dwell_point(seg, key))
            continue
        pts = seg[key]
        if t >= t1:
            trail.extend(_seg_points(seg, key))
        elif t0 <= t < t1:
            frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            idx = frac * (len(pts) - 1)
            i0 = int(np.floor(idx))
            local_frac = idx - i0
            trail.extend(_seg_points(seg, key, i0 + 1))
            # Segmentin ANLIK (henüz tamamlanmamış) noktasını da ekle; aksi
            # halde 2 noktalı doğru segmentlerinde (i0 her zaman 0 kalır)
            # segment bitene kadar çizgi ekranda hiç ilerlemez.
            if (local_frac > 1e-9 and i0 + 1 < len(pts)
                    and not (key == 'orig_points' and i0 in seg.get('orig_breaks', ()))):
                p0 = np.array(pts[i0]); p1 = np.array(pts[i0 + 1])
                trail.append(tuple(p0 + (p1 - p0) * local_frac))
            break
        else:
            break
    return trail


# ===== Hücre 10 =====
# -------------------------------------------------------------
# 5b) GEOMETRİK TEMAS NOKTASI (kompanzasyondan bağımsız)
# -------------------------------------------------------------

def _densify(points, max_step):
    out = [tuple(points[0])]
    for a, b in zip(points[:-1], points[1:]):
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(int(np.ceil(d / max_step)), 1)
        for k in range(1, n + 1):
            out.append((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
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
    """Segment içindeki NaN kırılma noktalarına göre temas noktalarını gruplara ayırır."""
    pts = seg['orig_points']
    breaks = set(seg.get('orig_breaks', []))
    chunks = []
    current = []
    for k, p in enumerate(pts):
        current.append(p)
        if k in breaks:
            if current:
                chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


# ===== Hücre 12 =====
import json, os, io, re, html, uuid, time, tempfile, threading, traceback, pprint
import numpy as np
import pandas as pd

CHANNEL_COLORS = ['#0072B2', '#D55E00', '#0072B2']   # renk körü dostu 2 renk (sırayla)
N_CH = 3

# -------------------------------------------------------------
# VARSAYILAN DEĞERLER  (JSON ile kaydettiğin değerleri istersen
# arayüzdeki "Varsayılanları Python kodu olarak al" butonuyla buraya yapıştırabilirsin)
# -------------------------------------------------------------
DEMO_GCODE = [
    "G1 X0 Y0 F800\nG1 X40 Y0\nG41\nG1 X40 Y30\nG2 X20 Y50 R20\nG1 X0 Y50\nG40\nWAIT\nG1 X0 Y0",
    "G1 X60 Y60 F600\nWAIT\nG1 X10 Y10\nG3 X30 Y10 R-15\nG1 X10 Y10",
    "G1 X-20 Y-20 F1000\nG4 P1\nWAIT\nG1 X-20 Y40",
]

DEFAULT_CHANNELS = [
    {'x0': -275.0,   'y0': -100.0, 'tr': 15.0, 'fd': 800.0, 'y_offset': 0.0, 'code': DEMO_GCODE[0],
     'dia_mode': 'radius', 'dia_axis': 'X'},
    {'x0': -286.075, 'y0': -99.5,  'tr': 15.0, 'fd': 800.0, 'y_offset': 0.0, 'code': DEMO_GCODE[1],
     'dia_mode': 'radius', 'dia_axis': 'X'},
    {'x0': -275.0,   'y0': -100.0, 'tr': 15.0, 'fd': 800.0, 'y_offset': 0.0, 'code': DEMO_GCODE[2],
     'dia_mode': 'radius', 'dia_axis': 'X'},
]
DEFAULT_MOLD = {
    'x0': -30.0, 'y0': -30.0,
    'code': "G1 X30 Y-30 F1000\nG1 X30 Y30\nG2 X-30 Y30 I-30 J0\nG1 X-30 Y-30",
    'dia_mode': 'radius', 'dia_axis': 'X',
}
DEFAULT_CONTACT_MODE = 'geometric'
CONTACT_MODE_CHOICES = [
    ('Geometrik: takım dairesi ile kalıp arasındaki en dar nokta (G40/G41/G42 fark etmez)', 'geometric'),
    ('Programlanan: G41/G42 konturu, G40 takım merkezi (eski davranış)', 'programmed'),
]

DIA_MODE_CHOICES = [('Yarıçap', 'radius'), ('Çap (÷2)', 'diameter')]
DIA_AXIS_CHOICES = [('X ekseni', 'X'), ('Y ekseni', 'Y'), ('X ve Y', 'XY')]

CH_STRIDE = 8                       # kanal başına bileşen: x0,y0,tr,fd,y_offset,code,dia_mode,dia_axis
MOLD_START = N_CH * CH_STRIDE       # 24: kalıp x0,y0,code,dia_mode,dia_axis
N_CFG = MOLD_START + 5 + 1          # 30 (+ çizilen yol modu)


def _num(v, default=0.0):
    """Boş bırakılan / bozuk sayı alanlarını güvenle float'a çevirir."""
    try:
        if v is None:
            return default
        f = float(v)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _dia(mode, axis):
    return (mode if mode in ('radius', 'diameter') else 'radius',
            axis if axis in ('X', 'Y', 'XY') else 'X')


def cfg_from_values(v):
    """Gradio bileşen değerleri listesi (N_CFG eleman) -> konfigürasyon sözlüğü."""
    channels = []
    for k in range(N_CH):
        b = k * CH_STRIDE
        dm, da = _dia(v[b + 6], v[b + 7])
        channels.append({
            'x0': _num(v[b]), 'y0': _num(v[b + 1]), 'tr': _num(v[b + 2], 10.0),
            'fd': _num(v[b + 3], 1000.0), 'y_offset': _num(v[b + 4]),
            'code': v[b + 5] or '', 'dia_mode': dm, 'dia_axis': da,
        })
    m = MOLD_START
    dm, da = _dia(v[m + 3], v[m + 4])
    mold = {'x0': _num(v[m]), 'y0': _num(v[m + 1]), 'code': v[m + 2] or '', 'dia_mode': dm, 'dia_axis': da}
    mode = v[m + 5] if v[m + 5] in ('geometric', 'programmed') else 'geometric'
    return {'channels': channels, 'mold': mold, 'contact_mode': mode}


# -------------------------------------------------------------
# ÇAP / YARIÇAP GİRİŞİ: seçilen eksendeki değerleri 2'ye böler
# -------------------------------------------------------------
def _fmt_num(v):
    s = f"{v:.6f}".rstrip('0').rstrip('.')
    return '0' if s in ('', '-0') else s


def apply_diameter(code, x0, y0, mode, axis):
    """mode='diameter' ise seçilen eksenin (X, Y veya ikisi) G-kodu koordinatlarını ve başlangıç
    değerini 2'ye böler. I/J/R değerleri (zaten yarıçap/mesafe) ve Y offset değişmez.
    Döndürür: (yeni_kod, x0, y0)"""
    if mode != 'diameter':
        return code, x0, y0
    axes = set(axis) if axis in ('X', 'Y', 'XY') else set()

    def repl(m):
        if m.group(1).upper() in axes:
            return f"{m.group(1)}{m.group(2)}{_fmt_num(float(m.group(3)) / 2.0)}"
        return m.group(0)

    out = []
    for line in code.split('\n'):
        body, sep, comment = line.partition(';')
        if not body.strip().upper().startswith('WAIT'):
            body = re.sub(r'(?<![A-Za-z])([XYxy])(\s*)(-?\d+\.?\d*)', repl, body)
        out.append(body + sep + comment)
    return "\n".join(out), (x0 / 2.0 if 'X' in axes else x0), (y0 / 2.0 if 'Y' in axes else y0)


def _dia_note(label, d):
    if d['dia_mode'] != 'diameter':
        return ''
    ax = {'X': 'X', 'Y': 'Y', 'XY': 'X ve Y'}[d['dia_axis']]
    return f"{label}: çap girişi ({ax} ekseni) — koordinatlar ve başlangıç değeri ÷2 uygulandı."


# -------------------------------------------------------------
# Widget'tan bağımsız build_channels (Colab'daki ile aynı mantık)
# -------------------------------------------------------------
def get_mold_points_cfg(mold):
    code, x0, y0 = apply_diameter(mold['code'], mold['x0'], mold['y0'],
                                  mold.get('dia_mode'), mold.get('dia_axis'))
    mold_cmds = parse_gcode(code)
    mold_segs = simulate_channel(mold_cmds, x0, y0, 0.0, 1000.0)
    pts = [(x0, y0)]
    for seg in mold_segs:
        if seg['type'] == 'move':
            pts.extend(seg['points'])
    return pts


def build_channels_cfg(cfg):
    mold_points = get_mold_points_cfg(cfg['mold'])
    use_geo = (cfg['contact_mode'] == 'geometric')
    notes = []
    n = _dia_note('Kalıp', cfg['mold'])
    if n:
        notes.append(n)
    if use_geo and len(mold_points) < 2:
        use_geo = False
        notes.append("Uyarı: Kalıp G-kodu boş (tek nokta) - geometrik temas noktası "
                     "hesaplanamadı, programlanan yol kullanılıyor.")
    channels_segments, start_points, radii = [], [], []
    for ci, ch in enumerate(cfg['channels']):
        code, x0, y0 = apply_diameter(ch['code'], ch['x0'], ch['y0'], ch.get('dia_mode'), ch.get('dia_axis'))
        n = _dia_note(f"Kanal {ci + 1}", ch)
        if n:
            notes.append(n)
        cmds = parse_gcode(code)
        segs = simulate_channel(cmds, x0, y0, ch['tr'], ch['fd'], y_offset=ch['y_offset'])
        segs = apply_corner_compensation(segs, ch['tr'])
        if use_geo:
            segs = apply_geometric_contact(segs, mold_points, ch['tr'], (x0, y0))
            gaps = [s['gap_min'] for s in segs if 'gap_min' in s]
            if gaps and min(gaps) < -0.05:
                notes.append(f"Uyarı: Kanal {ci + 1} takım dairesi kalıba giriyor "
                             f"(en küçük boşluk = {min(gaps):.2f}). Takım yarıçapını / yolu / G41-G42 yönünü kontrol et.")
        channels_segments.append(segs)
        start_points.append((x0, y0))
        radii.append(ch['tr'])
    return channels_segments, start_points, radii, mold_points, "\n".join(notes)


# ===== Hücre 14 =====
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
from PIL import Image


def _is_g0(seg):
    return seg.get('feed', 1000.0) >= 3000.0


def _plot_limits(mold_points, start_points, radii, margin_mult):
    all_x = [p[0] for p in mold_points] + [s[0] for s in start_points]
    all_y = [p[1] for p in mold_points] + [s[1] for s in start_points]
    margin = max(15.0, max(radii) * margin_mult)
    return (min(all_x) - margin, max(all_x) + margin), (min(all_y) - margin, max(all_y) + margin)


def _mold_closest_fn(mold_points):
    """Verilen noktanın kalıp konturundaki en yakın noktasını ve mesafesini döndüren fonksiyon üretir."""
    mold = np.asarray(mold_points, float)
    A = mold[:-1]
    AB = mold[1:] - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-18)

    def closest(point):
        p = np.array(point, float)
        ap = p - A
        t = np.clip((ap * AB).sum(1) / L2, 0.0, 1.0)
        proj = A + t[:, None] * AB
        dists = np.sqrt(((proj - p) ** 2).sum(1))
        k = dists.argmin()
        return tuple(proj[k]), float(dists[k])
    return closest


# -------------------------------------------------------------
# 7b) Büyük karşılaştırma grafiği (G0 hariç)
# -------------------------------------------------------------
def plot_comparison_fig(channels_segments, start_points, radii, mold_points):
    fig = Figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.80)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title('Büyük Boyutlu Net Kontur Karşılaştırması (G0 Hareketi Hariç - Her Kod Satırı Farklı Renk)',
                 fontsize=14, pad=15)
    ax.set_xlabel('X Koordinatı (mm)', fontsize=11)
    ax.set_ylabel('Y Koordinatı (mm)', fontsize=11)

    if len(mold_points) > 1:
        ax.plot([p[0] for p in mold_points], [p[1] for p in mold_points],
                color='black', linewidth=1.2, linestyle='-', label='Kalıp Konturu (Sabit)')

    for i in range(N_CH):
        drawn = set()
        for seg in channels_segments[i]:
            if seg['type'] == 'move' and not _is_g0(seg) and seg['orig_points']:
                c_idx = seg.get('color_idx', 0)
                color = CHANNEL_COLORS[c_idx % 2]
                for pts in _contact_chunks(seg):
                    key = f"K{i+1}_S{c_idx % 2}"
                    lbl = ''
                    if key not in drawn:
                        lbl = f'Kanal {i+1} - Satır Grubu {c_idx % 2 + 1}'
                        drawn.add(key)
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=3.0, label=lbl)
        sx, sy = start_points[i]
        ax.plot(sx, sy, 'o', color=CHANNEL_COLORS[0], markersize=8,
                label=f'Kanal {i+1} Başlangıç ({sx:.1f}, {sy:.1f})')

    xl, yl = _plot_limits(mold_points, start_points, radii, 3.0)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=9, shadow=True)
    return fig


# -------------------------------------------------------------
# 7c) Kanal bazlı detaylı grafik + kalıp mesafe ölçüleri
# -------------------------------------------------------------
def _densify_px(P, step=4.0):
    """Ekran (piksel) koordinatındaki çoklu çizgiyi ~step piksel aralıklı noktalara çevirir."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    out = [P[0:1]]
    for a, b in zip(P[:-1], P[1:]):
        L = float(np.hypot(*(b - a)))
        n = max(int(L / step), 1)
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.vstack(out)


def _place_distance_labels(fig, ax, labels, polylines, fontsize=7):
    """Mesafe etiketlerini birbirine ve çizilen yollara/çizgilere değmeyecek şekilde yerleştirir.
    labels: [(nokta, metin, (dx, dy) tercih edilen yön)], polylines: kaçınılacak çizgiler (veri koordinatı)."""
    if not labels:
        return
    canvas = FigureCanvasAgg(fig)
    canvas.draw()                                   # eksen sınırları / oran kesinleşsin
    renderer = canvas.get_renderer()
    tr = ax.transData
    inv = tr.inverted()
    axbox = ax.get_window_extent(renderer)

    obs = [_densify_px(tr.transform(np.asarray(pl, float))) for pl in polylines if len(pl) >= 1]
    obs.append(tr.transform(np.asarray([l[0] for l in labels], float)))
    obs = np.vstack(obs) if obs else np.zeros((0, 2))

    placed = []                                      # yerleşmiş kutular: (x0, y0, x1, y1) piksel
    angles = np.deg2rad(np.arange(0, 181, 20))
    radii_px = [12, 22, 34, 48, 66, 88, 115]
    pad = 2.5
    for kp, text, (dx, dy) in labels:
        probe = ax.text(0, 0, text, fontsize=fontsize, fontweight='bold')
        bb = probe.get_window_extent(renderer)
        probe.remove()
        w, h = bb.width + 2 * pad, bb.height + 2 * pad
        k_d = tr.transform(np.asarray(kp, float))
        base = np.arctan2(dy, dx) if np.hypot(dx, dy) > 1e-9 else np.pi / 2
        devs = [0.0]
        for a in angles[1:]:
            devs += [a, -a]
        best = None
        for r in radii_px:
            for dv in devs:
                th = base + dv
                ux, uy = np.cos(th), np.sin(th)
                reach = r + abs(ux) * w / 2 + abs(uy) * h / 2
                cx, cy = k_d[0] + ux * reach, k_d[1] + uy * reach
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                score = 0.0
                if box[0] < axbox.x0 or box[2] > axbox.x1 or box[1] < axbox.y0 or box[3] > axbox.y1:
                    score += 500
                for pb in placed:
                    if not (box[2] < pb[0] or box[0] > pb[2] or box[3] < pb[1] or box[1] > pb[3]):
                        score += 1000
                if len(obs):
                    inside = ((obs[:, 0] >= box[0]) & (obs[:, 0] <= box[2]) &
                              (obs[:, 1] >= box[1]) & (obs[:, 1] <= box[3]))
                    score += float(inside.sum())
                score += r * 0.01 + abs(dv) * 0.05           # yakın ve tercih edilen yönü hafifçe tercih et
                if best is None or score < best[0]:
                    best = (score, box, (cx, cy), r)
            if best is not None and best[0] < 1.0:
                break
        _, box, (cx, cy), r = best
        placed.append(box)
        lx, ly = inv.transform((cx, cy))
        arrow = dict(arrowstyle='-', color='darkred', lw=0.5, alpha=0.8, shrinkA=0, shrinkB=1) if r > 20 else None
        ax.annotate(text, xy=kp, xytext=(lx, ly), textcoords='data', ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color='darkred', arrowprops=arrow,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.15'))


def plot_channel_fig(i, channels_segments, start_points, radii, mold_points):
    closest = _mold_closest_fn(mold_points)
    xl, yl = _plot_limits(mold_points, start_points, radii, 3.0)

    fig = Figure(figsize=(16, 11), dpi=100)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.84)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_title(f'Kanal {i+1} Detaylı Kontur ve Kalıp Mesafesi Analizi', fontsize=16, pad=14)
    ax.set_xlabel('X Koordinatı (mm)', fontsize=12)
    ax.set_ylabel('Y Koordinatı (mm)', fontsize=12)

    polylines = []
    if len(mold_points) > 1:
        ax.plot([p[0] for p in mold_points], [p[1] for p in mold_points],
                color='black', linewidth=2.0, linestyle='-', label='Kalıp Konturu')
        polylines.append(mold_points)

    drawn = set()
    labels, seen = [], set()
    for seg in channels_segments[i]:
        if seg['type'] == 'move' and not _is_g0(seg) and seg['orig_points']:
            c_idx = seg.get('color_idx', 0)
            color = CHANNEL_COLORS[c_idx % 2]
            for pts in _contact_chunks(seg):
                key = f"S{c_idx % 2}"
                lbl = ''
                if key not in drawn:
                    lbl = f'Satır Grubu {c_idx % 2 + 1} Yolu'
                    drawn.add(key)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=3.0, label=lbl)
                polylines.append(pts)

            pts = seg['orig_points']
            n = len(pts)
            for kp in (pts[0], pts[n // 2], pts[-1]):
                k2 = (round(kp[0], 3), round(kp[1], 3))
                if k2 in seen:                                   # aynı nokta (segment sonu = sonraki başı) tekrar yazılmasın
                    continue
                seen.add(k2)
                cp, d = closest(kp)
                ax.plot([kp[0], cp[0]], [kp[1], cp[1]], color='red', linestyle=':', linewidth=0.9, alpha=0.8)
                ax.plot(kp[0], kp[1], 'o', color='red', markersize=3.5)
                polylines.append([kp, cp])
                labels.append((kp, f"{d:.2f}mm", (kp[0] - cp[0], kp[1] - cp[1])))

    sx, sy = start_points[i]
    ax.plot(sx, sy, 'o', color=CHANNEL_COLORS[0], markersize=10, label='Başlangıç Noktası')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0, fontsize=10, shadow=True)
    _place_distance_labels(fig, ax, labels, polylines, fontsize=7)
    return fig


def fig_to_svg_card(fig):
    """Vektörel SVG + 'Tam Ekran' düğmesi içeren HTML kartı üretir (gr.HTML içinde gösterilir)."""
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    svg = buf.getvalue()
    svg = svg[svg.find('<svg'):]
    svg = svg.replace('<svg ', '<svg style="max-width:100%;height:auto;" ', 1)
    uid = 'svgw' + uuid.uuid4().hex[:8]
    css = (f"#{uid}:fullscreen{{padding:10px!important;background:#fff!important;width:100vw!important;"
           f"height:100vh!important;box-sizing:border-box!important;display:flex!important;"
           f"flex-direction:column!important;align-items:center!important;justify-content:center!important;}}"
           f"#{uid}:fullscreen svg{{width:95vw!important;height:85vh!important;"
           f"max-width:98vw!important;max-height:88vh!important;}}")
    onclick = (f"var e=document.getElementById('{uid}');"
               "if(!document.fullscreenElement){e.requestFullscreen().catch(function(err){"
               "alert('Tam ekran moduna geçilemedi: '+err.message);});}"
               "else{document.exitFullscreen();}")
    return (f'<div id="{uid}" style="width:100%;display:flex;flex-direction:column;align-items:center;'
            f'padding:10px;background:#fff;border-radius:8px;border:1px solid #ddd;box-sizing:border-box;">'
            f'<style>{css}</style>'
            f'<button onclick="{onclick}" style="margin-bottom:10px;padding:8px 16px;font-size:12px;'
            f'font-weight:bold;cursor:pointer;background:#0072B2;color:#fff;border:none;border-radius:4px;">'
            f'🔍 Tam Ekran Yap / Çık</button>'
            f'<div style="width:100%;display:flex;justify-content:center;align-items:center;background:#fff;">'
            f'{svg}</div></div>')


# -------------------------------------------------------------
# Mesafe tablosu (Colab 7c-analiz hücresi)
# -------------------------------------------------------------
def distance_table(channels_segments, mold_points):
    if len(mold_points) < 2:
        return pd.DataFrame(columns=['Kanal', 'Segment No', 'Başlangıç (mm)', 'Orta (mm)', 'Bitiş (mm)'])
    closest = _mold_closest_fn(mold_points)
    rows = []
    for i in range(N_CH):
        for s_idx, seg in enumerate(channels_segments[i]):
            if seg['type'] == 'move' and not _is_g0(seg) and seg['orig_points']:
                pts = seg['orig_points']
                n = len(pts)
                rows.append({
                    'Kanal': f"Kanal {i+1}",
                    'Segment No': f"Kod: {seg.get('color_idx', s_idx)}",
                    'Başlangıç (mm)': round(closest(pts[0])[1], 4),
                    'Orta (mm)': round(closest(pts[n // 2])[1], 4),
                    'Bitiş (mm)': round(closest(pts[-1])[1], 4),
                })
    return pd.DataFrame(rows, columns=['Kanal', 'Segment No', 'Başlangıç (mm)', 'Orta (mm)', 'Bitiş (mm)'])


# -------------------------------------------------------------
# AutoCAD çember / path kodları (Colab bölüm 9 ve 10)
# -------------------------------------------------------------
def autocad_circles_text(timelines, total_time, radii, start_t, end_t, step_t):
    if end_t <= start_t or step_t <= 0:
        raise ValueError("Geçersiz zaman aralığı veya adım büyüklüğü (Bitiş > Başlangıç ve Adım > 0 olmalı).")
    n = int(np.floor((end_t - start_t) / step_t + 1e-9)) + 1
    if n > 100000:
        raise ValueError(f"Çok fazla çember üretilecek ({n}). Zaman adımını büyüt.")
    per_channel = [[] for _ in range(N_CH)]
    for t in start_t + step_t * np.arange(n):
        t_c = max(0.0, min(float(t), total_time))
        for i in range(N_CH):
            c = get_position(timelines[i], t_c, 'points')
            per_channel[i].append(f"_CIRCLE {c[0]:.4f},{c[1]:.4f} {radii[i]:.4f}")
    lines = []
    for i in range(N_CH):
        lines.append(f"; Kanal {i+1} Çember Kodları")
        lines.extend(per_channel[i])
        if i < N_CH - 1:
            lines.append("")
    return "\n".join(lines)


def autocad_paths_text(channels_segments):
    lines = []
    for i in range(N_CH):
        cmds = []
        for seg in channels_segments[i]:
            if seg['type'] == 'move' and seg['orig_points']:
                for pts in _contact_chunks(seg):
                    cmds.append("_PLINE " + " ".join(f"{q[0]:.4f},{q[1]:.4f}" for q in pts))
        if cmds:
            lines.append(f"; Kanal {i+1} İşlenmiş Kontur Kodları")
            lines.extend(cmds)
            if i < N_CH - 1:
                lines.append("")
    return "\n".join(lines)


# ===== Hücre 16 =====
# -------------------------------------------------------------
# ANİMASYON SAHNESİ (canlı oynatıcı ve MP4 aynı kodu kullanır)
# -------------------------------------------------------------
def _scene_bounds(channels_segments, start_points, radii, mold_points):
    all_x, all_y = [p[0] for p in mold_points], [p[1] for p in mold_points]
    for segs, (sx, sy) in zip(channels_segments, start_points):
        all_x.append(sx); all_y.append(sy)
        for seg in segs:
            if seg['type'] == 'move':
                all_x.extend(p[0] for p in seg['orig_points'])
                all_y.extend(p[1] for p in seg['orig_points'])
    margin = max(10.0, max(radii) * 2)
    return min(all_x) - margin, max(all_x) + margin, min(all_y) - margin, max(all_y) + margin


def build_scene(channels_segments, start_points, radii, mold_points):
    xmin, xmax, ymin, ymax = _scene_bounds(channels_segments, start_points, radii, mold_points)
    fig = Figure(figsize=(8.5, 6.0), dpi=90)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.73)          # sağda açıklama kutusu için boşluk
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("CNC Simülasyonu")
    if len(mold_points) > 1:
        ax.plot([p[0] for p in mold_points], [p[1] for p in mold_points],
                color='black', linewidth=1.0, linestyle='-', label='Kalıp (sabit)')

    sc = {'fig': fig, 'canvas': canvas, 'contact_lines': [], 'tool_circles': [],
          'spoke_lines': [], 'contact_dots': [], 'tool_centers': []}
    for i in range(N_CH):
        (cline,) = ax.plot([], [], color=CHANNEL_COLORS[i], linewidth=2.2, label=f'Kanal {i+1} — kontur')
        circ = Circle((0, 0), radii[i], facecolor=CHANNEL_COLORS[i], alpha=0.22,
                      edgecolor=CHANNEL_COLORS[i], linewidth=1.8)
        ax.add_patch(circ)
        (spoke,) = ax.plot([], [], color=CHANNEL_COLORS[i], linewidth=1.2)
        (dot,) = ax.plot([], [], 'o', color=CHANNEL_COLORS[i], markersize=5)
        (center,) = ax.plot([], [], '+', color=CHANNEL_COLORS[i], markersize=7, markeredgewidth=1.5)
        sc['contact_lines'].append(cline); sc['tool_circles'].append(circ)
        sc['spoke_lines'].append(spoke); sc['contact_dots'].append(dot); sc['tool_centers'].append(center)
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1.0), borderaxespad=0, fontsize=8)
    sc['time_text'] = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=9)
    return sc


def render_scene(sc, timelines, total_time, t):
    for i in range(N_CH):
        center_pos = get_position(timelines[i], t, 'points')
        contact_pos = get_position(timelines[i], t, 'orig_points')
        trail = get_trail(timelines[i], t, 'orig_points')
        if trail:
            sc['contact_lines'][i].set_data([p[0] for p in trail], [p[1] for p in trail])
        else:
            sc['contact_lines'][i].set_data([], [])
        sc['tool_circles'][i].center = center_pos
        sc['tool_centers'][i].set_data([center_pos[0]], [center_pos[1]])
        sc['contact_dots'][i].set_data([contact_pos[0]], [contact_pos[1]])
        sc['spoke_lines'][i].set_data([center_pos[0], contact_pos[0]], [center_pos[1], contact_pos[1]])
    sc['time_text'].set_text(f't = {t:.2f} / {total_time:.2f} s')


def scene_rgb(sc):
    sc['canvas'].draw()
    return np.array(sc['canvas'].buffer_rgba())[:, :, :3]


# -------------------------------------------------------------
# Canlı oynatıcı için sunucu tarafı önbellek (oturum başına bir 'token')
# -------------------------------------------------------------
_ANIM = {}
_ANIM_LOCK = threading.RLock()


def register_anim(channels_segments, start_points, radii, mold_points, timelines, total_time, fps, speed):
    sc = build_scene(channels_segments, start_points, radii, mold_points)
    n_frames = max(int(total_time * fps / speed), 2)
    token = uuid.uuid4().hex
    with _ANIM_LOCK:
        _ANIM[token] = {'scene': sc, 'timelines': timelines, 'total_time': total_time, 'n_frames': n_frames}
        while len(_ANIM) > 8:                       # eski oturumları temizle
            _ANIM.pop(next(iter(_ANIM)))
    return token, n_frames


def render_frame_image(token, frame):
    with _ANIM_LOCK:
        a = _ANIM.get(token)
        if not a:
            return None
        frame = int(max(0, min(int(frame or 0), a['n_frames'])))
        t = (frame / a['n_frames']) * a['total_time']
        render_scene(a['scene'], a['timelines'], a['total_time'], t)
        return Image.fromarray(scene_rgb(a['scene']))


def playback_interval(fps, speed):
    return max(1.0 / max(fps * speed, 1e-6), 0.08)   # Gradio sunucu turu nedeniyle alt sınır


# -------------------------------------------------------------
# MP4 (Colab bölüm 8) - kareleri belleğe biriktirmeden akış halinde yazar
# -------------------------------------------------------------
def _even_crop(img):
    h, w = img.shape[:2]
    return img[:h - (h % 2), :w - (w % 2)]


def mp4_generator(channels_segments, start_points, radii, mold_points, timelines, total_time, fps, path):
    """Her ~5 karede bir (kare_no, toplam) üretir; bittiğinde dosya yazılmış olur."""
    sc = build_scene(channels_segments, start_points, radii, mold_points)
    n_frames = max(int(total_time * fps), 2)
    writer, frames = None, None
    try:
        try:
            import imageio.v2 as iio
        except ImportError:
            import imageio as iio
        writer = iio.get_writer(path, fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=1)
    except Exception:
        writer, frames = None, []            # imageio-ffmpeg yoksa moviepy'ye düş

    for k in range(n_frames):
        render_scene(sc, timelines, total_time, (k / n_frames) * total_time)
        img = _even_crop(scene_rgb(sc))
        if writer is not None:
            writer.append_data(img)
        else:
            frames.append(img.copy())
        if k % 5 == 0 or k == n_frames - 1:
            yield k + 1, n_frames
    if writer is not None:
        writer.close()
    else:
        try:
            from moviepy.editor import ImageSequenceClip
        except Exception:
            from moviepy import ImageSequenceClip
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(path, fps=fps, codec='libx264', logger=None,
                             ffmpeg_params=['-pix_fmt', 'yuv420p'])


# -------------------------------------------------------------
# JSON kaydet / yükle (Colab bölüm 11 ile uyumlu + y_offset ve AutoCAD ayarları eklendi)
# -------------------------------------------------------------
def config_to_json_dict(cfg, fps, speed, ac_start, ac_end, ac_step):
    data = {'channels': [], 'mold': dict(cfg['mold']),
            'settings': {'contact_mode': cfg['contact_mode'], 'fps': fps, 'speed': speed},
            'autocad': {'start': ac_start, 'end': ac_end, 'step': ac_step}}
    for idx, ch in enumerate(cfg['channels']):
        data['channels'].append({'channel_id': idx + 1, 'x0': ch['x0'], 'y0': ch['y0'], 'tr': ch['tr'],
                                 'fd': ch['fd'], 'y_offset': ch['y_offset'], 'code': ch['code'],
                                 'dia_mode': ch['dia_mode'], 'dia_axis': ch['dia_axis']})
    return data


def json_dict_to_updates(data, gr_update):
    """JSON sözlüğünü bileşen güncellemelerine çevirir (N_CFG cfg + fps + speed + 3 AutoCAD).
    JSON'da olmayan alanlar mevcut değerini korur (gr_update() boş güncelleme)."""
    def pick(d, key):
        return gr_update(value=d[key]) if key in d else gr_update()
    ups = []
    chans = data.get('channels', []) or []
    for k in range(N_CH):
        ch = chans[k] if k < len(chans) else {}
        for key in ('x0', 'y0', 'tr', 'fd', 'y_offset', 'code', 'dia_mode', 'dia_axis'):
            ups.append(pick(ch, key))
    mold = data.get('mold', {}) or {}
    for key in ('x0', 'y0', 'code', 'dia_mode', 'dia_axis'):
        ups.append(pick(mold, key))
    st = data.get('settings', {}) or {}
    ups.append(pick(st, 'contact_mode'))
    ups.append(pick(st, 'fps'))
    ups.append(pick(st, 'speed'))
    ac = data.get('autocad', {}) or {}
    for key in ('start', 'end', 'step'):
        ups.append(pick(ac, key))
    return ups


# -------------------------------------------------------------
# TARAYICIDA ÇALIŞAN HIZLI OYNATICI (JS canvas) - veriyi bir kez gönderir,
# oynatma / hız / kaydırma tamamen tarayıcıda olur, sunucuya istek atmaz
# -------------------------------------------------------------
PLAYER_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;padding:0;height:100%;font-family:Arial,Helvetica,sans-serif;background:#fff;color:#222}
body{display:flex;flex-direction:column}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:7px 10px;background:#f3f4f6;border-bottom:1px solid #ddd;font-size:13px}
button{padding:6px 12px;font-size:13px;font-weight:bold;cursor:pointer;border:1px solid #bbb;border-radius:5px;background:#fff}
button.main{background:#0072B2;color:#fff;border-color:#0072B2}
button.on{background:#222;color:#fff}
input[type=range]{vertical-align:middle}
#cwrap{flex:1 1 auto;min-height:0;position:relative}
#cv{position:absolute;left:0;top:0;width:100%;height:100%}
#tm{flex:1;min-width:140px}
</style></head><body>
<div class="bar">
 <button id="play" class="main">▶ Oynat</button><button id="rst">⟲ Başa</button>
 <span>Hız:</span><input id="spd" type="range" min="0.1" max="30" step="0.1" value="1" style="width:150px">
 <b id="spdv" style="min-width:46px">1.0×</b>
 <span id="presets"></span>
 <label><input type="checkbox" id="loop" checked> Döngü</label>
 <button id="fs">⛶ Tam Ekran</button>
</div>
<div class="bar"><input id="tm" type="range" min="0" max="10000" value="0"><b id="tv"></b></div>
<div id="cwrap"><canvas id="cv"></canvas></div>
<script>
(function(){
var D=__DATA__;
var cv=document.getElementById('cv'), ctx=cv.getContext('2d');
var playB=document.getElementById('play'), rstB=document.getElementById('rst'), spd=document.getElementById('spd'),
    spdv=document.getElementById('spdv'), loopC=document.getElementById('loop'), fsB=document.getElementById('fs'),
    tm=document.getElementById('tm'), tv=document.getElementById('tv'), pre=document.getElementById('presets');
var t=0, playing=false, last=0, speed=Math.min(Math.max(D.speed||1,0.1),30), W=0, H=0;
var xmin=D.b[0], xmax=D.b[1], ymin=D.b[2], ymax=D.b[3];
function hex2rgba(h,a){var n=parseInt(h.slice(1),16);return 'rgba('+(n>>16&255)+','+(n>>8&255)+','+(n&255)+','+a+')';}
function nice(r){var e=Math.pow(10,Math.floor(Math.log10(r))), f=r/e; return (f<1.5?1:f<3.5?2:f<7.5?5:10)*e;}
function lerp(a,k,f){return a[k]+(a[k+1]-a[k])*f;}
function setSpeed(v){speed=v; spd.value=v; spdv.textContent=v.toFixed(1)+'×';
  var bs=pre.getElementsByTagName('button'); for(var i=0;i<bs.length;i++){bs[i].className=(Math.abs(parseFloat(bs[i].getAttribute('data-v'))-v)<1e-9)?'on':'';}}
[0.5,1,2,5,10,20].forEach(function(v){var b=document.createElement('button'); b.textContent=v+'×'; b.setAttribute('data-v',v);
  b.onclick=function(){setSpeed(v);}; pre.appendChild(b);});
function resize(){var dpr=window.devicePixelRatio||1, r=cv.getBoundingClientRect(); W=r.width; H=r.height;
  cv.width=Math.max(10,Math.round(W*dpr)); cv.height=Math.max(10,Math.round(H*dpr)); ctx.setTransform(dpr,0,0,dpr,0,0); draw();}
function draw(){
  ctx.clearRect(0,0,W,H);
  var L=58,B=40,T=34,R=Math.min(175,W*0.26);
  var pw=W-L-R, ph=H-T-B; if(pw<20||ph<20) return;
  var sc=Math.min(pw/(xmax-xmin), ph/(ymax-ymin));
  var ox=L+(pw-sc*(xmax-xmin))/2, oy=T+(ph-sc*(ymax-ymin))/2;
  function X(x){return ox+(x-xmin)*sc;} function Y(y){return oy+(ymax-y)*sc;}
  ctx.font='bold 15px Arial'; ctx.fillStyle='#222'; ctx.textAlign='center'; ctx.fillText('CNC Simülasyonu',ox+sc*(xmax-xmin)/2,20);
  // ızgara
  var st=nice(Math.max(xmax-xmin,ymax-ymin)/8);
  ctx.save(); ctx.beginPath(); ctx.rect(ox,oy,sc*(xmax-xmin),sc*(ymax-ymin)); ctx.clip();
  ctx.strokeStyle='rgba(0,0,0,0.12)'; ctx.lineWidth=1; ctx.beginPath();
  for(var gx=Math.ceil(xmin/st)*st; gx<=xmax; gx+=st){ctx.moveTo(X(gx),oy); ctx.lineTo(X(gx),oy+sc*(ymax-ymin));}
  for(var gy=Math.ceil(ymin/st)*st; gy<=ymax; gy+=st){ctx.moveTo(ox,Y(gy)); ctx.lineTo(ox+sc*(xmax-xmin),Y(gy));}
  ctx.stroke();
  // kalıp
  ctx.strokeStyle='#000'; ctx.lineWidth=1.4; ctx.beginPath();
  D.mold.forEach(function(p,i){ if(i===0) ctx.moveTo(X(p[0]),Y(p[1])); else ctx.lineTo(X(p[0]),Y(p[1])); }); ctx.stroke();
  // kanallar
  var u=t/D.dt, k=Math.min(Math.floor(u),D.n-1), f=u-k; if(k<0){k=0;f=0;}
  D.ch.forEach(function(c,i){
    var col=D.col[i];
    ctx.strokeStyle=col; ctx.lineWidth=2.3; ctx.beginPath();
    var pen=false, m=c.m[k], lx=0, ly=0;
    for(var j=0;j<m;j++){var p=c.F[j]; if(p===null){pen=false; continue;}
      if(!pen){ctx.moveTo(X(p[0]),Y(p[1])); pen=true;} else ctx.lineTo(X(p[0]),Y(p[1]));}
    var hx=c.hx[k], hy=c.hy[k];
    if(hx!==null){ if(c.hx[k+1]!==null && c.hx[k+1]!==undefined){hx=lerp(c.hx,k,f); hy=lerp(c.hy,k,f);}
      if(pen) ctx.lineTo(X(hx),Y(hy)); }
    ctx.stroke();
    var cx=lerp(c.cx,k,f), cy=lerp(c.cy,k,f), px=lerp(c.px,k,f), py=lerp(c.py,k,f);
    ctx.beginPath(); ctx.arc(X(cx),Y(cy),D.r[i]*sc,0,6.2832); ctx.fillStyle=hex2rgba(col,0.22); ctx.fill();
    ctx.strokeStyle=col; ctx.lineWidth=1.8; ctx.stroke();
    ctx.lineWidth=1.2; ctx.beginPath(); ctx.moveTo(X(cx),Y(cy)); ctx.lineTo(X(px),Y(py)); ctx.stroke();
    ctx.lineWidth=1.6; ctx.beginPath(); ctx.moveTo(X(cx)-6,Y(cy)); ctx.lineTo(X(cx)+6,Y(cy)); ctx.moveTo(X(cx),Y(cy)-6); ctx.lineTo(X(cx),Y(cy)+6); ctx.stroke();
    ctx.fillStyle=col; ctx.beginPath(); ctx.arc(X(px),Y(py),4,0,6.2832); ctx.fill();
  });
  ctx.restore();
  ctx.strokeStyle='#444'; ctx.lineWidth=1; ctx.strokeRect(ox,oy,sc*(xmax-xmin),sc*(ymax-ymin));
  // eksen yazıları
  ctx.fillStyle='#333'; ctx.font='11px Arial'; ctx.textAlign='center';
  for(var tx=Math.ceil(xmin/st)*st; tx<=xmax; tx+=st){ctx.fillText(+tx.toFixed(6),X(tx),oy+sc*(ymax-ymin)+14);}
  ctx.textAlign='right';
  for(var ty=Math.ceil(ymin/st)*st; ty<=ymax; ty+=st){ctx.fillText(+ty.toFixed(6),ox-5,Y(ty)+4);}
  ctx.textAlign='left'; ctx.font='bold 12px Arial'; ctx.fillStyle='#111';
  ctx.fillText('t = '+t.toFixed(2)+' / '+D.T.toFixed(2)+' s',ox+6,oy+16);
  // açıklama
  var lgx=W-R+14, lgy=T+4; ctx.font='11px Arial';
  function leg(col,txt,y){ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.beginPath(); ctx.moveTo(lgx,y); ctx.lineTo(lgx+22,y); ctx.stroke(); ctx.fillStyle='#222'; ctx.fillText(txt,lgx+28,y+4);}
  leg('#000','Kalıp (sabit)',lgy+8);
  for(var q=0;q<D.col.length;q++) leg(D.col[q],'Kanal '+(q+1)+' — kontur',lgy+28+q*20);
}
function ui(){tm.value=Math.round(t/D.T*10000); tv.textContent=t.toFixed(2)+' / '+D.T.toFixed(2)+' s';}
function setPlaying(p){playing=p; playB.textContent=p?'⏸ Duraklat':'▶ Oynat'; if(p){ if(t>=D.T) t=0; last=performance.now(); requestAnimationFrame(frame);} }
function frame(now){ if(!playing) return; var dt=(now-last)/1000; last=now; t+=dt*speed;
  if(t>=D.T){ if(loopC.checked){t=t%D.T;} else {t=D.T; setPlaying(false);} }
  draw(); ui(); if(playing) requestAnimationFrame(frame);}
playB.onclick=function(){setPlaying(!playing);};
rstB.onclick=function(){t=0; draw(); ui();};
spd.oninput=function(){setSpeed(parseFloat(spd.value));};
tm.oninput=function(){t=tm.value/10000*D.T; draw(); ui();};
fsB.onclick=function(){ if(!document.fullscreenElement){ document.documentElement.requestFullscreen().catch(function(e){alert('Tam ekran açılamadı: '+e.message);}); } else { document.exitFullscreen(); } };
document.addEventListener('fullscreenchange',function(){setTimeout(resize,100);});
if(window.ResizeObserver){new ResizeObserver(resize).observe(document.getElementById('cwrap'));} else {window.addEventListener('resize',resize);}
setSpeed(speed); resize(); ui();
})();
</script></body></html>"""


def _same_pt(a, b):
    if np.isnan(a[0]) and np.isnan(b[0]):
        return True
    return abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9


def build_player_data(channels_segments, start_points, radii, mold_points, timelines, total_time,
                      speed0=1.0, rate=30):
    xmin, xmax, ymin, ymax = _scene_bounds(channels_segments, start_points, radii, mold_points)
    n = int(min(max(np.ceil(total_time * rate), 2), 6000))
    dt = total_time / n
    r3 = lambda v: round(float(v), 3)
    chans = []
    for i in range(N_CH):
        tl = timelines[i]
        full = get_trail(tl, total_time + 1.0, 'orig_points')
        F = [None if np.isnan(p[0]) else [r3(p[0]), r3(p[1])] for p in full]
        c = {'F': F, 'cx': [], 'cy': [], 'px': [], 'py': [], 'm': [], 'hx': [], 'hy': []}
        for k in range(n + 1):
            t = k * dt
            cp = get_position(tl, t, 'points')
            pp = get_position(tl, t, 'orig_points')
            tr = get_trail(tl, t, 'orig_points')
            m, head = len(tr), None
            if m > 0 and (m > len(full) or not _same_pt(tr[-1], full[m - 1])):
                head = tr[-1]
                m -= 1
            c['cx'].append(r3(cp[0])); c['cy'].append(r3(cp[1]))
            c['px'].append(r3(pp[0])); c['py'].append(r3(pp[1]))
            c['m'].append(m)
            c['hx'].append(None if head is None or np.isnan(head[0]) else r3(head[0]))
            c['hy'].append(None if head is None or np.isnan(head[0]) else r3(head[1]))
        chans.append(c)
    return {'T': r3(total_time), 'n': n, 'dt': dt, 'b': [r3(xmin), r3(xmax), r3(ymin), r3(ymax)],
            'mold': [[r3(p[0]), r3(p[1])] for p in mold_points], 'r': [r3(r) for r in radii],
            'col': CHANNEL_COLORS[:N_CH], 'speed': float(speed0), 'ch': chans}


def build_player_html(data, height=720):
    doc = PLAYER_TEMPLATE.replace('__DATA__', json.dumps(data, separators=(',', ':')))
    return (f'<iframe srcdoc="{html.escape(doc, quote=True)}" sandbox="allow-scripts" allow="fullscreen" '
            f'style="width:100%;height:{height}px;border:1px solid #ddd;border-radius:8px;background:#fff"></iframe>')


# ===== Hücre 18 =====
import gradio as gr

N_ANIM_OUT = 13   # run_all çıktı sayısı


def _split_vals(vals):
    """vals = 22 cfg + fps + speed + ac_start + ac_end + ac_step"""
    cfg = cfg_from_values(vals[:N_CFG])
    fps = int(_num(vals[N_CFG], 20))
    speed = _num(vals[N_CFG + 1], 1.0) or 1.0
    ac_s, ac_e, ac_st = (_num(vals[N_CFG + 2], 0.0), _num(vals[N_CFG + 3], 0.0), _num(vals[N_CFG + 4], 0.1))
    return cfg, fps, speed, ac_s, ac_e, ac_st


# -------------------------------------------------------------
# 1) Ana çalıştırma: animasyon + analizler + AutoCAD kodları
# -------------------------------------------------------------
def run_all(*vals):
    try:
        cfg, fps, speed, ac_s, ac_e, ac_st = _split_vals(vals)
        segs, starts, radii, mold_pts, note = build_channels_cfg(cfg)
        timelines, total_time = synchronize(segs)
        if total_time <= 0:
            total_time = 1.0

        token, n_frames = register_anim(segs, starts, radii, mold_pts, timelines, total_time, fps, speed)
        frame0 = render_frame_image(token, 0)

        comp_html = fig_to_svg_card(plot_comparison_fig(segs, starts, radii, mold_pts))
        if len(mold_pts) > 1:
            ind_html = [fig_to_svg_card(plot_channel_fig(i, segs, starts, radii, mold_pts)) for i in range(N_CH)]
        else:
            msg = "<p>Kalıp noktası yetersiz.</p>"
            ind_html = [msg] * N_CH
        table = distance_table(segs, mold_pts)

        end_used = ac_e if ac_e > ac_s else total_time
        try:
            circ_txt = autocad_circles_text(timelines, total_time, radii, ac_s, end_used, ac_st)
        except ValueError as ve:
            circ_txt = f"Hata: {ve}"
        path_txt = autocad_paths_text(segs)

        try:
            player_html = build_player_html(build_player_data(segs, starts, radii, mold_pts, timelines,
                                                              total_time, speed0=speed))
        except Exception:
            player_html = "<p>Hızlı oynatıcı hazırlanamadı:</p><pre>" + html.escape(traceback.format_exc()) + "</pre>"

        status = (note + "\n" if note else "") + \
                 (f"Simülasyon hazır. Toplam süre: {total_time:.2f} sn — animasyon kare sayısı: {n_frames} — "
                  f"AutoCAD çember aralığı: {ac_s:g}–{end_used:.2f} sn")
        return (frame0,
                gr.update(minimum=0, maximum=n_frames, value=0),
                gr.update(value=playback_interval(fps, speed), active=False),
                token,
                comp_html, ind_html[0], ind_html[1], ind_html[2],
                table, circ_txt, path_txt, player_html, status)
    except Exception:
        err = "Simülasyon hazırlanırken hata oluştu:\n" + traceback.format_exc()
        return tuple([gr.update()] * (N_ANIM_OUT - 1) + [err])


# -------------------------------------------------------------
# 2) Oynatıcı kontrolleri (Oynat / Duraklat / Kare kaydırıcısı)
# -------------------------------------------------------------
def anim_render(frame, token):
    try:
        return render_frame_image(token, frame)
    except Exception:
        print("Kare güncellenirken hata:", traceback.format_exc())
        return gr.update()


def anim_play(cur, token):
    a = _ANIM.get(token)
    if not a:
        return gr.update(), gr.update(active=False)
    cur = int(cur or 0)
    if cur >= a['n_frames']:
        cur = 0
    return gr.update(value=cur), gr.update(active=True)


def anim_tick(cur, token):
    a = _ANIM.get(token)
    if not a:
        return gr.update(), gr.update(active=False)
    nxt = int(cur or 0) + 1
    if nxt >= a['n_frames']:
        return gr.update(value=a['n_frames']), gr.update(active=False)
    return gr.update(value=nxt), gr.update()


def anim_pause():
    return gr.update(active=False)


def anim_retune(fps, speed, token):
    """FPS / Hız çarpanı değişince kare sayısını ve oynatma aralığını yeniden ayarlar."""
    a = _ANIM.get(token)
    if not a:
        return gr.update(), gr.update()
    fps, speed = int(_num(fps, 20)), (_num(speed, 1.0) or 1.0)
    n_frames = max(int(a['total_time'] * fps / speed), 2)
    a['n_frames'] = n_frames
    return gr.update(minimum=0, maximum=n_frames, value=0), gr.update(value=playback_interval(fps, speed), active=False)


# -------------------------------------------------------------
# 3) MP4 kaydet (üreteç: ilerlemeyi durum kutusunda gösterir)
# -------------------------------------------------------------
def make_mp4(*vals):
    try:
        mp4_fps = int(_num(vals[N_CFG], 20)) or 20
        cfg = cfg_from_values(vals[:N_CFG])
        segs, starts, radii, mold_pts, note = build_channels_cfg(cfg)
        timelines, total_time = synchronize(segs)
        if total_time <= 0:
            total_time = 1.0
        out_dir = tempfile.mkdtemp(prefix='cnc_mp4_')
        path = os.path.join(out_dir, 'cnc_simulation.mp4')
        yield gr.update(), gr.update(), (note + "\n" if note else "") + "MP4 hazırlanıyor..."
        for done, total in mp4_generator(segs, starts, radii, mold_pts, timelines, total_time, mp4_fps, path):
            yield gr.update(), gr.update(), f"MP4 kareleri oluşturuluyor: {done}/{total}"
        yield path, path, f"Tamamlandı: cnc_simulation.mp4 ({total_time:.2f} sn, {mp4_fps} FPS). Aşağıdan indirebilirsin."
    except Exception:
        yield gr.update(), gr.update(), "MP4 oluşturulurken hata oluştu:\n" + traceback.format_exc()


# -------------------------------------------------------------
# 4) AutoCAD kodlarını tek başına yeniden üret
# -------------------------------------------------------------
def gen_circles_only(*vals):
    try:
        cfg, fps, speed, ac_s, ac_e, ac_st = _split_vals(vals)
        segs, starts, radii, mold_pts, note = build_channels_cfg(cfg)
        timelines, total_time = synchronize(segs)
        if total_time <= 0:
            total_time = 1.0
        end_used = ac_e if ac_e > ac_s else total_time
        txt = autocad_circles_text(timelines, total_time, radii, ac_s, end_used, ac_st)
        return txt, (note + "\n" if note else "") + f"AutoCAD çember kodları oluşturuldu ({ac_s:g}–{end_used:.2f} sn, adım {ac_st:g} sn)."
    except ValueError as ve:
        return gr.update(), f"Hata: {ve}"
    except Exception:
        return gr.update(), "AutoCAD kodları oluşturulurken hata oluştu:\n" + traceback.format_exc()


def gen_paths_only(*vals):
    try:
        cfg = cfg_from_values(vals[:N_CFG])
        segs, starts, radii, mold_pts, note = build_channels_cfg(cfg)
        return autocad_paths_text(segs), (note + "\n" if note else "") + "AutoCAD çizim yolları oluşturuldu."
    except Exception:
        return gr.update(), "AutoCAD çizim yolları oluşturulurken hata oluştu:\n" + traceback.format_exc()


# -------------------------------------------------------------
# 5) JSON kaydet / yükle + Python varsayılan kodu
# -------------------------------------------------------------
def export_json(*vals):
    try:
        cfg, fps, speed, ac_s, ac_e, ac_st = _split_vals(vals)
        data = config_to_json_dict(cfg, fps, speed, ac_s, ac_e, ac_st)
        out_dir = tempfile.mkdtemp(prefix='cnc_cfg_')
        path = os.path.join(out_dir, 'gcode_simulator_config.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return path, "Konfigürasyon 'gcode_simulator_config.json' olarak hazırlandı — yukarıdaki dosya kutusundan indirebilirsin."
    except Exception:
        return None, "JSON dışa aktarılırken hata oluştu:\n" + traceback.format_exc()


def import_json(file_obj):
    n_out = N_CFG + 2 + 3 + 1
    if file_obj is None:
        return tuple([gr.update()] * (n_out - 1) + ["JSON dosyası seçilmedi."])
    try:
        path = file_obj if isinstance(file_obj, str) else getattr(file_obj, 'name', None) or file_obj['name']
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ups = json_dict_to_updates(data, gr.update)
        return tuple(ups) + ("JSON yüklendi ve arayüz güncellendi. Simülasyonu güncellemek için 'Çalıştır' butonuna bas.",)
    except Exception as e:
        return tuple([gr.update()] * (n_out - 1) + [f"JSON yüklenirken hata oluştu. Dosya formatını kontrol et: {e}"])


def make_defaults_code(*vals):
    cfg = cfg_from_values(vals[:N_CFG])
    chans = pprint.pformat(cfg['channels'], sort_dicts=False, width=110)
    mold = pprint.pformat(cfg['mold'], sort_dicts=False, width=110)
    return (f"DEFAULT_CHANNELS = {chans}\n\nDEFAULT_MOLD = {mold}\n\n"
            f"DEFAULT_CONTACT_MODE = {cfg['contact_mode']!r}")


def _textbox(**kw):
    try:
        return gr.Textbox(show_copy_button=True, **kw)
    except TypeError:
        return gr.Textbox(**kw)


# -------------------------------------------------------------
# ARAYÜZ
# -------------------------------------------------------------
try:
    demo.close()          # hücre yeniden çalıştırılırsa eski sunucuyu kapat
except Exception:
    pass

with gr.Blocks(title="3 Kanallı CNC Simülatörü") as demo:
    gr.Markdown("# 3 Kanallı CNC Simülatörü\n"
                "Desteklenen komutlar: **G0, G1, G2/G3** (I/J veya R), **G4**, **G40/G41/G42**, **WAIT**. "
                "Girdileri soldan ayarla, **Çalıştır** butonuna bas; sağdaki sekmelerden animasyonu, grafikleri, "
                "mesafe tablosunu, AutoCAD kodlarını ve MP4'ü kullan.")

    with gr.Row():
        # ---------------- SOL: girdiler ----------------
        with gr.Column(scale=4, min_width=380):
            cfg_comps = []
            dia_pairs = []
            with gr.Tabs():
                for k in range(N_CH):
                    d = DEFAULT_CHANNELS[k]
                    with gr.Tab(f"Kanal {k+1}"):
                        x0 = gr.Number(value=d['x0'], label="Başlangıç X")
                        y0 = gr.Number(value=d['y0'], label="Başlangıç Y")
                        tr = gr.Number(value=d['tr'], label="Takım Yarıçapı")
                        fd = gr.Number(value=d['fd'], label="Varsayılan F")
                        yo = gr.Number(value=d['y_offset'], label="Y Offset (mm)")
                        code = gr.Textbox(value=d['code'], lines=8, label="G-Kodu")
                        dmode = gr.Radio(DIA_MODE_CHOICES, value=d['dia_mode'], label="Girdiğim değer türü",
                                         info="Çap seçilirse seçilen eksendeki G-kodu koordinatları ve başlangıç değeri 2'ye bölünür "
                                              "(I/J/R ve Y offset değişmez).")
                        daxis = gr.Radio(DIA_AXIS_CHOICES, value=d['dia_axis'], label="Çap hangi eksende?",
                                         visible=(d['dia_mode'] == 'diameter'))
                        dia_pairs.append((dmode, daxis))
                        cfg_comps += [x0, y0, tr, fd, yo, code, dmode, daxis]

            with gr.Accordion("Kalıp (Sabit Path)", open=True):
                m_x0 = gr.Number(value=DEFAULT_MOLD['x0'], label="Kalıp Başlangıç X")
                m_y0 = gr.Number(value=DEFAULT_MOLD['y0'], label="Kalıp Başlangıç Y")
                m_code = gr.Textbox(value=DEFAULT_MOLD['code'], lines=5, label="Kalıp G-Kodu")
                m_dmode = gr.Radio(DIA_MODE_CHOICES, value=DEFAULT_MOLD['dia_mode'], label="Kalıp: girdiğim değer türü",
                                   info="Çap seçilirse kalıp G-kodu koordinatları ve başlangıç değeri 2'ye bölünür.")
                m_daxis = gr.Radio(DIA_AXIS_CHOICES, value=DEFAULT_MOLD['dia_axis'], label="Kalıp: çap hangi eksende?",
                                   visible=(DEFAULT_MOLD['dia_mode'] == 'diameter'))
                dia_pairs.append((m_dmode, m_daxis))
                contact_mode = gr.Dropdown(choices=CONTACT_MODE_CHOICES, value=DEFAULT_CONTACT_MODE,
                                           label="Çizilen yol")
                cfg_comps += [m_x0, m_y0, m_code, m_dmode, m_daxis, contact_mode]

            with gr.Row():
                fps_sl = gr.Slider(5, 40, value=20, step=1, label="FPS")
                speed_sl = gr.Slider(0.2, 5.0, value=1.0, step=0.2, label="Hız Çarpanı")

            run_btn = gr.Button("▶ Simülasyonu ve Analizleri Çalıştır", variant="primary")

            with gr.Accordion("Konfigürasyon Yönetimi (JSON)", open=True):
                gr.Markdown("3 kanalın ve kalıbın tüm ayarlarını (başlangıç, yarıçap, F, Y offset, G-kodları, "
                            "çizilen yol modu, FPS/hız, AutoCAD zaman ayarları) tek JSON dosyası olarak kaydet / yükle.")
                json_btn = gr.Button("💾 Ayarları JSON Olarak Kaydet", variant="secondary")
                json_file_out = gr.File(label="İndirilecek JSON", interactive=False)
                json_upload = gr.File(label="JSON Ayar Dosyası Yükle", file_types=[".json"], file_count="single")

            with gr.Accordion("Varsayılanları Python kodu olarak al", open=False):
                gr.Markdown("Güncel arayüz değerlerini, ilgili hücredeki `DEFAULT_CHANNELS` / `DEFAULT_MOLD` "
                            "yerine yapıştırabileceğin kod olarak verir.")
                defaults_btn = gr.Button("Kodu Üret")
                defaults_out = _textbox(label="Python kodu", lines=10)

        # ---------------- SAĞ: çıktılar ----------------
        with gr.Column(scale=7):
            status = gr.Textbox(label="Durum / Uyarılar", lines=4, interactive=False)
            anim_token = gr.State("")

            with gr.Tabs():
                with gr.Tab("🎬 Animasyon"):
                    gr.Markdown("**Çalıştır**'dan sonra ▶ ile oynat, ⏸ ile durdur; kare kaydırıcısı ile istediğin "
                                "ana atla (ileri/geri). Oynatma her kareyi sunucudan çizdiği için yüksek hızda "
                                "MP4 sekmesindeki video daha akıcıdır. FPS / Hız çarpanını değiştirince kare sayısı "
                                "yeniden ayarlanır.")
                    anim_img = gr.Image(label="Simülasyon", type="pil", interactive=False)
                    with gr.Row():
                        play_btn = gr.Button("▶ Oynat", variant="primary")
                        pause_btn = gr.Button("⏸ Duraklat")
                    frame_sl = gr.Slider(0, 1, value=0, step=1, label="Kare", interactive=True)
                    play_timer = gr.Timer(0.1, active=False)

                with gr.Tab("⚡ Hızlı Oynatıcı"):
                    gr.Markdown("**Çalıştır**'dan sonra animasyon tarayıcıda akıcı olarak oynar. Hızı (0.1×–30×) "
                                "oynatıcının içindeki kaydırıcı / hazır düğmelerle istediğin an değiştirebilir, "
                                "zaman çubuğundan ileri-geri gidebilir, tam ekrana geçebilirsin. "
                                "1× = gerçek zamanlı simülasyon süresi.")
                    player_out = gr.HTML()

                with gr.Tab("🎥 MP4"):
                    gr.Markdown("Simülasyonu MP4 video olarak üretir ve indirmeni sağlar.")
                    mp4_fps = gr.Slider(5, 40, value=20, step=1, label="MP4 FPS")
                    mp4_btn = gr.Button("🎥 Simülasyonu MP4 Olarak Kaydet", variant="primary")
                    video_out = gr.Video(label="Simülasyon (MP4)", interactive=False)
                    mp4_file = gr.File(label="MP4 indir", interactive=False)

                with gr.Tab("🔍 Genel Karşılaştırma"):
                    comp_html = gr.HTML()

                with gr.Tab("📐 Bireysel Kanal Analizi"):
                    with gr.Tabs():
                        with gr.Tab("Kanal 1"):
                            ind1 = gr.HTML()
                        with gr.Tab("Kanal 2"):
                            ind2 = gr.HTML()
                        with gr.Tab("Kanal 3"):
                            ind3 = gr.HTML()

                with gr.Tab("📏 Mesafe Tablosu"):
                    gr.Markdown("Her hareket segmentinin başlangıç / orta / bitiş noktasının kalıp konturuna olan "
                                "en kısa mesafesi (G0 hariç).")
                    table_out = gr.DataFrame()

                with gr.Tab("💻 AutoCAD Çember"):
                    gr.Markdown("Seçilen zaman aralığında her kanalın takım merkezine `_CIRCLE` komutu üretir. "
                                "Bitiş zamanı 0 (veya Başlangıç'tan küçük) ise toplam süre kullanılır.")
                    with gr.Row():
                        ac_start = gr.Number(value=0.0, label="Başlangıç Zamanı (s)")
                        ac_end = gr.Number(value=0.0, label="Bitiş Zamanı (s)  [0 = otomatik]")
                        ac_step = gr.Number(value=0.1, label="Zaman Adımı (s)")
                    ac_circ_btn = gr.Button("AutoCAD Kodlarını Oluştur", variant="primary")
                    ac_circ_txt = _textbox(label="Çember Kodları (Kanal Bazlı)", lines=14)

                with gr.Tab("🔀 AutoCAD Path"):
                    gr.Markdown("Her kanalın işlenmiş kontur (temas noktası) izini `_PLINE` komutlarıyla üretir.")
                    ac_path_btn = gr.Button("AutoCAD Path Kodlarını Oluştur", variant="primary")
                    ac_path_txt = _textbox(label="Çizgi Yolu Kodları (PLINE)", lines=14)

    # ---------------- OLAYLAR ----------------
    all_inputs = cfg_comps + [fps_sl, speed_sl, ac_start, ac_end, ac_step]

    run_btn.click(
        run_all, inputs=all_inputs,
        outputs=[anim_img, frame_sl, play_timer, anim_token, comp_html, ind1, ind2, ind3,
                 table_out, ac_circ_txt, ac_path_txt, player_out, status])

    # oynatıcı
    frame_sl.change(anim_render, inputs=[frame_sl, anim_token], outputs=anim_img,
                    trigger_mode="always_last", show_progress="hidden")
    play_btn.click(anim_play, inputs=[frame_sl, anim_token], outputs=[frame_sl, play_timer])
    pause_btn.click(anim_pause, outputs=play_timer)
    play_timer.tick(anim_tick, inputs=[frame_sl, anim_token], outputs=[frame_sl, play_timer],
                    show_progress="hidden")
    fps_sl.release(anim_retune, inputs=[fps_sl, speed_sl, anim_token], outputs=[frame_sl, play_timer])
    speed_sl.release(anim_retune, inputs=[fps_sl, speed_sl, anim_token], outputs=[frame_sl, play_timer])

    # MP4
    mp4_btn.click(make_mp4, inputs=cfg_comps + [mp4_fps], outputs=[video_out, mp4_file, status])

    # AutoCAD
    ac_circ_btn.click(gen_circles_only, inputs=all_inputs, outputs=[ac_circ_txt, status])
    ac_path_btn.click(gen_paths_only, inputs=cfg_comps, outputs=[ac_path_txt, status])

    # JSON
    json_btn.click(export_json, inputs=all_inputs, outputs=[json_file_out, status])
    json_upload.upload(import_json, inputs=json_upload, outputs=all_inputs + [status])

    # Çap/yarıçap: eksen seçimi yalnızca "Çap" seçilince görünür
    for _dm, _da in dia_pairs:
        _dm.change(lambda m: gr.update(visible=(m == 'diameter')), inputs=_dm, outputs=_da)

    # Python varsayılan kodu
    defaults_btn.click(make_defaults_code, inputs=cfg_comps, outputs=defaults_out)

demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
