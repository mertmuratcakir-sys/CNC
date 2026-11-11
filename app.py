import os

# Şifre koruması: Render > Environment bölümünden APP_USER ve APP_PASSWORD tanımla
APP_USER = os.environ.get("APP_USER")
APP_PASSWORD = os.environ.get("APP_PASSWORD")
if not APP_USER or not APP_PASSWORD:
    raise RuntimeError("APP_USER ve APP_PASSWORD ortam değişkenleri tanımlı değil")

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
    for _ln, raw_line in enumerate(text.strip().split('\n')):
        line = raw_line.split(';')[0].strip()  # yorumları at
        if not line:
            continue
        if line.upper().startswith('WAIT'):
            commands.append({'type': 'WAIT', '_line': _ln})
            continue
        tokens = re.findall(r'([A-Za-z])\s*(-?\d+\.?\d*)', line)
        if not tokens:
            continue
        cmd = {'type': 'CMD', '_line': _ln}
        for letter, value in tokens:
            if letter.upper() == 'G':      # bir satırda birden fazla G kodu olabilir
                cmd.setdefault('Gs', []).append(float(value))
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


def simulate_channel(commands, start_x, start_y, tool_radius, default_feed=1000.0, y_offset=0.0, g0_feed=15000.0):
    """
    Bir kanalın G-kodu komutlarını işleyip hareket segmentleri listesi üretir.
    y_offset: G-kodundaki tüm Y değerlerine eklenecek ofset miktarı.
    """
    x, y = start_x, start_y
    motion_mode = 'G0'
    comp_mode = 'G40'
    feed = default_feed
    segments = []

    def _app(d):
        """Segmenti ekler; CNC ekranı için kaynak satırı ve programlanan (kompanzasyonsuz) geometriyi saklar."""
        d = dict(d)
        d['line'] = cmd.get('_line')
        if d.get('type') == 'move':
            d['prog_start'] = (float(x), float(y))
            d['prog_end'] = (float(target_x), float(target_y))
            if d.get('kind') == 'arc':
                d['prog_arc'] = (float(d['center'][0]), float(d['center'][1]), float(d['orig_radius']),
                                 float(d['start_ang']), float(d['end_ang']))
        elif d.get('type') == 'dwell':
            d['prog_start'] = d['prog_end'] = (float(x), float(y))
        segments.append(d)

    # Renklendirme için hareket içeren komutları sayacağız
    move_counter = 0

    for cmd in commands:
        if cmd['type'] == 'WAIT':
            _app({'type': 'wait'})
            continue

        for _gv in cmd.get('Gs', [cmd['G']] if 'G' in cmd else []):
            g = int(_gv)
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
            _app({'type': 'dwell', 'point': (x, y), 'duration': dwell})
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
            if motion_mode == 'G1':
                eff_feed = feed
            else:   # G0: satırda F yoksa varsayılan hızlı hareket hızı (g0_feed, varsayılan 15000)
                eff_feed = feed if 'F' in cmd else g0_feed
            dur = (dist / eff_feed) * 60 if eff_feed > 0 else 0.0

            _app({
                'type': 'move', 'kind': 'line',
                'points': [p0c, p1c], 'duration': dur, 'feed': eff_feed, 'g0': motion_mode == 'G0',
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
            _app(seg)

        x, y = target_x, target_y

    # Başlangıç noktası: program G41/G42 ile başlıyorsa (ilk hareketten önce veya ilk hareket satırında)
    # takım merkezi kompanze başlangıçta durur; yoksa girilen nokta takım merkezidir (G40 / G0 konumlama).
    _first = next((s_ for s_ in segments if s_['type'] == 'move'), None)
    if _first is not None and _first['comp_mode'] in ('G41', 'G42'):
        for s_ in segments:
            if s_ is _first:
                break
            if s_['type'] == 'dwell':
                s_['point'] = tuple(_first['points'][0])
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
                        'color_idx': segA.get('color_idx', 0), # Bir önceki hareketin rengini sürdür
                        # CNC ekranı: köşe geçişi bir önceki bloğa ait, programlanan konum köşede sabit
                        'line': segA.get('line'), 'g0': segA.get('g0', False),
                        'prog_start': segA.get('prog_end', V), 'prog_end': segA.get('prog_end', V),
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
    if getattr(tool_radius, 'shape', None) is not None:     # DXF makara profili
        return tool_contact_pts(C, np.asarray(mold, float), tool_radius)
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
CHANNEL_ENV_COLORS = ['#009E73', '#CC79A7', '#E69F00']   # radüs izi (form yüzeyi) renkleri, kanal bazlı
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
DEFAULT_G0_FEED = 15000.0        # G0 (hızlı hareket) varsayılan hızı, mm/dk (satırda F yazılmamışsa)
DEFAULT_LABEL_FONT = 7           # bireysel kanal analizi mesafe yazı boyutu
DEFAULT_LABEL_SHIFT = 0.0        # tüm mesafe yazılarını yukarı(+)/aşağı(-) kaydırma (mm)
DEFAULT_XNAME, DEFAULT_YNAME = 'X', 'Y'    # grafiklerde yatay / dikey eksen adı
DEFAULT_SHOW_ENV = True          # radüs izini (form yüzeyi) göster
DEFAULT_ENV_LIMIT = 0.0          # radüs izi sınırı = ham malzeme kalınlığı (mm); 0 = takım yarıçapı
CONTACT_MODE_CHOICES = [
    ('Geometrik: takım dairesi ile kalıp arasındaki en dar nokta (G40/G41/G42 fark etmez)', 'geometric'),
    ('Programlanan: G41/G42 konturu, G40 takım merkezi (eski davranış)', 'programmed'),
]

DIA_MODE_CHOICES = [('Yarıçap', 'radius'), ('Çap (÷2)', 'diameter')]
DIA_AXIS_CHOICES = [('X ekseni', 'X'), ('Y ekseni', 'Y'), ('X ve Y', 'XY')]

CH_STRIDE = 8                       # kanal başına bileşen: x0,y0,tr,fd,y_offset,code,dia_mode,dia_axis
MOLD_START = N_CH * CH_STRIDE       # 24: kalıp x0,y0,code,dia_mode,dia_axis
N_CFG = MOLD_START + 5 + 1 + 1      # 31: kalıp(5) + çizilen yol modu + G0 hızı
N_OPT = 13                          # grafik seçenekleri: yazı boyutu, kaydırma, yatay ad, dikey ad, radüs izi, iz sınırı,
                                    # ön form: kullan, kaydırma X, kaydırma Y, seçili katmanlar, model, radyal eksen, simülasyon


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
    g0 = _num(v[m + 6], DEFAULT_G0_FEED)
    return {'channels': channels, 'mold': mold, 'contact_mode': mode, 'g0_feed': g0}


def opts_from_values(v):
    """[yazı boyutu, kaydırma, yatay ad, dikey ad, radüs izi göster, iz sınırı] -> grafik seçenekleri."""
    v = list(v) + [None] * (N_OPT - len(v))
    font = min(max(_num(v[0], DEFAULT_LABEL_FONT), 3.0), 40.0)
    return {'label_font': font, 'label_shift': _num(v[1], DEFAULT_LABEL_SHIFT),
            'xname': (str(v[2]).strip() if v[2] is not None else '') or DEFAULT_XNAME,
            'yname': (str(v[3]).strip() if v[3] is not None else '') or DEFAULT_YNAME,
            'show_env': DEFAULT_SHOW_ENV if v[4] is None else bool(v[4]),
            'env_limit': max(_num(v[5], DEFAULT_ENV_LIMIT), 0.0),
            'pf_use': True if v[6] is None else bool(v[6]),
            'pf_dx': _num(v[7], 0.0), 'pf_dy': _num(v[8], 0.0),
            'pf_layers': list(v[9]) if isinstance(v[9], (list, tuple)) else [],
            'pf_model': v[10] if v[10] in ('strip', 'shear', 'flow') else 'strip',
            'pf_radial': v[11] if v[11] in ('auto', 'X', 'Y') else 'auto',
            'pf_sim': bool(v[12]) if v[12] is not None else False}


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
def get_mold_points_cfg(mold, g0_feed=DEFAULT_G0_FEED):
    code, x0, y0 = apply_diameter(mold['code'], mold['x0'], mold['y0'],
                                  mold.get('dia_mode'), mold.get('dia_axis'))
    mold_cmds = parse_gcode(code)
    mold_segs = simulate_channel(mold_cmds, x0, y0, 0.0, 1000.0, g0_feed=g0_feed)
    pts = [(x0, y0)]
    for seg in mold_segs:
        if seg['type'] == 'move':
            pts.extend(seg['points'])
    return pts


def build_channels_cfg(cfg):
    g0_feed = cfg.get('g0_feed', DEFAULT_G0_FEED)
    mold_points = get_mold_points_cfg(cfg['mold'], g0_feed)
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
        tr = ToolR(ch['tr'], ch.get('shape'))           # yarıçap (kompanzasyon) + varsa DXF makara profili
        segs = simulate_channel(cmds, x0, y0, float(tr), ch['fd'], y_offset=ch['y_offset'], g0_feed=g0_feed)
        segs = apply_corner_compensation(segs, float(tr))
        if use_geo:
            segs = apply_geometric_contact(segs, mold_points, tr, (x0, y0))
            gaps = [s['gap_min'] for s in segs if 'gap_min' in s]
            if gaps and min(gaps) < -0.05:
                notes.append(f"Uyarı: Kanal {ci + 1} takım dairesi kalıba giriyor "
                             f"(en küçük boşluk = {min(gaps):.2f}). Takım yarıçapını / yolu / G41-G42 yönünü kontrol et.")
        channels_segments.append(segs)
        start_points.append((x0, y0))
        radii.append(tr)
    return channels_segments, start_points, radii, mold_points, "\n".join(notes)


# ===== Hücre 14 =====
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle, Polygon as MplPolygon
from PIL import Image

# -------------------------------------------------------------
# TAKIM (MAKARA) ŞEKLİ: yarıçap + isteğe bağlı DXF profili
#  ToolR bir float'tır (G41/G42 kompanzasyonu ve örnekleme bu yarıçapı kullanır, CNC'deki gibi);
#  .shape varsa (merkeze göre kapalı poligon, DXF'teki 0,0 = makara merkezi) radüs izi, temas noktası,
#  takım–kalıp mesafesi ve çizimler bu gerçek profil ile yapılır.
# -------------------------------------------------------------
class ToolR(float):
    def __new__(cls, r, shape=None):
        o = float.__new__(cls, float(r))
        o.shape = None if shape is None or len(shape) < 3 else _close_poly(np.asarray(shape, float))
        return o

    def __reduce__(self):
        return (ToolR, (float(self), self.shape))


def _close_poly(P):
    P = np.asarray(P, float)
    if np.hypot(*(P[0] - P[-1])) > 1e-9:
        P = np.vstack([P, P[:1]])
    return P


def tool_shape(r):
    return getattr(r, 'shape', None)


def tool_reach(r):
    """Takım merkezinden profilin en uzak noktasına mesafe (daire için yarıçap)."""
    S = tool_shape(r)
    return float(np.max(np.hypot(S[:, 0], S[:, 1]))) if S is not None else float(r)


def tool_outline(r, n=120):
    """Merkeze göre takım anahattı (kapalı)."""
    S = tool_shape(r)
    if S is not None:
        return S
    a = np.linspace(0, 2 * np.pi, n + 1)
    return np.c_[float(r) * np.cos(a), float(r) * np.sin(a)]


def _poly_ray_s(O, Nr, A, D):
    """Işın başlangıçları O (R,K,2) (poligon merkezine göre), yönleri Nr (R,2), kenarlar A,D (E,2).
    Daire sürümüyle aynı anlam: başlangıç poligon içindeyse 0, değilse ışın doğrultusundaki (iki yön)
    en yakın kesişim (işaretli), hiç yoksa inf."""
    nx, ny = Nr[:, None, None, 0], Nr[:, None, None, 1]
    ax, ay = A[None, None, :, 0] - O[..., 0][..., None], A[None, None, :, 1] - O[..., 1][..., None]
    dx, dy = D[None, None, :, 0], D[None, None, :, 1]
    den = nx * dy - ny * dx
    with np.errstate(divide='ignore', invalid='ignore'):
        sv = (ax * dy - ay * dx) / den
        tv = (ax * ny - ay * nx) / den
    ok = (np.abs(den) > 1e-12) & (tv >= 0) & (tv < 1)
    inside = (np.sum(ok & (sv > 0), axis=-1) % 2) == 1
    sa = np.where(ok, sv, np.inf)
    k = np.argmin(np.abs(sa), axis=-1)
    best = np.take_along_axis(sa, k[..., None], axis=-1)[..., 0]
    return np.where(inside, 0.0, best)


def tool_ray_matrix(M, Nrm, C, tool_r):
    """(R,K) işaretli mesafe matrisi: her ışının her takım konumuna ilk girişi (daire ya da DXF profili)."""
    S = tool_shape(tool_r)
    if S is None:
        r2 = float(tool_r) ** 2
        dx = M[:, None, 0] - C[None, :, 0]
        dy = M[:, None, 1] - C[None, :, 1]
        b = Nrm[:, None, 0] * dx + Nrm[:, None, 1] * dy
        cterm = dx * dx + dy * dy - r2
        disc = b * b - cterm
        with np.errstate(invalid='ignore'):
            sq = np.sqrt(np.maximum(disc, 0.0))
        s = np.where(cterm <= 0, 0.0, -b + np.sign(b) * sq)
        return np.where((cterm > 0) & (disc < 0), np.inf, s)
    O = M[:, None, :] - C[None, :, :]
    return _poly_ray_s(O, Nrm, S[:-1], S[1:] - S[:-1])


def tool_contact_pts(C, mold, tool_r):
    """DXF profilli takım için: her merkezde profilin kalıba en yakın noktası ve (boşluk + yarıçap).
    Daire sürümüyle uyumlu: dönen d - yarıçap = boşluk (kalıba girerse negatif)."""
    S = tool_shape(tool_r)
    V = S[:-1]
    A = mold[:-1]; AB = mold[1:] - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-18)
    E0, ED = S[:-1], S[1:] - S[:-1]
    EL2 = np.maximum((ED ** 2).sum(1), 1e-18)
    out = np.empty_like(C); gap = np.empty(len(C))
    R = tool_reach(tool_r)
    for j, c in enumerate(C):
        dc = np.hypot(mold[:, 0] - c[0], mold[:, 1] - c[1])
        segd = np.minimum(dc[:-1], dc[1:])
        # en yakın kalıp parçalarına sınırla (en yakın mesafe en fazla merkez mesafesi + erişim kadar)
        lim = float(np.min(np.hypot(*(np.clip(((c - A) * AB).sum(1) / L2, 0, 1)[:, None] * AB + A - c).T))) + R
        keep = segd <= lim + np.hypot(*AB.T)
        Ak, ABk, L2k = A[keep], AB[keep], L2[keep]
        P = V + c
        t = np.clip(((P[:, None, :] - Ak[None]) * ABk[None]).sum(2) / L2k[None], 0, 1)
        Q = Ak[None] + t[..., None] * ABk[None]
        d2 = ((Q - P[:, None, :]) ** 2).sum(2)
        iv, ik = np.unravel_index(np.argmin(d2), d2.shape)
        best, bp = float(np.sqrt(d2[iv, ik])), P[iv]
        mk = np.unique(np.r_[np.where(keep)[0], np.where(keep)[0] + 1])
        Mk = mold[mk]
        Rel = Mk - c
        u = np.clip(((Rel[:, None, :] - E0[None]) * ED[None]).sum(2) / EL2[None], 0, 1)
        Pe = E0[None] + u[..., None] * ED[None]
        d2e = ((Pe - Rel[:, None, :]) ** 2).sum(2)
        im, ie = np.unravel_index(np.argmin(d2e), d2e.shape)
        if np.sqrt(d2e[im, ie]) < best:
            best, bp = float(np.sqrt(d2e[im, ie])), Pe[im, ie] + c
        # kalıp noktası profilin içindeyse: girmiş (negatif boşluk)
        inside = _pts_in_poly(Rel, S)
        if inside.any():
            dins = np.sqrt(d2e.min(1))[inside]
            best = -float(dins.max())
        out[j], gap[j] = bp, best
    return out, gap + float(tool_r)


def _pts_in_poly(P, S):
    x, y = P[:, 0][:, None], P[:, 1][:, None]
    x0, y0, x1, y1 = S[:-1, 0][None], S[:-1, 1][None], S[1:, 0][None], S[1:, 1][None]
    with np.errstate(divide='ignore', invalid='ignore'):
        cr = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0)
    return (cr.sum(1) % 2) == 1


def simplify_poly(P, tol=0.005, max_pts=160):
    """Douglas-Peucker sadeleştirme (kapalı poligon); çok noktalı DXF/CMM profillerini hafifletir."""
    P = _close_poly(P)
    def dp(Q, eps):
        if len(Q) < 3:
            return Q
        a, b = Q[0], Q[-1]
        ab = b - a; L = np.hypot(*ab)
        if L < 1e-12:
            d = np.hypot(*(Q - a).T)
        else:
            d = np.abs(ab[0] * (Q[:, 1] - a[1]) - ab[1] * (Q[:, 0] - a[0])) / L
        i = int(np.argmax(d))
        if d[i] <= eps:
            return np.vstack([a, b])
        return np.vstack([dp(Q[:i + 1], eps)[:-1], dp(Q[i:], eps)])
    far = int(np.argmax(np.hypot(*(P - P[0]).T)))
    eps = tol
    while True:
        R1, R2 = dp(P[:far + 1], eps), dp(P[far:], eps)
        out = np.vstack([R1[:-1], R2])
        if len(out) <= max_pts or eps > 1.0:
            return out
        eps *= 1.6



def _is_g0(seg):
    if 'g0' in seg:                                   # komuttan gelen kesin bilgi
        return bool(seg['g0'])
    return seg.get('feed', 1000.0) >= 3000.0          # eski segmentler için yedek kural


def _plot_limits(mold_points, start_points, radii, margin_mult, segs=None, extra=None):
    act = [any(s.get('type') in ('move', 'dwell') for s in sg) for sg in segs] if segs else [True] * len(start_points)
    sp = [p for p, a in zip(start_points, act) if a]
    all_x = [p[0] for p in mold_points] + [s[0] for s in sp]
    all_y = [p[1] for p in mold_points] + [s[1] for s in sp]
    for P in (extra or []):
        P = np.asarray(P, float)
        if len(P):
            all_x.extend(P[:, 0]); all_y.extend(P[:, 1])
    if segs:
        for sg, a in zip(segs, act):
            if not a:
                continue
            for s in sg:
                if s.get('type') == 'move' and not _is_g0(s) and s.get('orig_points'):
                    P = np.asarray(s['orig_points'], float)
                    all_x.extend(P[:, 0]); all_y.extend(P[:, 1])
    ra = [tool_reach(r) for r, a in zip(radii, act) if a] or [0.0]
    margin = max(5.0, max(ra) * margin_mult * 0.5)
    return (min(all_x) - margin, max(all_x) + margin), (min(all_y) - margin, max(all_y) + margin)


def _form_extra(form):
    """Grafik sınırlarına eklenecek ön form ve ürün noktaları."""
    out = []
    if not form:
        return out
    out += [A for A, _cl in form.get('pf_chains') or []]
    sm = (form.get('final') or {}).get('sim')
    if sm and sm.get('ok') and sm.get('final') is not None:
        f = sm['final']
        polys = f['polys'] if 'polys' in f else [f.get('fixed'), f.get('wall'), f.get('rem')]
        out += [P for P in polys if P is not None]
    return out


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
# RADÜS İZİ (FORM YÜZEYİ): takım dairelerinin kalıba bakan zarfı.
# Kalıbın her noktasından normal yönünde (iki yöne) ışın atılır; ışının ilk
# kestiği takım dairesi o noktadaki form yüzeyini verir. Takım ilerlemese,
# sadece yanaşsa bile radüs şekli böylece izde görünür.
# Kalıp ile bu iz arasında kalan bölge = inceltilmiş malzeme.
# -------------------------------------------------------------
def mold_rays(mold_points, max_pts=3000, fan_deg=8.0):
    """Kalıp konturunu sıklaştırır ve her noktaya birim normal verir.
    Köşelerde normali yelpaze gibi döndüren ek ışınlar ekler (köşe çevresinde iz kopmasın).
    Döndürür: (M (N,2), Nrm (N,2), fan (N,) bool: köşe yelpazesi ışını mı)"""
    P = np.asarray(mold_points, float)
    if len(P) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, bool)
    keep = np.r_[True, np.hypot(*(P[1:] - P[:-1]).T) > 1e-9]
    P = P[keep]
    if len(P) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, bool)
    seg_len = np.hypot(*(P[1:] - P[:-1]).T)
    step = max(0.25, float(seg_len.sum()) / max_pts)
    dirs = (P[1:] - P[:-1]) / seg_len[:, None]
    nrm = np.c_[-dirs[:, 1], dirs[:, 0]]                       # sol normal
    M, N, B = [], [], []
    for k in range(len(P) - 1):
        if k > 0:                                              # köşe yelpazesi
            a0 = np.arctan2(nrm[k - 1, 1], nrm[k - 1, 0])
            a1 = np.arctan2(nrm[k, 1], nrm[k, 0])
            da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
            nf = int(abs(np.degrees(da)) // fan_deg)
            for f in range(1, nf + 1):
                a = a0 + da * f / (nf + 1)
                M.append(P[k]); N.append((np.cos(a), np.sin(a))); B.append(True)
        n = max(int(np.ceil(seg_len[k] / step)), 1)
        for j in range(n):
            M.append(P[k] + dirs[k] * seg_len[k] * j / n); N.append(nrm[k]); B.append(False)
    M.append(P[-1]); N.append(nrm[-1]); B.append(False)
    return np.asarray(M), np.asarray(N), np.asarray(B)


def tool_samples(timeline, tool_r):
    """Kanalın takım merkezi yolunu (G0 hariç, hareket + bekleme) zamanıyla birlikte sık örnekler.
    Döndürür: (C (K,2) merkezler, T (K,) zamanlar) - zamana göre sıralı."""
    step = max(0.15, tool_r / 30.0)
    C, T = [], []
    for (t0, t1, seg) in timeline:
        if seg['type'] == 'dwell':
            C.append(seg['point']); T.append(t0)
            continue
        if seg['type'] != 'move' or _is_g0(seg):
            continue
        pts = np.asarray(seg['points'], float)
        if len(pts) == 1:
            C.append(pts[0]); T.append(t0)
            continue
        n_seg = len(pts) - 1
        for i in range(n_seg):
            a, b = pts[i], pts[i + 1]
            L = float(np.hypot(*(b - a)))
            m = max(int(np.ceil(L / step)), 1)
            for j in range(m + (1 if i == n_seg - 1 else 0)):
                u = j / m
                C.append(a + (b - a) * u)
                T.append(t0 + (t1 - t0) * (i + u) / n_seg)
    if not C:
        return np.zeros((0, 2)), np.zeros(0)
    return np.asarray(C, float), np.asarray(T, float)


def radius_envelope(M, Nrm, C, tool_r, chunk=384, maxs=150.0):
    """Her kalıp noktası için en yakın takım kesişimini bulur (daire ya da DXF profili).
    Döndürür: s (N,) işaretli mesafe (+ normal yönü, - ters yönü; inf = hiç kesmedi).
    Takım kalıp noktasını içine alıyorsa s = 0. DXF profilinde merkeze (erişim + maxs)'tan uzak
    ışınlar hesaplanmaz (inf)."""
    N_m = len(M)
    best = np.full(N_m, np.inf)
    if N_m == 0 or len(C) == 0:
        return best
    S = tool_shape(tool_r)
    if S is None:
        for a in range(0, len(C), chunk):
            s = tool_ray_matrix(M, Nrm, C[a:a + chunk], tool_r)
            k = np.argmin(np.abs(s), axis=1)
            cand = s[np.arange(N_m), k]
            better = np.abs(cand) < np.abs(best)
            best[better] = cand[better]
        return best
    reach = tool_reach(tool_r) + maxs
    ck = max(1, int(60000 // max(len(S), 1) // 64))
    for a in range(0, len(C), ck):
        Cc = C[a:a + ck]
        lo, hi = Cc.min(0) - reach, Cc.max(0) + reach
        rid = np.where((M[:, 0] >= lo[0]) & (M[:, 0] <= hi[0]) & (M[:, 1] >= lo[1]) & (M[:, 1] <= hi[1]))[0]
        if not len(rid):
            continue
        for b0 in range(0, len(rid), 256):
            r = rid[b0:b0 + 256]
            s = tool_ray_matrix(M[r], Nrm[r], Cc, tool_r)
            k = np.argmin(np.abs(s), axis=1)
            cand = s[np.arange(len(r)), k]
            better = np.abs(cand) < np.abs(best[r])
            best[r[better]] = cand[better]
    return best


def envelope_chunks(M, Nrm, s, limit, brk_jump=None, ok=None):
    """İşaretli mesafelerden çizilecek iz parçalarını üretir.
    limit'ten uzak (form olmamış) noktalar ve yön değişimleri izi böler.
    Döndürür: [(iz_noktaları (k,2), kalıp_noktaları (k,2)), ...]"""
    if len(M) == 0:
        return []
    if ok is None:
        ok = np.isfinite(s) & (np.abs(s) <= limit)
    E = M + Nrm * np.where(ok, s, 0.0)[:, None]
    side = np.sign(s)
    if brk_jump is None:
        brk_jump = max(2.0, 0.5 * limit)
    chunks, cur = [], []
    for i in range(len(M)):
        if not ok[i]:
            if len(cur) > 1:
                chunks.append(cur)
            cur = []
            continue
        if cur:
            j = cur[-1]
            jump = float(np.hypot(*(E[i] - E[j])))
            if (side[i] != side[j] and side[i] != 0 and side[j] != 0) or jump > brk_jump:
                if len(cur) > 1:
                    chunks.append(cur)
                cur = []
        cur.append(i)
    if len(cur) > 1:
        chunks.append(cur)
    return [(E[idx], M[idx]) for idx in chunks]


def compute_envelopes(timelines, radii, mold_points, limits):
    """Tüm kanallar için radüs izini hesaplar. limits[i] <= 0 ise takım yarıçapı kullanılır."""
    M, Nrm, _ = mold_rays(mold_points)
    out = []
    for i in range(N_CH):
        lim = limits[i] if limits[i] > 0 else radii[i]
        C, _T = tool_samples(timelines[i], radii[i])
        s = radius_envelope(M, Nrm, C, radii[i])
        out.append({'chunks': envelope_chunks(M, Nrm, s, lim), 'limit': lim})
    return out


def _draw_envelope(ax, env, color, label, fill=True):
    first = True
    for E, Mk in env['chunks']:
        poly = np.vstack([E, Mk[::-1]])
        if fill:
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.32, linewidth=0, zorder=1)
        ax.plot(E[:, 0], E[:, 1], color=color, linewidth=1.6, linestyle=(0, (5, 2)),
                label=(label if first else ''), zorder=3)
        first = False


# -------------------------------------------------------------
# 7b) Büyük karşılaştırma grafiği (G0 hariç)
# -------------------------------------------------------------
def plot_comparison_fig(channels_segments, start_points, radii, mold_points, xname='X', yname='Y',
                        envelopes=None, form=None):
    fig = Figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.80)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title('Büyük Boyutlu Net Kontur Karşılaştırması (G0 Hareketi Hariç - Her Kod Satırı Farklı Renk)',
                 fontsize=14, pad=15)
    ax.set_xlabel(xname, fontsize=11)
    ax.set_ylabel(yname, fontsize=11)

    if len(mold_points) > 1:
        ax.plot([p[0] for p in mold_points], [p[1] for p in mold_points],
                color='black', linewidth=1.2, linestyle='-', label='Kalıp Konturu (Sabit)')
    if form and form.get('pf_chains'):
        draw_preform_outline(ax, form['pf_chains'])
    if form and form.get('final'):
        _sm = form['final'].get('sim')
        if _sm and _sm.get('ok') and _sm.get('final') is not None:
            draw_sim_product(ax, _sm['final'])
            for _lbl, _col, _chunks in form['final'].get('lines', []):
                for _k, P in enumerate(_chunks):
                    ax.plot(P[:, 0], P[:, 1], color=_col, linewidth=1.4, linestyle=(0, (4, 2)),
                            label=(_lbl if _k == 0 else ''), zorder=5)
        else:
            draw_final_form(ax, form['final'])

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
        if envelopes:
            _draw_envelope(ax, envelopes[i], CHANNEL_ENV_COLORS[i], f'Kanal {i+1} - Radüs İzi (Form Yüzeyi)',
                           fill=not (form and form.get('final')))
        if not any(_s.get('type') in ('move', 'dwell') for _s in channels_segments[i]):
            continue
        sx, sy = start_points[i]
        ax.plot(sx, sy, 'o', color=CHANNEL_COLORS[0], markersize=8,
                label=f'Kanal {i+1} Başlangıç ({sx:.1f}, {sy:.1f})')

    xl, yl = _plot_limits(mold_points, start_points, radii, 3.0, channels_segments, _form_extra(form))
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


def _place_distance_labels(fig, ax, labels, polylines, fontsize=7, shift=0.0):
    """Mesafe etiketlerini birbirine ve çizilen yollara/çizgilere değmeyecek şekilde yerleştirir.
    labels: [(nokta, metin, (dx, dy) tercih edilen yön)], polylines: kaçınılacak çizgiler (veri koordinatı).
    Yazılar dikey (90°) çizilir; shift (mm) tüm yazıları birlikte yukarı(+)/aşağı(-) kaydırır."""
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
        probe = ax.text(0, 0, text, fontsize=fontsize, fontweight='bold', rotation=90)
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
        ly += shift                                              # toplu yukarı/aşağı kaydırma (mm)
        end_d = tr.transform((lx, ly))
        far = float(np.hypot(end_d[0] - k_d[0], end_d[1] - k_d[1])) > max(w, h) / 2 + 14
        arrow = dict(arrowstyle='-', color='darkred', lw=0.5, alpha=0.8, shrinkA=0, shrinkB=1) if far else None
        ax.annotate(text, xy=kp, xytext=(lx, ly), textcoords='data', ha='center', va='center', rotation=90,
                    fontsize=fontsize, fontweight='bold', color='darkred', arrowprops=arrow,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.15'))


def plot_channel_fig(i, channels_segments, start_points, radii, mold_points,
                     xname='X', yname='Y', label_font=7.0, label_shift=0.0, envelope=None, pf_chains=None):
    closest = _mold_closest_fn(mold_points)
    xl, yl = _plot_limits(mold_points, start_points, radii, 3.0, channels_segments,
                          [A for A, _cl in (pf_chains or [])])

    fig = Figure(figsize=(16, 11), dpi=100)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.84)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_title(f'Kanal {i+1} Detaylı Kontur ve Kalıp Mesafesi Analizi', fontsize=16, pad=14)
    ax.set_xlabel(xname, fontsize=12)
    ax.set_ylabel(yname, fontsize=12)

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

    if pf_chains:
        draw_preform_outline(ax, pf_chains)
        for A, _cl in pf_chains:
            polylines.append(A)
    if envelope:
        _draw_envelope(ax, envelope, CHANNEL_ENV_COLORS[i], 'Radüs İzi (Form Yüzeyi)')
        for E, _Mk in envelope['chunks']:
            polylines.append(E)
    sx, sy = start_points[i]
    ax.plot(sx, sy, 'o', color=CHANNEL_COLORS[0], markersize=10, label='Başlangıç Noktası')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0, fontsize=10, shadow=True)
    _place_distance_labels(fig, ax, labels, polylines, fontsize=label_font, shift=label_shift)
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
# DXF ÇIKTISI (AutoCAD R12 / AC1009 - ek kütüphane gerektirmez)
# -------------------------------------------------------------
DXF_LAYER_CHOICES = [
    ('Kalıp konturu', 'mold'),
    ('Kontur (temas yolu)', 'contour'),
    ('Radüs izi (form yüzeyi)', 'envelope'),
    ('Takım merkezi yolu', 'center'),
    ('Zaman adımlı takım çemberleri', 'circles'),
    ('G0 hareketleri (temas yolu)', 'g0'),
    ('Ön form (DXF)', 'preform'),
    ('Son form yüzeyi (ön form ile)', 'final'),
    ('Ürün (simülasyon sonu, hacim sabit)', 'product'),
]
DXF_DEFAULT_LAYERS = ['mold', 'contour', 'envelope', 'circles', 'preform', 'final']
_DXF_ACI = {'contour': [5, 30, 4], 'envelope': [3, 6, 1], 'center': [8, 8, 8], 'circles': [8, 8, 8], 'g0': [9, 9, 9]}
_DXF_NAME = {'contour': 'KONTUR', 'envelope': 'RADUS_IZI', 'center': 'TAKIM_MERKEZI', 'circles': 'CEMBER', 'g0': 'G0'}


class _DxfWriter:
    def __init__(self):
        self.layers = {}          # ad -> renk (ACI)
        self.ent = []
        self.ext = [np.inf, np.inf, -np.inf, -np.inf]

    def _grow(self, x, y, r=0.0):
        e = self.ext
        e[0] = min(e[0], x - r); e[1] = min(e[1], y - r); e[2] = max(e[2], x + r); e[3] = max(e[3], y + r)

    def layer(self, name, color):
        self.layers.setdefault(name, int(color))

    def polyline(self, layer, pts, closed=False):
        P = [(float(p[0]), float(p[1])) for p in pts if np.isfinite(p[0]) and np.isfinite(p[1])]
        if len(P) < 2:
            return
        out = ['0', 'POLYLINE', '8', layer, '66', '1', '70', '1' if closed else '0',
               '10', '0.0', '20', '0.0', '30', '0.0']
        for x, y in P:
            out += ['0', 'VERTEX', '8', layer, '10', f'{x:.6f}', '20', f'{y:.6f}', '30', '0.0']
            self._grow(x, y)
        out += ['0', 'SEQEND', '8', layer]
        self.ent.append(out)

    def circle(self, layer, c, r):
        x, y = float(c[0]), float(c[1])
        if not (np.isfinite(x) and np.isfinite(y)) or r <= 0:
            return
        self.ent.append(['0', 'CIRCLE', '8', layer, '10', f'{x:.6f}', '20', f'{y:.6f}', '30', '0.0',
                         '40', f'{float(r):.6f}'])
        self._grow(x, y, r)

    def text(self):
        e = self.ext if np.isfinite(self.ext[0]) else [0, 0, 1, 1]
        L = ['0', 'SECTION', '2', 'HEADER',
             '9', '$ACADVER', '1', 'AC1009',
             '9', '$INSUNITS', '70', '4',
             '9', '$EXTMIN', '10', f'{e[0]:.6f}', '20', f'{e[1]:.6f}', '30', '0.0',
             '9', '$EXTMAX', '10', f'{e[2]:.6f}', '20', f'{e[3]:.6f}', '30', '0.0',
             '0', 'ENDSEC',
             '0', 'SECTION', '2', 'TABLES',
             '0', 'TABLE', '2', 'LTYPE', '70', '1',
             '0', 'LTYPE', '2', 'CONTINUOUS', '70', '0', '3', 'Solid line', '72', '65', '73', '0', '40', '0.0',
             '0', 'ENDTAB',
             '0', 'TABLE', '2', 'LAYER', '70', str(len(self.layers) + 1),
             '0', 'LAYER', '2', '0', '70', '0', '62', '7', '6', 'CONTINUOUS']
        for name, col in self.layers.items():
            L += ['0', 'LAYER', '2', name, '70', '0', '62', str(col), '6', 'CONTINUOUS']
        L += ['0', 'ENDTAB', '0', 'ENDSEC', '0', 'SECTION', '2', 'ENTITIES']
        for e_ in self.ent:
            L += e_
        L += ['0', 'ENDSEC', '0', 'EOF']
        return "\n".join(L) + "\n"


def _split_nan(pts):
    """NaN ile işaretlenmiş kopuklukları ayrı parçalara böler."""
    out, cur = [], []
    for p in pts:
        if not (np.isfinite(p[0]) and np.isfinite(p[1])):
            if len(cur) > 1:
                out.append(cur)
            cur = []
        else:
            cur.append(p)
    if len(cur) > 1:
        out.append(cur)
    return out


def build_dxf_text(layers, channels_segments, radii, mold_points, timelines, total_time,
                   circ_start, circ_end, circ_step, envelopes=None):
    """Seçilen katmanlarla DXF metnini üretir. Döndürür: (dxf_metni, özet_satırları)"""
    layers = set(layers or [])
    w = _DxfWriter()
    summary = []
    if 'mold' in layers and len(mold_points) > 1:
        w.layer('KALIP', 7)
        w.polyline('KALIP', mold_points)
        summary.append('KALIP: kalıp konturu')
    if 'circles' in layers:
        if circ_end <= circ_start or circ_step <= 0:
            raise ValueError("Çemberler için geçersiz zaman aralığı (Bitiş > Başlangıç ve Adım > 0 olmalı).")
        n = int(np.floor((circ_end - circ_start) / circ_step + 1e-9)) + 1
        if n > 20000:
            raise ValueError(f"Çok fazla çember ({n}). Zaman adımını büyüt.")
    for i in range(N_CH):
        segs = channels_segments[i]
        if 'contour' in layers or 'g0' in layers:
            for seg in segs:
                if seg['type'] != 'move' or not seg['orig_points']:
                    continue
                key = 'g0' if _is_g0(seg) else 'contour'
                if key not in layers:
                    continue
                name = f"K{i+1}_{_DXF_NAME[key]}"
                w.layer(name, _DXF_ACI[key][i])
                for pts in _contact_chunks(seg):
                    w.polyline(name, pts)
        if 'center' in layers:
            name = f"K{i+1}_{_DXF_NAME['center']}"
            w.layer(name, _DXF_ACI['center'][i])
            for pts in _split_nan(get_trail(timelines[i], total_time + 1.0, 'points')):
                w.polyline(name, pts)
        if 'envelope' in layers and envelopes:
            name = f"K{i+1}_{_DXF_NAME['envelope']}"
            w.layer(name, _DXF_ACI['envelope'][i])
            for E, _Mk in envelopes['ch'][i]['chunks']:
                w.polyline(name, E)
        if 'circles' in layers:
            name = f"K{i+1}_{_DXF_NAME['circles']}"
            w.layer(name, _DXF_ACI['circles'][i])
            for t in circ_start + circ_step * np.arange(n):
                t_c = max(0.0, min(float(t), total_time))
                _c = np.asarray(get_position(timelines[i], t_c, 'points'), float)
                if tool_shape(radii[i]) is not None:
                    w.polyline(name, tool_shape(radii[i]) + _c)
                else:
                    w.circle(name, _c, radii[i])
    if envelopes and envelopes.get('pf_chains') and 'preform' in layers:
        w.layer('ON_FORM', 8)
        for A, cl in envelopes['pf_chains']:
            w.polyline('ON_FORM', A)
    if envelopes and envelopes.get('final') and 'final' in layers:
        w.layer('SON_FORM', 5)
        for P, _Mk in envelopes['final']['chunks']:
            w.polyline('SON_FORM', P)
        for _lbl, _col, chunks in envelopes['final'].get('lines', []):
            w.layer('IDEAL_FORM', 6)
            for P in chunks:
                w.polyline('IDEAL_FORM', P)
    _sm = (envelopes or {}).get('final', {}) or {}
    _sm = _sm.get('sim') if isinstance(_sm, dict) else None
    if 'product' in layers:
        if _sm and _sm.get('ok') and _sm.get('final') is not None:
            w.layer('URUN', 4)
            _f = _sm['final']
            _ps = _f['polys'] if 'polys' in _f else [_f.get('fixed'), _f.get('wall'), _f.get('bridge'), _f.get('rem')]
            for P in _ps:
                if P is not None:
                    w.polyline('URUN', P)
        else:
            summary.append('URUN: simülasyon modu kapalı / çalışmadı, atlandı')
    if ('preform' in layers or 'final' in layers) and not (envelopes and envelopes.get('pf_chains')):
        summary.append('ON_FORM/SON_FORM: ön form yüklenmediği için atlandı')
    for name in w.layers:
        if name != 'KALIP':
            cnt = sum(1 for e_ in w.ent if e_[3] == name)
            summary.append(f"{name}: {cnt} nesne")
    return w.text(), summary


# ===== Hücre 16 =====
# -------------------------------------------------------------
# ÖN FORM (DXF): lokal malzeme kalınlığı
#  - DXF'teki ön form çizgileri okunur (LINE, LWPOLYLINE, POLYLINE, ARC, CIRCLE, ELLIPSE, SPLINE)
#  - Kalıbın her noktasından normal yönünde ışın atılır; ışının ön formun dış yüzeyini kestiği
#    mesafe o noktadaki malzeme kalınlığıdır (kalıptan dış yüzeye).
#  - Takım bu kalınlıktan daha derine girdiği yerde form oluşur; 3 kanal aynı parçayı işlediği için
#    son yüzey, kanalların en derin girdiği yerdir: SON = min(ÖN FORM, K1, K2, K3)
# -------------------------------------------------------------
import collections

PF_GAP_TOL = 3.0          # ön formun iç yüzeyi ile kalıp arasında kabul edilen en büyük boşluk (mm)


def _dxf_pairs(text):
    lines = text.splitlines()
    out = []
    for i in range(0, len(lines) - 1, 2):
        try:
            out.append((int(lines[i].strip()), lines[i + 1].strip()))
        except ValueError:
            continue
    return out


DXF_ARC_TOL = [0.02]      # yay kiriş toleransı (mm); makara / kalıp DXF okunurken geçici olarak inceltilir


def _arc_pts(cx, cy, r, a0, a1, tol=None):
    """a0'dan a1'e (radyan, işaretli tarama) yay noktaları."""
    tol = DXF_ARC_TOL[0] if tol is None else tol
    if r <= 0:
        return []
    sweep = a1 - a0
    da = 2 * np.arccos(max(min(1 - tol / r, 1.0), -1.0)) if r > tol else 0.5
    da = min(max(da, 0.005), 0.2)
    n = max(int(np.ceil(abs(sweep) / da)), 2)
    a = a0 + sweep * np.arange(n + 1) / n
    return list(zip(cx + r * np.cos(a), cy + r * np.sin(a)))


def _bulge_pts(p1, p2, b, tol=None):
    if abs(b) < 1e-9:
        return [p1, p2]
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = float(np.hypot(dx, dy))
    if L < 1e-12:
        return [p1]
    theta = 4 * np.arctan(b)                      # + : saat yönü tersi
    r = L / (2 * abs(np.sin(theta / 2)))
    h = L * (1 - b * b) / (4 * b)                 # kiriş ortasından merkeze (sol dik yönde)
    cx = (x1 + x2) / 2 - dy / L * h
    cy = (y1 + y2) / 2 + dx / L * h
    a0 = np.arctan2(y1 - cy, x1 - cx)
    pts = _arc_pts(cx, cy, r, a0, a0 + theta, tol)
    if pts:
        pts[0], pts[-1] = (x1, y1), (x2, y2)
    return pts


def _nurbs_pts(deg, knots, ctrl, weights, closed, n_per_span=24):
    """De Boor ile (rasyonel) B-spline örnekleme."""
    P = np.asarray(ctrl, float)
    n = len(P)
    if n < 2:
        return [tuple(p) for p in P]
    W = np.asarray(weights, float) if (weights is not None and len(weights) == n) else np.ones(n)
    U = np.asarray(knots, float)
    p = int(deg)
    if len(U) != n + p + 1:                        # düğüm vektörü eksikse kıskaç (clamped) düzgün vektör
        inner = max(n - p - 1, 0)
        U = np.r_[np.zeros(p + 1), np.arange(1, inner + 1) / (inner + 1), np.ones(p + 1)]
    Pw = np.c_[P * W[:, None], W]
    u0, u1 = U[p], U[n]
    spans = max(len(np.unique(U[p:n + 1])) - 1, 1)
    us = np.linspace(u0, u1, spans * n_per_span + 1)
    out = []
    for u in us:
        k = int(np.searchsorted(U, u, side='right') - 1)
        k = min(max(k, p), n - 1)
        d = [Pw[j + k - p].copy() for j in range(p + 1)]
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                den = U[j + 1 + k - r] - U[j + k - p]
                a = 0.0 if den == 0 else (u - U[j + k - p]) / den
                d[j] = (1 - a) * d[j - 1] + a * d[j]
        out.append((d[p][0] / d[p][2], d[p][1] / d[p][2]))
    return out


def read_dxf_polylines(text):
    """ASCII DXF metnini okur. Döndürür: {'layers': {katman: [[[x,y],...], ...]}, 'units', 'skipped', 'counts'}"""
    pairs = _dxf_pairs(text)
    layers = collections.defaultdict(list)
    skipped = collections.Counter()
    counts = collections.Counter()
    units = None
    section = None
    i, n = 0, len(pairs)
    for k in range(n - 1):                         # birim
        if pairs[k] == (9, '$INSUNITS') and pairs[k + 1][0] == 70:
            try:
                units = int(pairs[k + 1][1])
            except ValueError:
                pass
            break

    def grab(j):
        """j'den sonraki ilk 0 koduna kadar olan grubu döndürür."""
        g = []
        j += 1
        while j < n and pairs[j][0] != 0:
            g.append(pairs[j]); j += 1
        return g, j

    def f(g, code, default=0.0):
        for c, v in g:
            if c == code:
                try:
                    return float(v)
                except ValueError:
                    return default
        return default

    def lay(g):
        for c, v in g:
            if c == 8:
                return v or '0'
        return '0'

    def ocs(g, pts):                               # 2D OCS: extrusion -Z ise X aynalanır
        return [(-x, y) for x, y in pts] if f(g, 230, 1.0) < 0 else pts

    while i < n:
        c, v = pairs[i]
        if c == 0 and v == 'SECTION' and i + 1 < n:
            section = pairs[i + 1][1]
            i += 2
            continue
        if c == 0 and v == 'ENDSEC':
            section = None
            i += 1
            continue
        if section != 'ENTITIES' or c != 0:
            i += 1
            continue
        etype = v
        g, j = grab(i)
        L = lay(g)
        pts = None
        if etype == 'LINE':
            pts = [(f(g, 10), f(g, 20)), (f(g, 11), f(g, 21))]
        elif etype == 'LWPOLYLINE':
            verts, bul = [], []
            for cc, vv in g:
                if cc == 10:
                    verts.append([float(vv), 0.0]); bul.append(0.0)
                elif cc == 20 and verts:
                    verts[-1][1] = float(vv)
                elif cc == 42 and verts:
                    bul[-1] = float(vv)
            closed = int(f(g, 70)) & 1
            if closed and len(verts) > 1:
                verts.append(list(verts[0])); bul.append(0.0)
            pts = []
            for a in range(len(verts) - 1):
                seg = _bulge_pts(tuple(verts[a]), tuple(verts[a + 1]), bul[a])
                pts.extend(seg if not pts else seg[1:])
            pts = ocs(g, pts)
        elif etype == 'POLYLINE':
            flags = int(f(g, 70))
            verts, bul = [], []
            while j < n and pairs[j] == (0, 'VERTEX'):
                vg, j = grab(j)
                if int(f(vg, 70)) & 16:            # spline çerçeve noktası, atla
                    continue
                verts.append((f(vg, 10), f(vg, 20))); bul.append(f(vg, 42))
            if j < n and pairs[j] == (0, 'SEQEND'):
                _sg, j = grab(j)
            if flags & (16 | 64):                  # 3B ağ / yüzey
                skipped['POLYLINE(3B)'] += 1
                i = j
                continue
            if (flags & 1) and len(verts) > 1:
                verts.append(verts[0]); bul.append(0.0)
            pts = []
            for a in range(len(verts) - 1):
                seg = _bulge_pts(verts[a], verts[a + 1], bul[a])
                pts.extend(seg if not pts else seg[1:])
            pts = ocs(g, pts)
        elif etype in ('ARC', 'CIRCLE'):
            cx, cy, r = f(g, 10), f(g, 20), f(g, 40)
            if etype == 'CIRCLE':
                a0, a1 = 0.0, 2 * np.pi
            else:
                a0, a1 = np.radians(f(g, 50)), np.radians(f(g, 51))
                while a1 <= a0:
                    a1 += 2 * np.pi
            pts = ocs(g, _arc_pts(cx, cy, r, a0, a1))
        elif etype == 'ELLIPSE':
            cx, cy, mx, my = f(g, 10), f(g, 20), f(g, 11), f(g, 21)
            ratio, t0, t1 = f(g, 40, 1.0), f(g, 41, 0.0), f(g, 42, 2 * np.pi)
            while t1 <= t0:
                t1 += 2 * np.pi
            t = np.linspace(t0, t1, max(int((t1 - t0) / 0.02), 8) + 1)
            ux, uy = mx, my
            vx, vy = -my * ratio, mx * ratio
            pts = list(zip(cx + ux * np.cos(t) + vx * np.sin(t), cy + uy * np.cos(t) + vy * np.sin(t)))
            if f(g, 230, 1.0) < 0:
                pts = [(2 * cx - x, y) for x, y in pts]
        elif etype == 'SPLINE':
            deg = int(f(g, 71, 3))
            knots = [float(vv) for cc, vv in g if cc == 40]
            wts = [float(vv) for cc, vv in g if cc == 41]
            ctrl, fit = [], []
            for cc, vv in g:
                if cc == 10:
                    ctrl.append([float(vv), 0.0])
                elif cc == 20 and ctrl:
                    ctrl[-1][1] = float(vv)
                elif cc == 11:
                    fit.append([float(vv), 0.0])
                elif cc == 21 and fit:
                    fit[-1][1] = float(vv)
            if len(ctrl) >= 2:
                pts = _nurbs_pts(deg, knots, ctrl, wts if wts else None, int(f(g, 70)) & 1)
            elif len(fit) >= 2:
                pts = [tuple(p) for p in fit]
        if pts is not None and len(pts) >= 2:
            layers[L].append([[round(float(x), 6), round(float(y), 6)] for x, y in pts])
            counts[etype] += 1
        elif pts is None:
            skipped[etype] += 1
        i = j
    return {'layers': dict(layers), 'units': units, 'skipped': dict(skipped), 'counts': dict(counts)}


def load_preform_file(path):
    """Yüklenen DXF dosyasını okur, mm'ye çevirir. Döndürür: (state, mesaj)"""
    raw = open(path, 'rb').read()
    if raw[:22] == b'AutoCAD Binary DXF\r\n\x1a\x00':
        raise ValueError("Binary DXF desteklenmiyor. AutoCAD'de DXF'i ASCII olarak kaydet (SAVEAS > DXF).")
    text = None
    for enc in ('utf-8', 'cp1254', 'latin-1'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    res = read_dxf_polylines(text)
    if not res['layers']:
        extra = (" Okunmayan nesneler: " + ", ".join(f"{k}×{v}" for k, v in res['skipped'].items())) if res['skipped'] else ""
        raise ValueError("DXF'te okunabilir çizgi bulunamadı (LINE/POLYLINE/ARC/SPLINE...)." + extra)
    # Koordinatlar her zaman mm kabul edilir (kalıp koordinatlarına uyacak şekilde çizildiği için).
    # Başlıktaki birim ($INSUNITS) çoğu şablonda yanlış kalır (ör. AutoCAD Mechanical ANSI şablonu = inç),
    # bu yüzden ölçekleme YAPILMAZ; sadece bilgi verilir.
    _u = {0: 'birimsiz', 1: 'inç', 2: 'feet', 4: 'mm', 5: 'cm', 6: 'm'}.get(res['units'], str(res['units']))
    unit_note = "mm" if res['units'] in (None, 0, 4) else f"mm kabul edildi (DXF başlığında '{_u}' yazıyor, ölçekleme yapılmadı)"
    state = {'name': os.path.basename(path), 'layers': res['layers']}
    msg = (f"Ön form yüklendi: {state['name']} — birim: {unit_note}. Nesneler: "
           + ", ".join(f"{k}×{v}" for k, v in res['counts'].items())
           + ". Katmanlar: " + ", ".join(f"{k} ({len(v)})" for k, v in res['layers'].items()))
    if res['skipped']:
        msg += ("\nOkunmayan nesneler: " + ", ".join(f"{k}×{v}" for k, v in res['skipped'].items())
                + (" — blok (INSERT) içindeki çizgiler için AutoCAD'de EXPLODE uygula." if 'INSERT' in res['skipped'] else ""))
    return state, msg


def preform_layer_choices(state):
    if not state or not state.get('layers'):
        return []
    return [(f"{k} ({len(v)} çizgi)", k) for k, v in state['layers'].items()]


def _join_chains(state, selected, dx=0.0, dy=0.0, tol=0.01):
    """Seçili katmanlardaki çizgileri uç uca birleştirir.
    Döndürür: [(noktalar (k,2), kapalı_mı), ...]"""
    if not state or not state.get('layers'):
        return []
    sel = set(selected) if selected else set(state['layers'])
    pls = [np.asarray(pl, float).reshape(-1, 2) + (dx, dy) for k, v in state['layers'].items() if k in sel
           for pl in v if len(pl) >= 2]
    used = [False] * len(pls)
    chains = []
    for s0 in range(len(pls)):
        if used[s0]:
            continue
        used[s0] = True
        ch = [p for p in pls[s0]]
        grown = True
        while grown:
            grown = False
            if np.hypot(*(np.asarray(ch[0]) - ch[-1])) < tol and len(ch) > 2:
                break
            for k in range(len(pls)):
                if used[k]:
                    continue
                p = pls[k]
                if np.hypot(*(p[0] - ch[-1])) < tol:
                    ch.extend(p[1:]); used[k] = grown = True
                elif np.hypot(*(p[-1] - ch[-1])) < tol:
                    ch.extend(p[::-1][1:]); used[k] = grown = True
                elif np.hypot(*(p[-1] - ch[0])) < tol:
                    ch[:0] = list(p[:-1]); used[k] = grown = True
                elif np.hypot(*(p[0] - ch[0])) < tol:
                    ch[:0] = list(p[::-1][:-1]); used[k] = grown = True
                if grown:
                    break
        A = np.asarray(ch, float)
        closed = len(A) > 3 and np.hypot(*(A[0] - A[-1])) < tol
        if closed:
            A[-1] = A[0]
        chains.append((A, bool(closed)))
    return chains


# -------------------------------------------------------------
# KONTUR ONARIMI (açık / boşluklu / üst üste binmiş çizgiler, CMM taramaları)
#  1) Çizgiler birbirini kestiği ve üst üste bindiği yerlerden bölünür
#  2) Çok yakın uçlar tek düğümde birleştirilir, tekrar eden parçalar silinir (çakışmalar)
#  3) Açık uçlar en yakın açık uca bağlanır (boşluk ≤ tolerans; tolerans otomatik büyütülür)
#  4) Sarkan uçlar budanır, düzlemsel ağın DIŞ SINIRI ön form konturu olarak alınır
#  (iç bölme çizgileri, çift tarama pasları dış sınırı değiştirmez)
# -------------------------------------------------------------
REPAIR_SNAP = 0.02                           # bu mesafedeki noktalar aynı nokta sayılır (mm)
REPAIR_GAPS = (0.05, 0.2, 0.5, 1.0, 2.0)     # denenecek boşluk kapatma toleransları (mm)
_REPAIR_CACHE = {}


def _uf_find(par, i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i


def _snap_nodes(Q, tol):
    """Noktaları tol içinde kümele. Döndürür: (her nokta için düğüm no, düğüm koordinatları)"""
    n = len(Q)
    par = list(range(n))
    cell = {}
    key = np.floor(Q / tol).astype(np.int64)
    for i in range(n):
        kx, ky = int(key[i, 0]), int(key[i, 1])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cell.get((kx + dx, ky + dy), ()):
                    if (Q[i, 0] - Q[j, 0]) ** 2 + (Q[i, 1] - Q[j, 1]) ** 2 <= tol * tol:
                        a, b = _uf_find(par, i), _uf_find(par, j)
                        if a != b:
                            par[a] = b
        cell.setdefault((kx, ky), []).append(i)
    roots = np.array([_uf_find(par, i) for i in range(n)])
    uniq, inv = np.unique(roots, return_inverse=True)
    nodes = np.zeros((len(uniq), 2))
    np.add.at(nodes, inv, Q)
    nodes /= np.bincount(inv)[:, None]
    return inv, nodes


def _node_segments(P0, P1, tol, chunk=400):
    """Her parçayı diğerleriyle kesişim noktalarından ve üzerine düşen uçlardan böler."""
    m = len(P0)
    D = P1 - P0
    L2 = np.maximum((D ** 2).sum(1), 1e-18)
    lo, hi = np.minimum(P0, P1) - tol, np.maximum(P0, P1) + tol
    cuts = [[0.0, 1.0] for _ in range(m)]
    for a in range(0, m, chunk):
        b = min(a + chunk, m)
        ov = ((lo[a:b, None, 0] <= hi[None, :, 0]) & (hi[a:b, None, 0] >= lo[None, :, 0]) &
              (lo[a:b, None, 1] <= hi[None, :, 1]) & (hi[a:b, None, 1] >= lo[None, :, 1]))
        ov[np.arange(b - a), np.arange(a, b)] = False
        I, J = np.nonzero(ov)
        if not len(I):
            continue
        I = I + a
        p, r, q, sv = P0[I], D[I], P0[J], D[J]
        den = r[:, 0] * sv[:, 1] - r[:, 1] * sv[:, 0]
        w = q - p
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (w[:, 0] * sv[:, 1] - w[:, 1] * sv[:, 0]) / den
            u = (w[:, 0] * r[:, 1] - w[:, 1] * r[:, 0]) / den
        ok = (np.abs(den) > 1e-12) & (t > 1e-9) & (t < 1 - 1e-9) & (u >= -1e-9) & (u <= 1 + 1e-9)
        for i, tt in zip(I[ok], t[ok]):
            cuts[i].append(float(tt))
        for E in (P0[J], P1[J]):                     # diğer parçanın ucu bu parçanın üzerinde mi (T / çakışma)
            tt = ((E - p) * r).sum(1) / L2[I]
            dd = np.hypot(*(p + r * np.clip(tt, 0, 1)[:, None] - E).T)
            ok2 = (tt > 1e-9) & (tt < 1 - 1e-9) & (dd <= tol)
            for i, t2 in zip(I[ok2], tt[ok2]):
                cuts[i].append(float(t2))
    A, B = [], []
    for i in range(m):
        ts = np.unique(np.round(cuts[i], 12))
        pts = P0[i] + np.outer(ts, D[i])
        A.append(pts[:-1]); B.append(pts[1:])
    return np.vstack(A), np.vstack(B)


def _outer_boundary(nodes, edges):
    """Düzlemsel ağ: sarkan uçları buda, yüzleri dolaş, en büyük bileşenin dış sınırını döndür."""
    E = set()
    for u, v in edges:
        if u != v:
            E.add((min(u, v), max(u, v)))
    adj = collections.defaultdict(set)
    for u, v in E:
        adj[u].add(v); adj[v].add(u)
    stack = [k for k, nb in adj.items() if len(nb) == 1]
    while stack:                                      # sarkan uçları buda
        k = stack.pop()
        for j in list(adj.get(k, ())):
            adj[j].discard(k)
            if len(adj[j]) == 1:
                stack.append(j)
        adj.pop(k, None)
    adj = {k: v for k, v in adj.items() if v}
    if not adj:
        return None
    order = {}
    for k, nb in adj.items():
        nb = list(nb)
        ang = [np.arctan2(nodes[j, 1] - nodes[k, 1], nodes[j, 0] - nodes[k, 0]) for j in nb]
        order[k] = [nb[q] for q in np.argsort(ang)]
    pos = {(k, j): q for k, lst in order.items() for q, j in enumerate(lst)}
    seen, best, best_a = set(), None, 0.0
    for k, lst in order.items():
        for j in lst:
            if (k, j) in seen:
                continue
            cyc, h = [], (k, j)
            while h not in seen and len(cyc) <= 4 * len(pos):
                seen.add(h)
                u, v = h
                cyc.append(u)
                lv = order[v]
                h = (v, lv[(pos[(v, u)] - 1) % len(lv)])  # ikizinden saat yönünde bir sonraki
            C = nodes[cyc]
            a = 0.5 * float(np.sum(C[:, 0] * np.roll(C[:, 1], -1) - np.roll(C[:, 0], -1) * C[:, 1]))
            if a < best_a:                            # dış yüz: saat yönünde (negatif alan)
                best, best_a = cyc, a
    if best is None:
        return None
    C = nodes[best[::-1]]
    return np.vstack([C, C[:1]])


def repair_contour(chains):
    """Açık / boşluklu / üst üste binmiş çizgilerden tek bir kapalı kontur üretir.
    Döndürür: (zincirler, not). Onarım gerekmezse zincirler aynen döner, not boş olur."""
    if not chains:
        return chains, ''
    big = _largest_closed(chains)
    plen = lambda A: float(np.hypot(*np.diff(A, axis=0).T).sum())
    if big is not None:
        per = plen(big)
        open_len = sum(plen(A) for A, cl in chains if not cl)
        other = [A for A, cl in chains if cl and A is not big]
        ab = abs(poly_area(big))
        if open_len < 0.01 * per and all(abs(poly_area(A)) < 0.01 * ab for A in other):
            return chains, ''                         # temiz kontur: dokunma
    P0 = np.vstack([A[:-1] for A, _ in chains if len(A) >= 2])
    P1 = np.vstack([A[1:] for A, _ in chains if len(A) >= 2])
    keep = np.hypot(*(P1 - P0).T) > 1e-9
    P0, P1 = P0[keep], P1[keep]
    if len(P0) < 3:
        return chains, ''
    A, B = _node_segments(P0, P1, REPAIR_SNAP)
    inv, nodes = _snap_nodes(np.vstack([A, B]), REPAIR_SNAP)
    m = len(A)
    base = [(int(inv[i]), int(inv[m + i])) for i in range(m)]
    deg = collections.Counter()
    for u, v in set((min(a, b), max(a, b)) for a, b in base if a != b):
        deg[u] += 1; deg[v] += 1
    ends = np.array([k for k, d in deg.items() if d == 1], int)
    results = []
    for gap in REPAIR_GAPS:
        extra = []
        if len(ends) > 1:                             # açık uçları karşılıklı en yakın açık uca bağla
            Q = nodes[ends]
            d = np.hypot(Q[:, None, 0] - Q[None, :, 0], Q[:, None, 1] - Q[None, :, 1])
            np.fill_diagonal(d, np.inf)
            nn = np.argmin(d, axis=1)
            for i, j in enumerate(nn):
                if i < j and nn[j] == i and d[i, j] <= gap:
                    extra.append((int(ends[i]), int(ends[j])))
        poly = _outer_boundary(nodes, base + extra)
        results.append((gap, poly, abs(poly_area(poly)) if poly is not None else 0.0, len(extra)))
    amax = max(r[2] for r in results)
    if amax <= 0:
        return chains, ("UYARI: Ön form çizgilerinden kapalı kontur oluşturulamadı "
                        f"(en büyük boşluk {REPAIR_GAPS[-1]:.1f} mm'den fazla olabilir).")
    gap, poly, area, nj = next(r for r in results if r[2] >= 0.995 * amax)
    n_in = sum(1 for _ in chains)
    note = (f"Kontur onarıldı: {n_in} ayrı çizgi zinciri → tek kapalı kontur"
            + (f", {nj} boşluk kapatıldı (≤ {gap:g} mm)" if nj else "")
            + f", üst üste binen / tekrar eden parçalar ayıklandı. Kesit alanı {area:,.1f} mm².")
    return [(poly, True)], note


def poly_area(P):
    if P is None or len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


def preform_chains(state, selected, dx=0.0, dy=0.0, tol=0.01, with_note=False):
    """Seçili katmanlardaki çizgileri uç uca birleştirir; gerekirse konturu onarır
    (boşluk, çakışma, açık uç). Döndürür: [(noktalar (k,2), kapalı_mı), ...] (with_note: + not)"""
    try:
        key = (json.dumps(state.get('layers') if state else None, sort_keys=True),
               tuple(sorted(selected)) if selected else None, float(dx), float(dy), tol)
    except Exception:
        key = None
    if key is not None and key in _REPAIR_CACHE:
        ch, note = _REPAIR_CACHE[key]
    else:
        ch = _join_chains(state, selected, dx, dy, tol)
        try:
            ch, note = repair_contour(ch)
        except Exception as e:                       # onarım başarısızsa ham zincirlerle devam
            note = f"UYARI: Kontur onarımı yapılamadı ({e}); çizgiler olduğu gibi kullanıldı."
        if key is not None:
            if len(_REPAIR_CACHE) > 16:
                _REPAIR_CACHE.clear()
            _REPAIR_CACHE[key] = (ch, note)
    return (ch, note) if with_note else ch


def preform_thickness(M, Nrm, chains, gap_tol=PF_GAP_TOL, chunk=192):
    """Her kalıp ışını için ön formun dış yüzeyine olan mesafe (normal + ve - yönü).
    Kapalı konturlar: parite ile 'malzeme içi' aralığı; açık çizgiler: ilk kesişim.
    Döndürür: (Tp, Tm) - yoksa nan."""
    Nm = len(M)
    Tp = np.full(Nm, np.nan)
    Tm = np.full(Nm, np.nan)
    if Nm == 0 or not chains:
        return Tp, Tm
    closed = [c for c, cl in chains if cl]
    use = closed if closed else [c for c, cl in chains]
    A = np.vstack([c[:-1] for c in use])
    E = np.vstack([c[1:] - c[:-1] for c in use])
    eps = 1e-6
    for a in range(0, Nm, chunk):
        Mc, Nc = M[a:a + chunk], Nrm[a:a + chunk]
        den = Nc[:, None, 0] * E[None, :, 1] - Nc[:, None, 1] * E[None, :, 0]
        wx = A[None, :, 0] - Mc[:, None, 0]
        wy = A[None, :, 1] - Mc[:, None, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            d = (wx * E[None, :, 1] - wy * E[None, :, 0]) / den
            u = (wx * Nc[:, None, 1] - wy * Nc[:, None, 0]) / den
        ok = (np.abs(den) > 1e-12) & (u >= 0) & (u < 1)
        d = np.where(ok, d, np.nan)
        for sgn, T in ((1, Tp), (-1, Tm)):
            dd = np.sort(np.where(sgn * d > eps, sgn * d, np.inf), axis=1)
            cnt = np.isfinite(dd).sum(axis=1)
            d1 = dd[:, 0]
            d2 = dd[:, 1] if dd.shape[1] > 1 else np.full(len(dd), np.inf)
            if closed:
                inside = (cnt % 2) == 1
                res = np.where(inside, d1, np.where((cnt >= 2) & (d1 <= gap_tol), d2, np.nan))
            else:
                res = np.where(cnt >= 1, d1, np.nan)
            T[a:a + chunk] = np.where(np.isfinite(res), res, np.nan)
    if not closed:                                  # açık çizgi: uzak, alakasız kesişimleri at
        for T in (Tp, Tm):
            fin = np.isfinite(T)
            if fin.any():
                med = float(np.median(T[fin]))
                T[fin & (T > 3 * med + 5)] = np.nan
    return Tp, Tm


def _chunks_from_mask(ok, pts, brk):
    """ok maskesi doğru olan ardışık ışınları, sıçramada (brk) bölerek indeks listelerine ayırır."""
    out, cur = [], []
    for i in range(len(ok)):
        if not ok[i]:
            if len(cur) > 1:
                out.append(cur)
            cur = []
            continue
        if cur and np.hypot(*(pts[i] - pts[cur[-1]])) > brk:
            if len(cur) > 1:
                out.append(cur)
            cur = []
        cur.append(i)
    if len(cur) > 1:
        out.append(cur)
    return out


PF_MODEL_CHOICES = [
    ('Sac ön form (sıvama + inceltme): kalınlık = sacın kendi kalınlığı (DXF), malzeme sac boyunca akar', 'strip'),
    ('Shear forming (disk/koni ön form): sinüs kuralı  t = t₀·sin α', 'shear'),
    ('Kalıp üzerinde (flow forming): kalınlık = kalıptan ön form dış yüzeyine', 'flow'),
]
PF_RADIAL_CHOICES = [
    ('Otomatik (çap ayarından; yoksa Y)', 'auto'),
    ('Radyal eksen X (dönme ekseni Y yönünde)', 'X'),
    ('Radyal eksen Y (dönme ekseni X yönünde)', 'Y'),
]


def resolve_radial(choice, cfg):
    """Radyal eksen: seçim 'auto' ise kalıbın / kanalların çap ayarından bulunur, yoksa X."""
    if choice in ('X', 'Y'):
        return choice, 'seçildi'
    for d in [cfg.get('mold', {})] + list(cfg.get('channels', [])):
        if d.get('dia_mode') == 'diameter':
            ax = d.get('dia_axis', 'X')
            return ('Y' if ax == 'Y' else 'X'), 'çap ayarından'
    return 'Y', 'varsayılan (parça yarıçapı Y)'


def radial_guard(chains, radial):
    """Parça dönme ekseninin (radyal koordinat = 0) iki tarafına geçen ön form fiziksel olamaz.
    Seçilen eksende ön form 0'ı kesiyor, diğerinde kesmiyorsa diğer eksene geçilir.
    Döndürür: (radyal_eksen, uyarı_metni)"""
    pts = [A for A, _cl in chains if len(A)]
    if not pts:
        return radial, ''
    P = np.vstack(pts)
    cross = lambda k: float(P[:, k].min()) < -0.5 and float(P[:, k].max()) > 0.5
    k = 0 if radial == 'X' else 1
    if cross(k) and not cross(1 - k):
        other = 'Y' if radial == 'X' else 'X'
        return other, (f"UYARI: Ön form {radial}=0 çizgisinin iki tarafına geçiyor; {radial} parça yarıçapı olamaz. "
                       f"Parça yarıçapı ekseni otomatik olarak {other} alındı (ayarı '{other}' veya 'Otomatik' yap).")
    return radial, ''


def preform_axial_thickness(M, chains, rad_idx, chunk=256):
    """Shear forming için: her kalıp ışınının radyal koordinatında, dönme eksenine paralel
    doğrunun ön form (kapalı kontur) içinden geçen toplam uzunluğu = eksenel ön form kalınlığı t₀.
    Döndürür: t0 (N,) - o yarıçapta malzeme yoksa nan."""
    Nm = len(M)
    t0 = np.full(Nm, np.nan)
    closed = [c for c, cl in chains if cl]
    if Nm == 0 or not closed:
        return t0
    ax_idx = 1 - rad_idx
    A = np.vstack([c[:-1] for c in closed]); B = np.vstack([c[1:] for c in closed])
    ra, rb = A[:, rad_idx], B[:, rad_idx]
    za, zb = A[:, ax_idx], B[:, ax_idx]
    rho_all = M[:, rad_idx]
    for a in range(0, Nm, chunk):
        rho = rho_all[a:a + chunk][:, None]
        cross = ((ra[None] <= rho) & (rb[None] > rho)) | ((rb[None] <= rho) & (ra[None] > rho))
        with np.errstate(divide='ignore', invalid='ignore'):
            z = za[None] + (rho - ra[None]) / (rb - ra)[None] * (zb - za)[None]
        z = np.sort(np.where(cross, z, np.inf), axis=1)
        cnt = cross.sum(axis=1)
        tot = np.zeros(len(rho))
        for k in range(0, z.shape[1] - 1, 2):
            pair = (k + 1) < cnt
            if not pair.any():
                break
            tot += np.where(pair, z[:, k + 1] - z[:, k], 0.0)
        t0[a:a + chunk] = np.where(cnt >= 2, tot, np.nan)
    return t0


def _tool_side(S, n):
    """Her ışın için takımın bulunduğu taraf (+1/-1); takım yoksa genel çoğunluk tarafı."""
    Sa = np.vstack([np.where(np.isfinite(s), np.abs(s), np.inf) for s in S])
    kbest = np.argmin(Sa, axis=0)
    sbest = np.array([S[kbest[j]][j] for j in range(n)])
    loc = np.where(np.isfinite(sbest) & (sbest != 0), np.sign(sbest), 0)
    glob = 1 if loc.sum() >= 0 else -1
    return np.where(loc != 0, loc, glob).astype(int), sbest


def compute_form(timelines, radii, mold_points, limit, preform=None):
    """Radüs izi + (varsa) ön forma göre son form.
    preform: {'chains': [...], 'model': 'shear'|'flow', 'radial': 'X'|'Y'} veya None.
    Döndürür: {'ch': [kanal izleri], 'final': {...} veya None, 'pf_chains': [...]}"""
    M, Nrm, fan = mold_rays(mold_points)
    S = []
    for i in range(N_CH):
        C, _T = tool_samples(timelines[i], radii[i])
        S.append(radius_envelope(M, Nrm, C, radii[i]))
    pf = preform or {}
    chains = pf.get('chains') or []
    if not chains or len(M) == 0:
        ch = []
        for i in range(N_CH):
            lim = limit if limit > 0 else radii[i]
            ch.append({'chunks': envelope_chunks(M, Nrm, S[i], lim), 'limit': lim})
        return {'ch': ch, 'final': None, 'pf_chains': []}
    seg = np.r_[0.0, np.hypot(*(M[1:] - M[:-1]).T)]
    arc = np.cumsum(seg)
    model = pf.get('model', 'strip')
    if model == 'strip':
        out = _form_strip(M, Nrm, fan, S, chains, pf.get('radial', 'Y'), arc, timelines, radii,
                          bool(pf.get('sim')))
        return out
    if model == 'shear':
        out = _form_shear(M, Nrm, fan, S, chains, pf.get('radial', 'Y'), arc)
    else:
        out = _form_flow(M, Nrm, fan, S, chains, arc)
        out['final']['radial'] = pf.get('radial', 'Y')
    fin = out['final']
    fin['ds'] = np.r_[0.0, np.diff(arc)]
    if pf.get('sim'):
        sim = sim_setup(fin, chains)
        fin['sim'] = sim
        if sim.get('ok'):
            sim['W0'] = sim_w0(fin, sim)
            side_t, _sb = _tool_side(S, len(M))
            sim['side'] = np.where(fin['side'] != 0, fin['side'], side_t).astype(int)
            sim['lim'] = sim_limit(M, Nrm, chains, sim) if model == 'flow' else np.full(len(M), sim['tmax'])
            sim['tR'] = sim_reach_times(M, arc, timelines, radii) if model == 'flow' else None
            sim['Rb'] = float(np.clip(sim['lmed'], 3.0, 60.0))   # kuyruğun en küçük bükülme yarıçapı
            Fs, fs = sim_formed(S, sim['side'], sim['lim'], sim['tR'])
            sim['F'], sim['formed'] = Fs, fs
            sim['final'] = sim_state(fin, sim, Fs, fs)
    return out


# -------------------------------------------------------------
# SAC ÖN FORM MODELİ (sıvama + inceltme)
#  Ön form ince bir sactır: kapalı konturdan orta hat + lokal kalınlık çıkarılır (yalnız DXF'ten).
#  Malzeme koordinatı = sac orta hattı boyunca yay uzunluğu σ. Makaranın geçtiği yerde sac kalıba
#  sıvanır, kalınlık = min(takım boşluğu, sac kalınlığı). Her pas (ışınların ilk şekillendiği zamanın
#  sırasına göre yönü bulunur) kendi başlangıç kesitinden malzeme tüketir; kalan sac, o pasın ucuna
#  hacim korunarak taşınır (uçta uzama). Paslar arasındaki dokunulmamış sac yerinde kalır.
# -------------------------------------------------------------
STRIP_TOL = 1.0          # takım boşluğu ≤ (en büyük sac kalınlığı + bu pay) ise makara saca iş yapıyor (mm)


def _densify_closed(P, step):
    out = []
    for a_, b_ in zip(P[:-1], P[1:]):
        L = float(np.hypot(*(b_ - a_)))
        n = max(int(np.ceil(L / step)), 1)
        for k in range(n):
            out.append(a_ + (b_ - a_) * k / n)
    out.append(P[-1])
    return np.asarray(out)


def _resample(P, n):
    d = np.r_[0.0, np.cumsum(np.hypot(*(P[1:] - P[:-1]).T))]
    u = np.linspace(0, d[-1], n)
    return np.c_[np.interp(u, d, P[:, 0]), np.interp(u, d, P[:, 1])]


def _project_on_polyline(Q, P):
    """Q noktalarının P çoklu çizgisi üzerindeki en yakın noktası ve P boyunca parametresi."""
    A, AB = P[:-1], P[1:] - P[:-1]
    L2 = np.maximum((AB ** 2).sum(1), 1e-18)
    segl = np.sqrt(L2)
    cum = np.r_[0.0, np.cumsum(segl)]
    out = np.empty_like(Q); par = np.empty(len(Q))
    for a_ in range(0, len(Q), 256):
        q = Q[a_:a_ + 256]
        t = np.clip(((q[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
        pr = A[None] + t[..., None] * AB[None]
        d2 = ((pr - q[:, None, :]) ** 2).sum(2)
        k = d2.argmin(1); r = np.arange(len(q))
        out[a_:a_ + 256] = pr[r, k]
        par[a_:a_ + 256] = cum[k] + t[r, k] * segl[k]
    return out, par


def strip_from_polygon(poly, ri, n=900):
    """Kapalı ince sac konturundan: iki yüzey (A, B), orta hat, kalınlık ve kümülatif hacim tablosu."""
    P = np.asarray(poly, float)
    if not np.allclose(P[0], P[-1]):
        P = np.vstack([P, P[:1]])
    per = float(np.hypot(*(P[1:] - P[:-1]).T).sum())
    D = _densify_closed(P, max(per / 4000.0, 0.05))[:-1]
    sub = D[::max(len(D) // 600, 1)]
    dd = ((sub[:, None, :] - sub[None]) ** 2).sum(2)
    i, j = np.unravel_index(int(dd.argmax()), dd.shape)
    ia = int(np.argmin(((D - sub[i]) ** 2).sum(1))); ib = int(np.argmin(((D - sub[j]) ** 2).sum(1)))
    if ia > ib:
        ia, ib = ib, ia
    S1 = D[ia:ib + 1]
    S2 = np.vstack([D[ib:], D[:ia + 1]])[::-1]          # iki yüzey de a ucundan b ucuna
    base, other = (S1, S2) if len(S1) >= len(S2) else (S2, S1)
    pA = _resample(base, n)
    pB, _par = _project_on_polyline(pA, other)
    mid = 0.5 * (pA + pB)
    t = np.hypot(*(pA - pB).T)
    sig = np.r_[0.0, np.cumsum(np.hypot(*(mid[1:] - mid[:-1]).T))]
    rmid = np.abs(mid[:, ri])
    dens = 2 * np.pi * rmid * t
    V = np.r_[0.0, np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(sig))]
    poly_s = np.vstack([pA, pB[::-1], pA[:1]])
    Vp = poly_volume(poly_s, ri)
    if V[-1] <= 0 or Vp <= 0:
        return {'ok': False, 'why': 'Sac kalınlığı / hacmi hesaplanamadı.'}
    V = V * (Vp / V[-1])                                   # toplamı poligon hacmine eşitle
    return {'ok': True, 'sig': sig, 'pA': pA, 'pB': pB, 'mid': mid, 't': t, 'V': V, 'Vtot': float(Vp),
            'poly': poly_s, 'ri': ri, 'tmax': float(np.percentile(t, 98)), 'tmed': float(np.median(t))}


def _flip_strip(strip):
    """Sac orta hattının yönünü ters çevirir (σ=0 diğer uca geçer); hacim tablosu yeniden kurulur."""
    st = dict(strip)
    for k in ('pA', 'pB', 'mid', 't'):
        st[k] = strip[k][::-1].copy()
    st['sig'] = np.r_[0.0, np.cumsum(np.hypot(*(st['mid'][1:] - st['mid'][:-1]).T))]
    st['V'] = strip['Vtot'] - strip['V'][::-1]
    st['poly'] = np.vstack([st['pA'], st['pB'][::-1], st['pA'][:1]])
    return st


def strip_ray_map(strip, M, N):
    """Her kalıp ışınının sac orta hattını kestiği yerin σ değeri (kesmezse en yakın nokta)."""
    mid, sig = strip['mid'], strip['sig']
    A, E = mid[:-1], mid[1:] - mid[:-1]
    segl = np.hypot(*E.T)
    out = np.empty(len(M)); hits = np.zeros(len(M), bool)
    for a_ in range(0, len(M), 256):
        Mc, Nc = M[a_:a_ + 256], N[a_:a_ + 256]
        den = Nc[:, None, 0] * E[None, :, 1] - Nc[:, None, 1] * E[None, :, 0]
        wx = A[None, :, 0] - Mc[:, None, 0]; wy = A[None, :, 1] - Mc[:, None, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            d = (wx * E[None, :, 1] - wy * E[None, :, 0]) / den
            u = (wx * Nc[:, None, 1] - wy * Nc[:, None, 0]) / den
        ok = (np.abs(den) > 1e-12) & (u >= 0) & (u <= 1)
        dabs = np.where(ok, np.abs(d), np.inf)
        k = dabs.argmin(1); r = np.arange(len(Mc))
        hit = np.isfinite(dabs[r, k])
        sh = sig[k] + np.clip(u[r, k], 0, 1) * segl[k]
        _pr, par = _project_on_polyline(Mc, mid)
        out[a_:a_ + 256] = np.where(hit, sh, par)
        hits[a_:a_ + 256] = hit
    return out, hits


def first_formed_times(timelines, radii, M, N, side, thr):
    """Her ışının ilk şekillendiği zaman (takım boşluğu ≤ thr, aynı tarafta). Hiç olmadıysa inf."""
    tau = np.full(len(M), np.inf)
    for i in range(N_CH):
        C, T = tool_samples(timelines[i], radii[i])
        if len(C) == 0:
            continue
        poly = tool_shape(radii[i]) is not None
        step = 256 if not poly else 12
        reach = tool_reach(radii[i]) + thr
        for a_ in range(0, len(C), step):
            Cc, Tc = C[a_:a_ + step], T[a_:a_ + step]
            if poly:                                  # yalnız takıma yakın ışınlar
                lo, hi = Cc.min(0) - reach, Cc.max(0) + reach
                rid = np.where((M[:, 0] >= lo[0]) & (M[:, 0] <= hi[0]) & (M[:, 1] >= lo[1]) & (M[:, 1] <= hi[1]))[0]
                if not len(rid):
                    continue
            else:
                rid = np.arange(len(M))
            s_ = tool_ray_matrix(M[rid], N[rid], Cc, radii[i])
            okm = np.isfinite(s_) & ((s_ == 0) | (np.sign(s_) == side[rid, None])) & (np.abs(s_) <= thr)
            anyk = okm.any(1)
            first = np.where(anyk, Tc[np.argmax(okm, axis=1)], np.inf)
            tau[rid] = np.minimum(tau[rid], first)
    return tau


def strip_groups(tau, arc, gap_mm=6.0):
    """Şekillenen ışınları ardışık gruplara ayırır, her grubun ilerleme yönünü (+1/-1 ışın sırası)
    ilk şekillenme zamanından bulur; aynı yönlü gruplar birleşir. Döndürür: gdir (N,) 0/+1/-1"""
    f = np.isfinite(tau)
    gdir = np.zeros(len(tau), int)
    idx = np.where(f)[0]
    if len(idx) == 0:
        return gdir
    runs, cur = [], [idx[0]]
    for k in idx[1:]:
        if arc[k] - arc[cur[-1]] > gap_mm:
            runs.append(cur); cur = []
        cur.append(k)
    runs.append(cur)
    for r in runs:
        r = np.asarray(r)
        x, y = arc[r], tau[r]
        slope = np.polyfit(x, y, 1)[0] if len(r) > 2 and np.ptp(x) > 0 else 0.0
        gdir[r] = 1 if slope >= 0 else -1
    return gdir


def _strip_section(strip, sg):
    sig = strip['sig']
    pa = np.array([np.interp(sg, sig, strip['pA'][:, 0]), np.interp(sg, sig, strip['pA'][:, 1])])
    pb = np.array([np.interp(sg, sig, strip['pB'][:, 0]), np.interp(sg, sig, strip['pB'][:, 1])])
    return pa, pb


def _strip_poly(strip, s0, s1):
    """σ ∈ [s0, s1] arasındaki sac parçasının poligonu."""
    lo, hi = min(s0, s1), max(s0, s1)
    sig = strip['sig']
    if hi - lo < 1e-9:
        return None
    m = (sig > lo) & (sig < hi)
    a0, b0 = _strip_section(strip, lo); a1, b1 = _strip_section(strip, hi)
    A = np.vstack([a0, strip['pA'][m], a1]); B = np.vstack([b0, strip['pB'][m], b1])
    return np.vstack([A, B[::-1], A[:1]])


def strip_runs(formed, arc, gap_mm=6.0):
    """Kalıba sıvanmış ardışık bölgeler (ışın sırası): aralarındaki boşluk gap_mm'den kısaysa birleşir."""
    idx = np.where(formed)[0]
    if len(idx) == 0:
        return []
    runs, a_ = [], idx[0]
    for p_, q_ in zip(idx[:-1], idx[1:]):
        if arc[q_] - arc[p_] > gap_mm:
            runs.append((int(a_), int(p_))); a_ = q_
    runs.append((int(a_), int(idx[-1])))
    return runs


def _mold_signed(P, M, N, sd):
    """P noktalarının kalıba işaretli mesafesi (+ malzeme tarafı, − kalıbın içi), en yakın ışına göre."""
    out = np.empty(len(P))
    for a_ in range(0, len(P), 256):
        q = P[a_:a_ + 256]
        d2 = ((q[:, None, :] - M[None]) ** 2).sum(2)
        k = d2.argmin(1)
        out[a_:a_ + 256] = ((q - M[k]) * (N[k] * sd[k][:, None])).sum(1)
    return out


def _attach_piece(poly, ri, src, tgt, M, N, sd, iters=4):
    """Uç parçasını hacmi koruyarak (remap) tgt'ye bağlar; kalıba gömülürse yarıçap yönünde kaldırır."""
    if poly is None:
        return None, tgt
    tg = np.array(tgt, float)
    sub = poly[::max(len(poly) // 160, 1)]
    for _ in range(iters):
        Q = remap_volume(sub, ri, src, tg)
        pen = float(np.max(-_mold_signed(Q, M, N, sd)))
        if pen <= 0.01:
            break
        tg[ri] += (pen + 0.02) * (1.0 if tg[ri] >= 0 else -1.0)
    return remap_volume(poly, ri, src, tg), tg


def strip_state(fin, sim, U, formed):
    """Sac simülasyonu (hacim sabit), verilen anda.
    Kalıba sıvanmış her bölgenin SABİT NOKTASI = o bölgede ilk şekillenen ışın (malzeme orada kımıldamaz).
    Bölge iki yana büyür: sol kısmın hacmi sacın sabit noktanın solundan, sağ kısmınki sağından tüketilir.
    Uçlardaki serbest sac bölgenin ucuna hacim korunarak taşınır (uçta uzama / kısalma; kalıba gömülmez);
    iki bölge arasında kalan sac yerinde kalır."""
    M, N = fin['M'], fin['N']
    sd = np.asarray(sim['side'], float)
    strip, ri = sim['strip'], sim['strip']['ri']
    sig, V, Vt = strip['sig'], strip['V'], strip['Vtot']
    sigR, tau, W0, dsc, arc = sim['sigR'], sim['tau'], sim['W0'], sim['dsc'], fin['profile']['arc']
    out = {'polys': [], 'walls': [], 'pieces': [], 'Vwall': 0.0, 'exhausted': False, 'runs': [], 'elong': {}}
    runs = strip_runs(formed, arc)
    if not runs:
        out['pieces'] = [strip['poly']]
        out['polys'] = [strip['poly']]
        out['Vpieces'] = poly_volume(strip['poly'], ri)
        return out
    info = []
    for (a_, b_) in runs:
        idx = np.arange(a_, b_ + 1)
        fm = formed[idx]
        w = np.clip(np.where(fm, np.minimum(np.nan_to_num(U[idx], nan=0.0), W0[idx]), W0[idx]), 0, None)
        rmid = np.abs((M[idx] + N[idx] * (sd[idx] * w / 2)[:, None])[:, ri])
        dV = 2 * np.pi * rmid * w * dsc[idx]
        tf = np.where(fm, tau[idx], np.inf)
        ties = idx[tf <= tf.min() + 1e-6]               # aynı anda ilk temas eden ışınlar: ortadakini al
        n = int(ties[len(ties) // 2])
        sn = float(sigR[n]); Vn = float(np.interp(sn, sig, V))
        L = idx < n
        VL, VR = float(dV[L].sum()), float(dV[~L].sum())
        lo, hi = a_, b_
        exL = exR = False
        if VL > Vn:                                    # sol tarafta malzeme bitti
            cl = np.cumsum(dV[L][::-1]); k = int(np.searchsorted(cl, Vn))
            lo = max(a_, n - 1 - k); VL = Vn; exL = True
        if VR > Vt - Vn:                               # sağ tarafta malzeme bitti
            cr = np.cumsum(dV[~L]); k = int(np.searchsorted(cr, Vt - Vn))
            hi = min(b_, n + k); VR = Vt - Vn; exR = True
        keep = (idx >= lo) & (idx <= hi)
        idk, wk = idx[keep], w[keep]
        Pout = M[idk] + N[idk] * (sd[idk] * wk)[:, None]
        wall = np.vstack([Pout, M[idk][::-1], Pout[:1]])
        sa = float(np.interp(Vn - VL, V, sig)); sb = float(np.interp(Vn + VR, V, sig))
        info.append({'lo': int(lo), 'hi': int(hi), 'n': n, 'sa': sa, 'sb': sb, 'sn': sn, 'Pout': Pout, 'exL': exL, 'exR': exR,
                     'eL': float(arc[n] - arc[lo]) - (sn - sa), 'eR': float(arc[hi] - arc[n]) - (sb - sn)})
        out['walls'].append(wall); out['Vwall'] += VL + VR
        out['exhausted'] |= exL or exR
    pieces = []

    def section_io(sg, k):
        pa, pb = _strip_section(strip, sg)
        nrm = N[k] * sd[k]
        return (pa, pb) if np.dot(pa, nrm) <= np.dot(pb, nrm) else (pb, pa)

    def bridge(k, Pout_k, inner, outer):
        return np.array([M[k], Pout_k, outer, inner, M[k]])

    # sol uç parçası: [0, sa_ilk] -> ilk bölgenin sol ucuna
    f0 = info[0]
    if not f0['exL'] and f0['sa'] > 1e-6:
        inner, outer = section_io(f0['sa'], f0['lo'])
        pc = _strip_poly(strip, 0.0, f0['sa'])
        pc2, tg = _attach_piece(pc, ri, inner, M[f0['lo']], M, N, sd)
        if pc2 is not None:
            pieces.append(pc2)
            out['polys'].append(bridge(f0['lo'], f0['Pout'][0], remap_volume(inner, ri, inner, tg),
                                       remap_volume(outer, ri, inner, tg)))
    # aradaki parçalar yerinde kalır
    for f1, f2 in zip(info[:-1], info[1:]):
        if f2['sa'] - f1['sb'] > 1e-6:
            pc = _strip_poly(strip, f1['sb'], f2['sa'])
            if pc is not None:
                pieces.append(pc)
                i1, o1 = section_io(f1['sb'], f1['hi']); i2, o2 = section_io(f2['sa'], f2['lo'])
                out['polys'] += [bridge(f1['hi'], f1['Pout'][-1], i1, o1), bridge(f2['lo'], f2['Pout'][0], i2, o2)]
    # sağ uç parçası: [sb_son, sonu] -> son bölgenin sağ ucuna
    fl = info[-1]
    if not fl['exR'] and sig[-1] - fl['sb'] > 1e-6:
        inner, outer = section_io(fl['sb'], fl['hi'])
        pc = _strip_poly(strip, fl['sb'], sig[-1])
        pc2, tg = _attach_piece(pc, ri, inner, M[fl['hi']], M, N, sd)
        if pc2 is not None:
            pieces.append(pc2)
            out['polys'].append(bridge(fl['hi'], fl['Pout'][-1], remap_volume(inner, ri, inner, tg),
                                       remap_volume(outer, ri, inner, tg)))
    out['pieces'] = pieces
    out['polys'] = out['walls'] + out['polys'] + pieces
    out['Vpieces'] = float(sum(poly_volume(P_, ri) for P_ in pieces))
    ax = 1 - ri
    an = 'X' if ax == 0 else 'Y'
    up = M[-1, ax] >= M[0, ax]                          # ışın sırası eksen boyunca artıyor mu
    out['elong'] = {f"{an}{'−' if up else '+'} ucu": info[0]['eL'], f"{an}{'+' if up else '−'} ucu": info[-1]['eR']}
    out['runs'] = [(f['lo'], f['hi'], f['n']) for f in info]
    return out


def _form_strip(M, Nrm, fan, S, chains, radial, arc, timelines, radii, sim_on):
    rad = radial if radial in ('X', 'Y') else 'Y'
    ri = 0 if rad == 'X' else 1
    poly = _largest_closed(chains)
    side, _sb = _tool_side(S, len(M))
    if poly is None:
        fin = {'mode': 'strip', 'chunks': [], 'regions': [], 'lines': [], 'M': M, 'N': Nrm, 'side': side,
               'T': np.full(len(M), np.nan), 'Ti': None, 'F': np.full(len(M), np.nan),
               'formed': np.zeros(len(M), bool), 'W0': np.zeros(len(M)), 'radial': rad,
               'profile': {'arc': arc, 'x': M[:, 0], 'y': M[:, 1], 'T0': np.full(len(M), np.nan),
                           'T': np.full(len(M), np.nan), 'F': np.full(len(M), np.nan),
                           'formed': np.zeros(len(M), bool), 'main': ~fan, 'alpha': None},
               'ds': np.r_[0.0, np.diff(arc)],
               'sim': {'ok': False, 'why': 'Sac ön form için DXF kapalı kontur olmalı.'} if sim_on else None}
        ch = [{'chunks': [], 'limit': 1.0} for _ in range(N_CH)]
        return {'ch': ch, 'final': fin, 'pf_chains': chains}
    strip = strip_from_polygon(poly, ri)
    sigR, hit = strip_ray_map(strip, M, Nrm)
    if strip.get('ok') and len(sigR) > 1 and sigR[-1] < sigR[0]:     # sacı ışın sırasıyla aynı yöne çevir
        strip = _flip_strip(strip)
        sigR, hit = strip_ray_map(strip, M, Nrm)
    W0 = np.interp(sigR, strip['sig'], strip['t'])
    thr = strip['tmax'] + STRIP_TOL
    U = np.full(len(M), np.inf)
    ch = []
    brk = max(2.0, strip['tmed'])
    for i in range(N_CH):
        s_ = S[i]
        same = np.isfinite(s_) & ((s_ == 0) | (np.sign(s_) == side))
        okc = same & (np.abs(s_) <= thr)
        U = np.minimum(U, np.where(okc, np.abs(s_), np.inf))
        ch.append({'chunks': envelope_chunks(M, Nrm, s_, thr, ok=okc), 'limit': thr})
    formed = np.isfinite(U)
    F = np.where(formed, np.minimum(U, W0), np.nan)
    sd = side.astype(float)
    P_act = M + Nrm * (sd * np.nan_to_num(F))[:, None]
    P_id = M + Nrm * (sd * W0)[:, None]
    thin = formed & (F < W0 - 0.01)
    ds = np.r_[0.0, np.diff(arc)]
    fin = {
        'mode': 'strip',
        'chunks': [(P_act[idx], M[idx]) for idx in _chunks_from_mask(formed, P_act, brk)],
        'line_label': 'Kalıba sıvanan sac yüzeyi (takım boşluğu)',
        'regions': [('İnceltme (sac kalınlığından ince)', DISP_COLOR,
                     [(P_id[idx], P_act[idx]) for idx in _chunks_from_mask(thin, P_act, brk)])],
        'lines': [('Sac kalınlığı kalıp üzerinde (inceltmesiz)', IDEAL_COLOR,
                   [P_id[idx] for idx in _chunks_from_mask(formed, P_id, brk)])],
        'profile': {'arc': arc, 'x': M[:, 0], 'y': M[:, 1], 'T0': np.where(hit | formed, W0, np.nan),
                    'T': np.where(hit | formed, W0, np.nan), 'F': F, 'formed': formed, 'main': ~fan, 'alpha': None},
        'M': M, 'N': Nrm, 'side': side, 'T': W0, 'Ti': W0, 'F': F, 'formed': formed, 'W0': W0, 'thr': thr,
        'radial': rad, 'ds': ds, 'strip': strip,
    }
    if sim_on:
        tau = first_formed_times(timelines, radii, M, Nrm, side, thr)
        dsc = 0.5 * (np.r_[arc[1:], arc[-1]] - np.r_[arc[0], arc[:-1]])
        sim = {'ok': True, 'kind': 'strip', 'strip': strip, 'sigR': sigR, 'W0': W0, 'side': side, 'tau': tau,
               'dsc': dsc, 'thr': thr, 'Vtot': strip['Vtot'], 'ri': ri}
        sim['final'] = strip_state(fin, sim, U, formed)
        fin['sim'] = sim
    return {'ch': ch, 'final': fin, 'pf_chains': chains}


def _form_shear(M, Nrm, fan, S, chains, radial, arc):
    """SHEAR FORMING (sinüs kuralı): malzeme aynı yarıçapta kalarak eksen yönünde kalıba sıvanır.
    İdeal son et kalınlığı (kalıba dik) = t₀(r) · sin α, α = kalıp yüzeyinin dönme ekseniyle açısı.
    Gerçek et kalınlığı = takım boşluğu (kalıptan takım izine). Fark: aşırı / eksik inceltme."""
    rad_idx = 0 if radial == 'X' else 1
    axis = np.array([0.0, 1.0]) if radial == 'X' else np.array([1.0, 0.0])
    t0 = preform_axial_thickness(M, chains, rad_idx)
    sin_a = np.abs(Nrm @ axis)
    Ti = t0 * sin_a
    has = np.isfinite(t0) & (t0 > 1e-9)
    side, _sb = _tool_side(S, len(M))
    sd = side.astype(float)
    U = np.full(len(M), np.inf)
    ch = []
    ref = float(np.median(t0[has])) if has.any() else 5.0
    brk = max(2.0, 0.5 * ref)
    for i in range(N_CH):
        s = S[i]
        same = np.isfinite(s) & ((s == 0) | (np.sign(s) == side))
        okc = has & same & (np.abs(s) <= t0 + 1e-9)        # takım boşluğu ön form kalınlığından az -> şekillendiriyor
        U = np.minimum(U, np.where(okc, np.abs(s), np.inf))
        ch.append({'chunks': envelope_chunks(M, Nrm, s, ref, ok=okc), 'limit': ref})
    formed = np.isfinite(U)
    F = np.where(formed, U, np.nan)
    P_act = M + Nrm * (sd * np.nan_to_num(F))[:, None]
    P_id = M + Nrm * (sd * np.nan_to_num(Ti))[:, None]
    tol = 0.01
    over = formed & (F < Ti - tol)
    under = formed & (F > Ti + tol)
    final = {
        'mode': 'shear',
        'chunks': [(P_act[idx], M[idx]) for idx in _chunks_from_mask(formed, P_act, brk)],
        'line_label': 'Gerçek son yüzey (takım boşluğu)',
        'regions': [('Aşırı inceltme (sinüs kuralından ince)', DISP_COLOR,
                     [(P_id[idx], P_act[idx]) for idx in _chunks_from_mask(over, P_act, brk)]),
                    ('Eksik inceltme / malzeme fazlası', UNDER_COLOR,
                     [(P_act[idx], P_id[idx]) for idx in _chunks_from_mask(under, P_act, brk)])],
        'lines': [('İdeal son yüzey (sinüs kuralı t₀·sin α)', IDEAL_COLOR,
                   [P_id[idx] for idx in _chunks_from_mask(has & (Ti > 1e-6), P_id, brk)])],
        'profile': {'arc': arc, 'x': M[:, 0], 'y': M[:, 1], 'T0': np.where(has, t0, np.nan),
                    'T': np.where(has, Ti, np.nan), 'F': F, 'formed': formed, 'main': ~fan,
                    'alpha': np.degrees(np.arcsin(np.clip(sin_a, 0, 1)))},
        'M': M, 'N': Nrm, 'side': side, 'T': np.where(has, t0, np.nan), 'Ti': np.where(has, Ti, np.nan),
        'radial': radial, 'F': F, 'formed': formed, 'W0': np.where(has, np.nan_to_num(Ti), 0.0),
    }
    return {'ch': ch, 'final': final, 'pf_chains': chains}


def _form_flow(M, Nrm, fan, S, chains, arc):
    """KALIP ÜZERİNDE (flow forming): ön form kalıbın üzerinde; kalınlık = kalıptan ön form dış yüzeyine.
    Son yüzey = min(ön form, takım izi)."""
    Tp, Tm = preform_thickness(M, Nrm, chains)
    hp, hm = np.isfinite(Tp), np.isfinite(Tm)
    side_t, sbest = _tool_side(S, len(M))
    tool_neg = np.isfinite(sbest) & (sbest < 0)
    side = np.where(hp & ~hm, 1, np.where(hm & ~hp, -1, np.where(hp & hm, np.where(tool_neg, -1, 1), 0)))
    T = np.where(side > 0, Tp, np.where(side < 0, Tm, np.nan))
    has = np.isfinite(T)
    U = np.full(len(M), np.inf)
    ch = []
    med_T = float(np.median(T[has])) if has.any() else 5.0
    brk = max(2.0, 0.5 * med_T)
    for i in range(N_CH):
        s = S[i]
        same = np.isfinite(s) & ((s == 0) | (np.sign(s) == side))
        u = np.where(same & has, np.abs(s), np.inf)
        U = np.minimum(U, u)
        okc = has & (u < T - 1e-6)
        ch.append({'chunks': envelope_chunks(M, Nrm, s, med_T, ok=okc), 'limit': med_T})
    F = np.where(has, np.minimum(T, U), np.nan)
    formed = has & (U < T - 1e-6)
    sd = side.astype(float)
    P_out = M + Nrm * (sd * np.where(has, T, 0.0))[:, None]
    P_fin = M + Nrm * (sd * np.where(has, F, 0.0))[:, None]
    final = {
        'mode': 'flow',
        'chunks': [(P_fin[idx], M[idx]) for idx in _chunks_from_mask(has, P_fin, brk)],
        'line_label': 'Son form yüzeyi (tüm kanallar)',
        'regions': [('Akıtılan / ezilen malzeme', DISP_COLOR,
                     [(P_out[idx], P_fin[idx]) for idx in _chunks_from_mask(formed, P_fin, brk)])],
        'lines': [],
        'profile': {'arc': arc, 'x': M[:, 0], 'y': M[:, 1], 'T0': T, 'T': T, 'F': F,
                    'formed': formed, 'main': ~fan, 'alpha': None},
        'M': M, 'N': Nrm, 'side': side, 'T': T, 'Ti': None,
        'F': np.where(formed, F, np.nan), 'formed': formed, 'W0': np.where(has, np.nan_to_num(T), 0.0),
    }
    return {'ch': ch, 'final': final, 'pf_chains': chains}


# -------------------------------------------------------------
# DİNAMİK FORM SİMÜLASYONU (hacim sabit)
#  Ön form, sabitlenen uçtan serbest uca sıralı bir "hacim bütçesi"dir (malzeme koordinatı):
#    shear forming : malzeme yarıçapını korur  -> koordinat = yarıçap  (kesit: eksene paralel doğru)
#    flow forming  : malzeme eksen boyunca dizili -> koordinat = eksenel konum (kesit: radyal doğru)
#  Her an: kalıp başından makaranın ulaştığı en uzak noktaya kadar malzeme kalıba sıvanır
#  (kalınlık = makara geçtiyse takım boşluğu, geçmediyse doğal kalınlık). Bu duvarın hacmi
#  ön formun başından tüketilir; kalan ön form şeklini koruyarak duvarın ucuna taşınır.
#  İnceltmeden artan hacim böylece kalan malzemeyi ileri iter = uçta uzama. Hacim: 2π·r·kalınlık·uzunluk.
# -------------------------------------------------------------
def _largest_closed(chains):
    best, ba = None, 0.0
    for A, cl in chains:
        if not cl:
            continue
        x, y = A[:, 0], A[:, 1]
        ar = abs(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        if ar > ba:
            best, ba = A, ar
    return best


def _section(poly, k, c):
    """Kapalı poligonun 'koordinat k = c' doğrusu ile kesişimleri (diğer koordinat, sıralı)."""
    A, B = poly[:-1], poly[1:]
    a, b = A[:, k], B[:, k]
    m = ((a <= c) & (b > c)) | ((b <= c) & (a > c))
    if not m.any():
        return np.zeros(0)
    o = 1 - k
    t = (c - a[m]) / (b[m] - a[m])
    return np.sort(A[m, o] + t * (B[m, o] - A[m, o]))


def clip_halfplane(poly, k, c, d):
    """Sutherland–Hodgman: d·(p[k] − c) ≥ 0 olan kısmı bırakır. Kapalı poligon döndürür (veya None)."""
    P = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    out = []
    n = len(P)
    for i in range(n):
        cur, prev = P[i], P[i - 1]
        cin, pin = d * (cur[k] - c) >= 0, d * (prev[k] - c) >= 0
        if cin != pin:
            t = (c - prev[k]) / (cur[k] - prev[k])
            out.append(prev + t * (cur - prev))
        if cin:
            out.append(cur)
    if len(out) < 3:
        return None
    out.append(out[0])
    return np.asarray(out)


def poly_volume(poly, ri):
    """Eksenel simetrik hacim (Pappus): 2π · alan · ağırlık merkezinin yarıçapı."""
    if poly is None or len(poly) < 4:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    cr = x[:-1] * y[1:] - x[1:] * y[:-1]
    A = 0.5 * cr.sum()
    if abs(A) < 1e-12:
        return 0.0
    cx = ((x[:-1] + x[1:]) * cr).sum() / (6 * A)
    cy = ((y[:-1] + y[1:]) * cr).sum() / (6 * A)
    return float(2 * np.pi * abs(A) * abs(cy if ri == 1 else cx))


def sim_setup(fin, chains, n=1200):
    """Malzeme tablosu: koordinat c -> kümülatif hacim V(c) ve kesitin iki ucu."""
    poly = _largest_closed(chains)
    if poly is None:
        return {'ok': False, 'why': 'Simülasyon için ön form kapalı kontur olmalı.'}
    ri = 0 if fin.get('radial', 'Y') == 'X' else 1
    shear = fin.get('mode') == 'shear'
    k = ri if shear else 1 - ri
    M = fin['M']
    if shear:
        dsgn = 1 if abs(M[-1, ri]) >= abs(M[0, ri]) else -1
    else:
        dsgn = 1 if M[-1, k] >= M[0, k] else -1
    lo, hi = float(poly[:, k].min()), float(poly[:, k].max())
    cs = np.linspace(lo, hi, n) if dsgn > 0 else np.linspace(hi, lo, n)
    dens = np.zeros(n)
    Ls = np.zeros(n)
    elo = np.full((n, 2), np.nan)
    ehi = np.full((n, 2), np.nan)
    for j, c in enumerate(cs):
        z = _section(poly, k, c)
        if len(z) < 2:
            continue
        zz = z[:len(z) // 2 * 2].reshape(-1, 2)
        Ls[j] = float((zz[:, 1] - zz[:, 0]).sum())
        if k == ri:                                  # eksene paralel kesit: halka uzunluğu
            dens[j] = 2 * np.pi * abs(c) * float((zz[:, 1] - zz[:, 0]).sum())
        else:                                        # radyal kesit: halka alanı
            dens[j] = float(np.pi * np.abs(zz[:, 1] ** 2 - zz[:, 0] ** 2).sum())
        p1 = np.zeros(2); p2 = np.zeros(2)
        p1[k] = c; p1[1 - k] = z[0]; p2[k] = c; p2[1 - k] = z[-1]
        elo[j], ehi[j] = p1, p2
    V = np.r_[0.0, np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.abs(np.diff(cs)))]
    if V[-1] <= 0:
        return {'ok': False, 'why': 'Ön form hacmi hesaplanamadı (radyal eksen / koordinatlar?).'}
    for E in (elo, ehi):                             # boş kesitleri en yakın geçerli değerle doldur
        ok = np.isfinite(E[:, 0])
        if ok.any():
            idx = np.where(ok, np.arange(n), -1)
            idx = np.maximum.accumulate(idx)
            first = int(np.argmax(ok))
            idx[idx < 0] = first
            E[:] = E[idx]
    Lv = Ls[Ls > 0]
    return {'ok': True, 'poly': poly, 'ri': ri, 'k': k, 'dir': dsgn, 'c': cs, 'V': V,
            'lo': elo, 'hi': ehi, 'Vtot': float(V[-1]), 'Vpoly': poly_volume(poly, ri),
            'tmax': float(Lv.max()), 'lmed': float(np.median(Lv)), 'shear': shear}


def sim_w0(fin, sim):
    """Makaranın geçmediği ama duvarın içinde kalan ışınlarda doğal (inceltilmemiş) kalınlık."""
    T = np.asarray(fin['T'], float)
    if sim['shear']:
        axis = np.array([0.0, 1.0]) if sim['ri'] == 0 else np.array([1.0, 0.0])
        sin_a = np.abs(fin['N'] @ axis)
        return np.where(np.isfinite(T), T, sim['lmed']) * sin_a
    return np.where(np.isfinite(T), T, sim['lmed'])


SIM_TOL = 1.0     # flow simülasyonu: makara boşluğu ≤ (ön formun kalıptan dış yüzeyine mesafesi + bu pay) ise şekillenir


def sim_limit(M, Nrm, chains, sim):
    """Flow simülasyonu için ışın başına 'makara malzemeye değiyor' sınırı: ön formun kalıptan
    (aradaki boşluk dahil) dış yüzeyine mesafesi. Ön formun hiç kesmediği ışınlarda: ortanca × 1,25.
    Böylece makaranın önündeki ışınlar, makara uzaktan geçerken şekillenmiş sayılmaz."""
    Tpg, Tmg = preform_thickness(M, Nrm, chains, gap_tol=1e9)
    Tout = np.where(np.asarray(sim['side']) < 0, Tmg, Tpg)
    fo = np.isfinite(Tout)
    thr_nan = 1.25 * float(np.median(Tout[fo])) if fo.any() else sim['tmax']
    return np.minimum(np.where(fo, Tout + SIM_TOL, thr_nan), sim['tmax'])


def sim_reach_times(M, arc, timelines, radii, chunk=2000):
    """Işın başına makaranın o noktanın hizasına ilk geldiği an: takım merkezinin kalıptaki izdüşümü
    (en yakın kalıp noktası) ile ışın arasındaki yay mesafesi ≤ takım yarıçapı. Makaranın önündeki
    ışınlar (makara henüz gelmeden, ön yüzüyle uzaktan değdiği yerler) böylece erken şekillenmez."""
    tR = np.full(len(M), np.inf)
    for i in range(N_CH):
        C, TT = tool_samples(timelines[i], radii[i])
        if len(C) == 0:
            continue
        for a in range(0, len(C), chunk):
            Cc, Tc = C[a:a + chunk], TT[a:a + chunk]
            d2 = ((Cc[:, None, :] - M[None, :, :]) ** 2).sum(-1)
            af = arc[np.argmin(d2, axis=1)]
            near = np.abs(arc[:, None] - af[None, :]) <= radii[i] + 1e-9
            tR = np.minimum(tR, np.where(near, Tc[None, :], np.inf).min(axis=1))
    return tR


def sim_formed(S, side, tmax, tR=None, tnow=np.inf):
    """Simülasyonda şekillendirme: malzeme kalıba gelerek uzadığı için ön formun ilk konumuna bakılmaz;
    makara kalıba ön form kalınlığından (tmax) daha yakın geçtiyse o nokta şekillenir."""
    U = np.full(len(side), np.inf)
    for s in S:
        same = np.isfinite(s) & ((s == 0) | (np.sign(s) == side))
        U = np.minimum(U, np.where(same & (np.abs(s) <= tmax + 1e-9), np.abs(s), np.inf))
    if tR is not None:
        U = np.where(tR <= tnow + 1e-9, U, np.inf)
    formed = np.isfinite(U)
    return np.where(formed, U, np.nan), formed


def remap_volume(P, ri, src, tgt):
    """Hacmi koruyarak taşı: eksenel koordinat kaydırılır, yarıçap r'² = r² + (r_hedef² − r_kaynak²)
    ile yeniden yerleştirilir (d(r²) korunur -> 2π·r·dA hacmi aynen korunur)."""
    P = np.asarray(P, float).copy()
    ax = 1 - ri
    P[..., ax] += tgt[ax] - src[ax]
    d = tgt[ri] ** 2 - src[ri] ** 2
    r = P[..., ri]
    P[..., ri] = np.sign(np.where(r == 0, 1.0, r)) * np.sqrt(np.maximum(r * r + d, 0.0))
    return P


def _interp_pt(Vq, V, P):
    return np.array([np.interp(Vq, V, P[:, 0]), np.interp(Vq, V, P[:, 1])])


BEND_STEP = 0.5        # bükülme ekseni adımı (mm)


def _nearest_ray(q, M):
    d2 = (q[0] - M[:, 0]) ** 2 + (q[1] - M[:, 1]) ** 2
    return int(np.argmin(d2))


def _mold_pen(q, M, N, sd):
    """Noktanın kalıba işaretli mesafesi ve en yakın ışın; kalıbın iki ucunun ötesindeki noktalar serbest (+inf)."""
    k = _nearest_ray(q, M)
    if k == len(M) - 1 and np.dot(q - M[-1], M[-1] - M[-2]) > 0:
        return np.inf, k
    if k == 0 and np.dot(q - M[0], M[0] - M[1]) > 0:
        return np.inf, k
    return float(np.dot(q - M[k], N[k] * sd[k])), k


def _mold_pen_all(P, M, N, sd):
    sg = _mold_signed(P, M, N, sd)
    tE, tS = M[-1] - M[-2], M[0] - M[1]
    beyond = (((P - M[-1]) @ tE) > 0) & (np.hypot(*(P - M[-1]).T) <= np.hypot(*(P - M[0]).T))
    beyond |= (((P - M[0]) @ tS) > 0) & (np.hypot(*(P - M[0]).T) < np.hypot(*(P - M[-1]).T))
    return np.where(beyond, np.inf, sg)


def _bend_tail(sim, c, use_lo, src, outer, tgt, M, N, sd, ri, Vtarget, remP, outP):
    """Duvarın ucuna taşınan kalan ön form (kuyruk) kalıba giriyorsa: kalıba değdiği yerden dışa doğru bükülür.
    Omurga = kuyruğun İÇ yüzeyi (kesit tablosundaki iç noktalar): kendi şekliyle ilerler; kalıba değen
    kesit kalıp yüzeyine oturur ve o andan itibaren kesitler biriken bükülme açısı kadar döner (kalıcı bükülme,
    kalıp geri çekilince geri yaylanmaz). Kesit kalınlığı yarıçap değişimine göre (2π·r·t) düzeltilir,
    toplam hacim tam korunur. Kalıba girmeyen kuyruğa dokunulmaz. Döndürür: (kuyruk poligonu, köprü dış noktası)"""
    if remP is None or len(remP) < 4:
        return remP, outP
    if float(_mold_pen_all(remP, M, N, sd).min()) >= -0.01:
        return remP, outP
    cs, d = sim['c'], sim['dir']
    m = d * cs > d * c
    if int(m.sum()) < 2:
        return remP, outP
    Ls, Hs = sim['lo'][m], sim['hi'][m]
    Is, Os = (Ls, Hs) if use_lo else (Hs, Ls)
    I = remap_volume(np.vstack([src, Is]), ri, src, tgt)
    O = remap_volume(np.vstack([outer, Os]), ri, src, tgt)
    n = len(I)
    In = I.copy()
    th = np.zeros(n)
    t = 0.0
    Rb = float(sim.get('Rb', 20.0))
    for j in range(1, n):
        sg = I[j] - I[j - 1]
        L = float(np.hypot(*sg))
        if L < 1e-12:
            In[j], th[j] = In[j - 1], t
            continue
        ct, st = np.cos(t), np.sin(t)
        rs = np.array([ct * sg[0] - st * sg[1], st * sg[0] + ct * sg[1]])
        cand = In[j - 1] + rs
        pen, k = _mold_pen(cand, M, N, sd)
        if pen < 0:                                   # kalıba değdi: dön (en fazla L/Rb), kalan girişi dışa it
            proj = cand - pen * (N[k] * sd[k])
            dd = proj - In[j - 1]
            need = float(np.arctan2(rs[0] * dd[1] - rs[1] * dd[0], rs[0] * dd[0] + rs[1] * dd[1]))
            lim = L / Rb
            use = min(max(need, -lim), lim)
            t += use
            cu, su = np.cos(use), np.sin(use)
            cand = In[j - 1] + np.array([cu * rs[0] - su * rs[1], su * rs[0] + cu * rs[1]])
            pen2, k2 = _mold_pen(cand, M, N, sd)
            if pen2 < 0:
                cand = cand - pen2 * (N[k2] * sd[k2])
        In[j], th[j] = cand, t
    W = O - I
    Wr = np.c_[np.cos(th) * W[:, 0] - np.sin(th) * W[:, 1], np.sin(th) * W[:, 0] + np.cos(th) * W[:, 1]]
    Wr *= (np.abs(I[:, ri]) / np.maximum(np.abs(In[:, ri]), 1e-6))[:, None]

    def build(fv):
        On = In + Wr * fv
        return np.vstack([In, On[::-1], In[:1]]), On

    fa, fb = 1.0, 1.02
    va, vb = poly_volume(build(fa)[0], ri) - Vtarget, poly_volume(build(fb)[0], ri) - Vtarget
    for _ in range(6):                                # hacmi tam koru (kalınlık çarpanı)
        if abs(vb) < 1e-9 * max(Vtarget, 1.0) or vb == va:
            break
        fa, fb, va = fb, fb - vb * (fb - fa) / (vb - va), vb
        vb = poly_volume(build(fb)[0], ri) - Vtarget
    poly, On = build(fb)
    return poly, On[:1]


def _sim_vat(sim, c):
    """Malzeme koordinatı c'ye kadar (sabit uçtan) biriken ön form hacmi."""
    cs, d = sim['c'], sim['dir']
    return float(np.interp(d * c, d * cs, sim['V']))


def sim_state(fin, sim, F, formed):
    """Verilen anda (F: makara boşluğu, formed: makara geçti mi) ürün şekli ve hacim dengesi.
    Duvar, makaranın ilk sıvadığı ışından (kalıp yönünde en baştaki şekillenmiş ışın) başlar;
    o kesitin öncesindeki ön form (göbek, flanş...) olduğu gibi yerinde kalır. Duvarın hacmi
    bu kesitten itibaren ön formdan tüketilir, kalan kısım duvarın ucuna taşınır (uzama)."""
    M, N = fin['M'], fin['N']
    sd = np.asarray(sim.get('side', fin['side']), float)
    ds, W0, ri = fin['ds'], sim.get('W0', fin['W0']), sim['ri']
    out = {'wall': None, 'rem': sim['poly'], 'bridge': None, 'fixed': None, 'T': np.zeros(2), 'Vwall': 0.0,
           'Vrem': sim['Vtot'], 'Vfix': 0.0, 'exhausted': False, 'front': -1, 'start': 0}
    if not formed.any():
        return out
    fi = np.where(formed)[0]
    a, front = int(fi[0]), int(fi[-1])
    k = sim['k']
    ca = float(M[a, k])
    V0 = _sim_vat(sim, ca)
    Vt = sim['Vtot'] - V0                            # duvar için kullanılabilir hacim
    fixed = clip_halfplane(sim['poly'], k, ca, -sim['dir']) if V0 > 0 else None
    w = np.where(formed[a:front + 1], F[a:front + 1], W0[a:front + 1])
    w = np.clip(np.nan_to_num(w), 0.0, None)
    Mf, Nf, sf = M[a:front + 1], N[a:front + 1], sd[a:front + 1]
    rmid = np.abs((Mf + Nf * (sf * w / 2)[:, None])[:, ri])
    cum = np.cumsum(2 * np.pi * rmid * w * ds[a:front + 1])
    out.update({'fixed': fixed, 'Vfix': V0, 'start': a})
    if cum[-1] > Vt:                                 # malzeme yetmedi: duvar burada biter
        kc = int(np.searchsorted(cum, Vt))
        m = max(kc, 1)
        front, out['exhausted'] = a + m, True
        w, Mf, Nf, sf, cum = w[:m + 1], Mf[:m + 1], Nf[:m + 1], sf[:m + 1], cum[:m + 1]
    Vc = float(min(cum[-1], Vt))
    Pout = Mf + Nf * (sf * w)[:, None]
    out.update({'wall': np.vstack([Pout, Mf[::-1], Pout[:1]]), 'Vwall': Vc, 'Vrem': Vt - Vc, 'front': front})
    if out['exhausted']:
        out['rem'] = None
        return out
    c = float(np.interp(V0 + Vc, sim['V'], sim['c']))
    plo, phi = _interp_pt(V0 + Vc, sim['V'], sim['lo']), _interp_pt(V0 + Vc, sim['V'], sim['hi'])
    nrm = Nf[-1] * sf[-1]
    inner, outer = (plo, phi) if np.dot(plo, nrm) <= np.dot(phi, nrm) else (phi, plo)
    tgt = Mf[-1]
    rem = clip_halfplane(sim['poly'], k, c, sim['dir'])
    out['T'] = tgt - inner
    out['src'], out['tgt'] = inner, tgt
    remP = None if rem is None else remap_volume(rem, ri, inner, tgt)
    outP = remap_volume(outer, ri, inner, tgt)[None, :]
    if remP is not None:
        remP, outP = _bend_tail(sim, c, inner is plo, inner, outer, tgt, M, N, sd, ri,
                                sim['Vtot'] - V0 - Vc, remP, outP)
        out['bent'] = remP is not None and outP is not None and not np.allclose(outP[0], remap_volume(outer, ri, inner, tgt))
    out['rem'] = remP
    out['bridge'] = np.array([tgt, Pout[-1], outP[0], tgt])
    out['c'] = c
    return out


def sim_xy(sim, st):
    """Parçanın X ve Y eksenlerindeki toplam boyu: ön form → şimdiki ürün.
    Döndürür: {'X': (önce, sonra), 'Y': (önce, sonra)}"""
    if sim.get('kind') == 'strip':
        P0, polys = sim['strip']['poly'], st.get('polys') or []
    else:
        P0, polys = sim['poly'], [st.get('fixed'), st.get('wall'), st.get('bridge'), st.get('rem')]
    polys = [np.asarray(P, float) for P in polys if P is not None and len(P)]
    A = np.vstack(polys) if polys else np.asarray(P0, float)
    return {ax: (float(P0[:, i].max() - P0[:, i].min()), float(A[:, i].max() - A[:, i].min()))
            for i, ax in enumerate(('X', 'Y'))}


def sim_xy_text(sim, st, names=('X', 'Y')):
    q = sim_xy(sim, st)
    return ", ".join(f"{nm} boyu {q[ax][0]:.2f} → {q[ax][1]:.2f} mm ({q[ax][1] - q[ax][0]:+.2f})"
                     for ax, nm in zip(('X', 'Y'), names))


def sim_summary(sim, st):
    """Hacim dengesi ve uç uzaması (malzeme yönünde)."""
    if sim.get('kind') == 'strip':
        Vw, Vp = st['Vwall'], st['Vpieces']
        el = st.get('elong', {})
        return {'Vtot': sim['Vtot'], 'Vwall': Vw, 'Vrem': Vp, 'err': (Vw + Vp - sim['Vtot']) / sim['Vtot'] * 100,
                'elong': max(el.values()) if el else 0.0, 'elong_groups': el, 'axis': 'sac', 'exhausted': st['exhausted']}
    k, d = sim['k'], sim['dir']
    ext = (lambda P: float(P[:, k].max() if d > 0 else P[:, k].min()))
    end_old = ext(sim['poly'])
    ref = st.get('rem') if st.get('rem') is not None else st.get('wall')
    elong = d * (ext(ref) - end_old) if ref is not None else 0.0
    Vrem = poly_volume(st['rem'], sim['ri']) if st.get('rem') is not None else 0.0
    Vfix = poly_volume(st['fixed'], sim['ri']) if st.get('fixed') is not None else 0.0
    return {'Vtot': sim['Vtot'], 'Vwall': st['Vwall'], 'Vrem': Vrem, 'Vfix': Vfix,
            'err': (st['Vwall'] + Vrem + Vfix - sim['Vtot']) / sim['Vtot'] * 100, 'elong': elong,
            'axis': 'X' if k == 0 else 'Y', 'exhausted': st.get('exhausted', False)}


SIM_COLOR = '#3d6fb6'


def draw_sim_product(ax, st, label='Ürün (simülasyon sonu)'):
    first = True
    polys = [st.get('fixed'), st.get('wall'), st.get('bridge'), st.get('rem')]
    if 'polys' in st:
        polys = st['polys']
    for poly in polys:
        if poly is None:
            continue
        ax.fill(poly[:, 0], poly[:, 1], color=SIM_COLOR, alpha=0.55, linewidth=0, zorder=3,
                label=(label if first else ''))
        ax.plot(poly[:, 0], poly[:, 1], color='#16325c', linewidth=1.0, zorder=4)
        first = False


# ---------- çizim yardımcıları ----------
PF_COLOR, FINAL_COLOR, DISP_COLOR = '#7f7f7f', '#1f3a5f', '#d62728'
FILL_COLOR, FILL_ALPHA = '#2f7fd8', 0.38           # kalıp ile son yüzey arası (et) dolgusu
UNDER_COLOR, IDEAL_COLOR = '#ff7f0e', '#6a3d9a'


def draw_preform_outline(ax, chains, label='Ön Form (DXF)'):
    first = True
    for A, _cl in chains:
        ax.plot(A[:, 0], A[:, 1], color=PF_COLOR, linewidth=1.3, linestyle=(0, (8, 3, 2, 3)),
                label=(label if first else ''), zorder=2)
        first = False


def draw_final_form(ax, final):
    first = True
    for P, Mk in final['chunks']:
        poly = np.vstack([P, Mk[::-1]])
        ax.fill(poly[:, 0], poly[:, 1], color=FILL_COLOR, alpha=FILL_ALPHA, linewidth=0, zorder=1)
        ax.plot(P[:, 0], P[:, 1], color=FINAL_COLOR, linewidth=1.8,
                label=(final.get('line_label', 'Son form yüzeyi') if first else ''), zorder=4)
        first = False
    for label, color, chunks in final.get('regions', []):
        first = True
        for Pa, Pb in chunks:
            poly = np.vstack([Pa, Pb[::-1]])
            ax.fill(poly[:, 0], poly[:, 1], facecolor=color, alpha=0.30, edgecolor=color,
                    hatch='////', linewidth=0, label=(label if first else ''), zorder=2)
            first = False
    for label, color, chunks in final.get('lines', []):
        first = True
        for P in chunks:
            ax.plot(P[:, 0], P[:, 1], color=color, linewidth=1.5, linestyle=(0, (4, 2)),
                    label=(label if first else ''), zorder=5)
            first = False


def thickness_profile(form, step_mm=1.0):
    """Et kalınlığı profili: grafik (SVG kartı), özet metni, tablo."""
    if not form or not form.get('final'):
        return ("<p>Et kalınlığı profili için <b>Ön form (DXF)</b> yükleyip simülasyonu çalıştır.</p>",
                "", pd.DataFrame())
    fin = form['final']
    p = fin['profile']
    shear = fin.get('mode') == 'shear'
    m = p['main'] & np.isfinite(p['T0'])
    if not m.any():
        why = ("Ön formun radyal aralığı kalıpla örtüşmüyor — radyal ekseni ve DXF koordinatlarını kontrol et."
               if shear else "Ön form kalıbın hiçbir noktasının üzerinde bulunamadı. DXF koordinatlarını / kaydırmayı kontrol et.")
        return f"<p>{why}</p>", "", pd.DataFrame()
    arc, T0, T, F, fm = p['arc'], p['T0'], p['T'], p['F'], m & p['formed']
    fig = Figure(figsize=(14, 6.5), dpi=100)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.86)
    W = lambda a, mask: np.where(mask, a, np.nan)
    ax2 = ax.twinx()
    if shear:
        dev = np.where(fm & (T > 1e-9), (F - T) / np.where(T > 1e-9, T, 1) * 100, np.nan)
        ax.plot(arc, W(T0, m), color=PF_COLOR, linewidth=1.5, linestyle='--', label='Ön form eksenel kalınlığı t₀')
        ax.plot(arc, W(T, m), color=IDEAL_COLOR, linewidth=1.8, linestyle=(0, (4, 2)), label='İdeal (sinüs kuralı) t₀·sin α')
        ax.plot(arc, W(F, fm), color=FINAL_COLOR, linewidth=2.4, label='Gerçek (takım boşluğu)')
        ax.fill_between(arc, W(F, fm), W(T, fm), where=fm & (F < T), color=DISP_COLOR, alpha=0.25, label='Aşırı inceltme')
        ax.fill_between(arc, W(F, fm), W(T, fm), where=fm & (F > T), color=UNDER_COLOR, alpha=0.25, label='Eksik inceltme')
        ax2.plot(arc, W(dev, fm), color='#444', linewidth=0.9, alpha=0.8, label='Sapma (%) — + kalın / − ince')
        ax2.axhline(0, color='#444', linewidth=0.6, linestyle=':')
        ax2.set_ylabel('Sinüs kuralından sapma (%)', fontsize=11)
        ax.set_title('Et Kalınlığı — Shear Forming (sinüs kuralı ile karşılaştırma)', fontsize=14, pad=12)
    else:
        dev = np.where(fm & (T > 1e-9), (T - F) / np.where(T > 1e-9, T, 1) * 100, np.nan)
        ax.fill_between(arc, W(F, m), W(T, m), where=fm, color=DISP_COLOR, alpha=0.18, label='İnceltme')
        ax.plot(arc, W(T, m), color=PF_COLOR, linewidth=1.6, linestyle='--', label='Ön form kalınlığı')
        ax.plot(arc, W(F, m), color=FINAL_COLOR, linewidth=2.2, label='Son kalınlık')
        ax2.plot(arc, W(dev, fm), color=DISP_COLOR, linewidth=1.0, alpha=0.8, label='İnceltme oranı (%)')
        ax2.set_ylabel('İnceltme oranı (%)', color=DISP_COLOR, fontsize=11)
        ax2.tick_params(axis='y', colors=DISP_COLOR)
        ax2.set_ylim(0, 100)
        ax.set_title('Et Kalınlığı Profili (Ön Form → Son Form)', fontsize=14, pad=12)
    ax.set_xlabel('Kalıp konturu boyunca mesafe (mm) — kalıp başlangıcından', fontsize=11)
    ax.set_ylabel('Et kalınlığı (kalıba dik, mm)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9, shadow=True)

    lines = []
    if shear:
        al = p['alpha']
        lines.append(f"**Model:** shear forming, sinüs kuralı (radyal eksen {fin.get('radial', 'X')}). "
                     f"**Ön form eksenel kalınlığı t₀:** {np.nanmin(T0[m]):.3f} – {np.nanmax(T0[m]):.3f} mm")
        if fm.any():
            lines += [f"**Şekillenen bölge:** kalıp boyunca {arc[fm].min():.1f} – {arc[fm].max():.1f} mm "
                      f"(kalıp açısı α {np.nanmin(al[fm]):.1f}° – {np.nanmax(al[fm]):.1f}°)",
                      f"**Gerçek et kalınlığı:** {np.nanmin(F[fm]):.3f} – {np.nanmax(F[fm]):.3f} mm; "
                      f"**ideal:** {np.nanmin(T[fm]):.3f} – {np.nanmax(T[fm]):.3f} mm"]
            d = F[fm] - T[fm]
            k_lo, k_hi = int(np.nanargmin(d)), int(np.nanargmax(d))
            lines.append(f"**En büyük aşırı inceltme:** {min(d[k_lo], 0):+.3f} mm (kalıp boyunca {arc[fm][k_lo]:.1f} mm) — "
                         f"**en büyük eksik inceltme:** {max(d[k_hi], 0):+.3f} mm (kalıp boyunca {arc[fm][k_hi]:.1f} mm)")
        else:
            lines.append("**Takım ön form kalınlığından daha dar bir boşlukla hiç geçmedi** — şekillenme yok.")
        cols = ['Kalıp boyunca (mm)', 'X', 'Y', 'Kalıp açısı α (°)', 't₀ eksenel (mm)', 'İdeal t₀·sin α (mm)',
                'Gerçek (mm)', 'Sapma (mm)', 'Sapma (%)']
    else:
        _mn = 'sac ön form (sıvama + inceltme)' if fin.get('mode') == 'strip' else 'kalıp üzerinde'
        lines.append(f"**Model:** {_mn}. **Ön form (sac) kalınlığı:** {np.nanmin(T[m]):.3f} – {np.nanmax(T[m]):.3f} mm")
        if fm.any():
            lines += [f"**Formlanan bölge:** kalıp boyunca {arc[fm].min():.1f} – {arc[fm].max():.1f} mm",
                      f"**Son kalınlık:** en az {np.nanmin(F[fm]):.3f} mm, en çok {np.nanmax(F[fm]):.3f} mm, "
                      f"ortalama {np.nanmean(F[fm]):.3f} mm",
                      f"**En büyük inceltme:** %{np.nanmax(dev[fm]):.1f} (kalıp boyunca {arc[fm][np.nanargmax(dev[fm])]:.1f} mm)"]
        else:
            lines.append("**Takım ön forma hiç girmedi** — form oluşmadı.")
        cols = ['Kalıp boyunca (mm)', 'X', 'Y', 'Ön form (mm)', 'Son (mm)', 'İnceltme (%)']
    sm = fin.get('sim')
    if sm is not None:
        if sm.get('ok') and sm.get('final') is not None:
            q = sim_summary(sm, sm['final'])
            if q.get('elong_groups') is not None:
                el = ", ".join(f"{k}: {v:.2f} mm" for k, v in q['elong_groups'].items())
                el = f"**uç uzaması (sac boyunca): {el or '—'}**"
            else:
                el = f"**uç uzaması ({q['axis']} yönünde): {q['elong']:.2f} mm**"
            lines.append(f"**Simülasyon (hacim sabit):** ön form {q['Vtot']:,.0f} mm³ = kalıba sıvanan "
                         f"{q['Vwall']:,.0f} + kalan {q['Vrem']:,.0f} mm³ (fark %{q['err']:.2f}); " + el
                         + (" — **malzeme bitti**" if q['exhausted'] else ""))
            lines.append("**Eksenlere göre uzama:** " + sim_xy_text(sm, sm['final']))
        else:
            lines.append(f"**Simülasyon çalışmadı:** {sm.get('why', '')}")
    idx = np.where(m)[0]
    keep, last = [], -1e18
    for j in idx:
        if arc[j] - last >= step_mm - 1e-9:
            keep.append(j); last = arc[j]
    r = lambda a, n=4: np.round(a[keep], n)
    if shear:
        df = pd.DataFrame({cols[0]: r(arc, 3), cols[1]: r(p['x'], 3), cols[2]: r(p['y'], 3),
                           cols[3]: r(p['alpha'], 2), cols[4]: r(T0), cols[5]: r(T), cols[6]: r(F),
                           cols[7]: np.round((F - T)[keep], 4), cols[8]: np.round(dev[keep], 2)})
    else:
        df = pd.DataFrame({cols[0]: r(arc, 3), cols[1]: r(p['x'], 3), cols[2]: r(p['y'], 3),
                           cols[3]: r(T), cols[4]: r(F), cols[5]: np.round(np.nan_to_num(dev[keep]), 2)})
    return fig_to_svg_card(fig), "  \n".join(lines), df


# ===== Hücre 18 =====
# =============================================================
# MAKARA PROFİLİ (DXF), KALIP DXF → G-KODU, TAKIM AŞINMASI
# =============================================================
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

WEAR_CMAP = mcolors.LinearSegmentedColormap.from_list(      # tek tonlu sıra: açık → koyu turuncu
    'wear_orange', ['#fdf1ea', '#f6b896', '#eb6834', '#b8441a', '#6e2408'])
INK, INK2, GRID = '#222222', '#666666', '#e6e6e6'


# -------------------------------------------------------------
# 1) MAKARA PROFİLİ: DXF'teki 0,0 = makara merkezi (CNC'nin kontrol ettiği nokta)
# -------------------------------------------------------------
def _load_fine(path, tol=0.002):
    """DXF'i ince yay bölmesiyle oku (makara / kalıp hassasiyeti)."""
    old = DXF_ARC_TOL[0]
    DXF_ARC_TOL[0] = tol
    try:
        return load_preform_file(path)
    finally:
        DXF_ARC_TOL[0] = old


def load_tool_dxf(path, tr=None):
    """Makara kesit profilini DXF'ten okur (kapalı kontur; açık/boşluklu çizgiler onarılır).
    Döndürür: ({'name', 'shape'}, mesaj)"""
    st, _msg = _load_fine(path)
    ch, rnote = preform_chains(st, None, with_note=True)
    P = _largest_closed(ch)
    if P is None:
        raise ValueError("Makara DXF'inde kapalı bir kontur bulunamadı. Makara kesitini kapalı kontur "
                         "(polyline / çizgi + yay) olarak çiz; 0,0 makara merkezi olmalı.")
    P = simplify_poly(P, 0.003, 240)
    if poly_area(P) < 0:
        P = P[::-1]
    name = os.path.basename(path)
    lines = [f"Makara profili yüklendi: {name} — {len(P) - 1} kenar."]
    if rnote:
        lines.append(rnote)
    inside = bool(_pts_in_poly(np.zeros((1, 2)), P)[0])
    reach = float(np.max(np.hypot(P[:, 0], P[:, 1])))
    lines.append(f"Profil sınırları X {P[:, 0].min():.2f}…{P[:, 0].max():.2f}, Y {P[:, 1].min():.2f}…{P[:, 1].max():.2f} mm; "
                 f"merkezden en uzak nokta {reach:.2f} mm.")
    if not inside:
        lines.append("UYARI: 0,0 noktası profilin dışında. DXF'te makara merkezini 0,0'a taşıdığından emin ol.")
    if tr:
        lines.append(f"G41/G42 kompanzasyonu girilen takım yarıçapıyla ({float(tr):g} mm) yapılır (CNC'deki gibi); "
                     "radüs izi, temas noktası, mesafe ölçümü ve çizimler bu profille yapılır.")
    return {'name': name, 'shape': [[round(float(x), 5), round(float(y), 5)] for x, y in P]}, "\n".join(lines)


def tool_preview_html(entry, tr, color='#0072B2'):
    """Yüklenen makara profilinin küçük önizlemesi (0,0 merkez, kesikli: kompanzasyon yarıçapı)."""
    if not entry:
        return ""
    P = np.asarray(entry['shape'], float)
    fig = Figure(figsize=(3.6, 3.2), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.fill(P[:, 0], P[:, 1], color=color, alpha=0.18, linewidth=0)
    ax.plot(P[:, 0], P[:, 1], color=color, linewidth=1.6)
    if tr:
        a = np.linspace(0, 2 * np.pi, 181)
        ax.plot(float(tr) * np.cos(a), float(tr) * np.sin(a), color=INK2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.plot([0], [0], '+', color=INK, markersize=9, markeredgewidth=1.5)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, color=GRID, linewidth=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{entry.get('name', 'makara')}  (+ = 0,0 merkez, kesikli = R{float(tr):g})", fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=INK2)
    fig.tight_layout()
    return fig_to_svg_card(fig)


def apply_tool_shapes(cfg, tool_state):
    """Arayüzdeki makara profillerini (kanal no → {'shape'}) konfigürasyona işler."""
    ts = tool_state if isinstance(tool_state, dict) else {}
    for i, ch in enumerate(cfg['channels']):
        e = ts.get(str(i)) or ts.get(i)
        ch['shape'] = np.asarray(e['shape'], float) if e and e.get('shape') else None
    return cfg


# -------------------------------------------------------------
# 2) KALIP DXF → G-KODU (doğru + yay uydurma; JSON ile saklanabilir)
# -------------------------------------------------------------
MOLD_FIT_TOL = 0.002        # G-kodu uydurma toleransı (mm)


def _fmt4(v):
    s = f"{v:.4f}".rstrip('0').rstrip('.')
    return '0' if s in ('', '-0') else s


def _circle3(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None
    ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
    uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
    return np.array([ux, uy])


def _fits_line(P, i, j, tol):
    a, b = P[i], P[j]
    ab = b - a
    L = float(np.hypot(*ab))
    if L < 1e-12:
        return False
    Q = P[i:j + 1]
    d = np.abs(ab[0] * (Q[:, 1] - a[1]) - ab[1] * (Q[:, 0] - a[0])) / L
    t = ((Q - a) @ ab) / (L * L)
    return bool(d.max() <= tol and t.min() >= -1e-9 and t.max() <= 1 + 1e-9 and np.all(np.diff(t) > -1e-9))


def _fits_arc(P, i, j, tol):
    """i..j noktaları tek bir yaya (< 180°, tek yönlü) tolerans içinde oturuyor mu? (merkez, yarıçap, saat yönü)"""
    if j - i < 2:
        return None
    m = (i + j) // 2
    c = _circle3(P[i], P[m], P[j])
    if c is None:
        return None
    Q = P[i:j + 1]
    r = float(np.hypot(*(P[i] - c)))
    if r > 1e5 or abs(np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1]) - r).max() > tol:
        return None
    ang = np.unwrap(np.arctan2(Q[:, 1] - c[1], Q[:, 0] - c[0]))
    da = np.diff(ang)
    if not (np.all(da > 1e-9) or np.all(da < -1e-9)):
        return None
    if abs(ang[-1] - ang[0]) >= np.pi * 0.995:
        return None
    return c, r, bool(da[0] < 0)


def _grow(n, i, j0, test):
    """test(i, j) doğru kaldıkça j'yi j0'dan büyüt (üstel adım + ikili arama). Döndürür: en büyük geçerli j ya da None."""
    if j0 >= n or not test(i, j0):
        return None
    good, step = j0, 1
    while good < n - 1:
        j = min(good + step, n - 1)
        if test(i, j):
            good, step = j, step * 2
            continue
        lo, hi = good, j
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if test(i, mid):
                lo = mid
            else:
                hi = mid
        return lo
    return good


def polyline_to_gcode(P, tol=MOLD_FIT_TOL):
    """Nokta dizisini G1 / G2 / G3 (R) satırlarına çevirir. Döndürür: (satırlar, (x0, y0))"""
    P = np.asarray(P, float)
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-9]
    P = P[keep]
    out, i, n = [], 0, len(P)
    while i < n - 1:
        jl = _grow(n, i, i + 1, lambda a, b: _fits_line(P, a, b, tol)) or i + 1
        ja = _grow(n, i, i + 2, lambda a, b: _fits_arc(P, a, b, tol) is not None)
        if ja is not None and ja >= i + 3 and ja > jl:
            c, r, cw = _fits_arc(P, i, ja, tol)
            e = P[ja]
            out.append(f"{'G2' if cw else 'G3'} X{_fmt4(e[0])} Y{_fmt4(e[1])} R{_fmt4(r)}")
            i = ja
        else:
            # doğru, ardından gelen teğet yayın ilk noktalarını yutmasın: yay jl-1'den başlayabiliyorsa geri çekil
            while jl - 1 > i:
                jb = _grow(n, jl - 1, jl + 1, lambda a, b: _fits_arc(P, a, b, tol) is not None)
                if jb is not None and jb >= jl + 2:
                    jl -= 1
                else:
                    break
            e = P[jl]
            out.append(f"G1 X{_fmt4(e[0])} Y{_fmt4(e[1])}")
            i = jl
    return out, (float(P[0, 0]), float(P[0, 1]))


def _split_face(P, ri, face):
    """Kapalı konturdan kalıp yüzeyini seç: eksen yönünde iki uç noktası arasındaki iki yoldan
    yarıçapı büyük olan (dış) ya da küçük olan (iç)."""
    Q = P[:-1] if np.hypot(*(P[0] - P[-1])) < 1e-9 else P
    ai = 1 - ri
    i0, i1 = int(np.argmin(Q[:, ai])), int(np.argmax(Q[:, ai]))
    n = len(Q)
    path_a = [Q[(i0 + k) % n] for k in range((i1 - i0) % n + 1)]
    path_b = [Q[(i0 - k) % n] for k in range((i0 - i1) % n + 1)]

    def mean_r(path):
        A = np.asarray(path)
        L = np.hypot(*np.diff(A, axis=0).T)
        mid = np.abs((A[:-1, ri] + A[1:, ri]) / 2)
        return float((mid * L).sum() / max(L.sum(), 1e-12))

    pa, pb = np.asarray(path_a), np.asarray(path_b)
    big, small = (pa, pb) if mean_r(path_a) >= mean_r(path_b) else (pb, pa)
    return big if face == 'outer' else small


MOLD_FACE_CHOICES = [('Dış yüz (yarıçapı büyük taraf)', 'outer'), ('İç yüz (yarıçapı küçük taraf)', 'inner'),
                     ('Tüm kontur (olduğu gibi)', 'all')]


def dxf_to_mold(path, face='outer', radial='X', reverse=False, tol=MOLD_FIT_TOL):
    """Kalıp DXF'ini kalıp G-koduna çevirir. Açık çizgi ise olduğu gibi; kapalı konturdan seçilen yüz alınır.
    Varsayılan yön: eksen koordinatı büyük uçtan küçüğe (ters çevrilebilir).
    Döndürür: ({'x0','y0','code','dia_mode','dia_axis','source'}, mesaj)"""
    st, _m = _load_fine(path)
    ch = _join_chains(st, None)
    if not ch:
        raise ValueError("Kalıp DXF'inde çizgi bulunamadı.")
    plen = lambda A: float(np.hypot(*np.diff(A, axis=0).T).sum())
    ch = sorted(ch, key=lambda c: -plen(c[0]))
    P, closed = ch[0]
    ri = 0 if radial == 'X' else 1
    notes = []
    if len(ch) > 1:
        notes.append(f"DXF'te {len(ch)} ayrı çizgi zinciri var; en uzunu ({plen(P):.1f} mm) kalıp olarak alındı.")
    if closed and face != 'all':
        P = _split_face(P, ri, face)
        notes.append(f"Kapalı kontur: {'dış' if face == 'outer' else 'iç'} yüz alındı (parça yarıçapı ekseni {radial}).")
    ai = 1 - ri
    if not closed or face != 'all':
        if P[0, ai] < P[-1, ai]:
            P = P[::-1]
    if reverse:
        P = P[::-1]
    lines, (x0, y0) = polyline_to_gcode(P, tol)
    code = "\n".join(lines)
    # gidiş-dönüş kontrolü: G-kodundan yeniden üretilen yol, DXF'e ne kadar yakın?
    try:
        pts = np.asarray(get_mold_points_cfg({'x0': x0, 'y0': y0, 'code': code, 'dia_mode': 'radius', 'dia_axis': 'X'}), float)
        dev = _poly_dev(pts, P)
        notes.append(f"Doğrulama: G-kodundan çizilen kalıp DXF'ten en fazla {dev:.3f} mm sapıyor (iki tarafın yay bölme hatası dahil).")
    except Exception as e:
        notes.append(f"Doğrulama yapılamadı: {e}")
    n2 = sum(1 for ln in lines if ln.startswith(('G2', 'G3')))
    msg = (f"Kalıp DXF'i G-koduna çevrildi: {os.path.basename(path)} → {len(lines)} satır "
           f"({len(lines) - n2} G1, {n2} G2/G3), başlangıç X{_fmt4(x0)} Y{_fmt4(y0)}.\n" + "\n".join(notes) +
           "\nKalıp alanları dolduruldu. Saklamak için 'Kalıbı JSON olarak indir' (yalnız kalıp) ya da "
           "'Ayarları JSON Olarak Kaydet' (tüm ayarlar).")
    return {'x0': round(x0, 4), 'y0': round(y0, 4), 'code': code, 'dia_mode': 'radius', 'dia_axis': 'X',
            'source': os.path.basename(path)}, msg


def _poly_dev(A, B):
    """A noktalarının B çoklu çizgisine en büyük uzaklığı (ve tersi)."""
    def one(P, Q):
        a, ab = Q[:-1], Q[1:] - Q[:-1]
        L2 = np.maximum((ab ** 2).sum(1), 1e-18)
        best = 0.0
        for k in range(0, len(P), 512):
            X = P[k:k + 512]
            t = np.clip(((X[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0, 1)
            d = np.hypot(*(a[None] + t[..., None] * ab[None] - X[:, None, :]).transpose(2, 0, 1)).min(1)
            best = max(best, float(d.max()))
        return best
    return max(one(A, B), one(B, A))


# -------------------------------------------------------------
# 3) TAKIM AŞINMASI (yaklaşık model)
#  Makara profilinin her noktası için, her takım adımında:
#     dW = K · h^a · (v / v_ref)^b · Δs
#  h  : o profil noktasının gelen malzeme yüzeyinin ne kadar altında olduğu = süpürdüğü kalınlık (mm)
#  v  : ilerleme hızı F (mm/dk),  Δs : adımda alınan yol (mm)
#  Gelen yüzey: ön form kalınlığı (varsa) ile önceki pasların / diğer kanalların bıraktığı yüzeyin küçüğü.
#  Aynı pasın hemen arkasındaki izi (takımın kendisi) sayılmaz: makara önündeki malzemeyle temas eder.
#  Yalnız ön yüz (normali ilerleme yönüne ve kalıba bakan profil noktaları) temas yayıdır.
# -------------------------------------------------------------
WEAR_NB = 240


def _wear_samples(timeline, tool_r):
    """Takım merkezi örnekleri + zaman + satır no + ilerleme hızı + adım yolu (G0 hariç)."""
    step = max(0.15, float(tool_r) / 30.0)
    C, T, LN, V, DS, U = [], [], [], [], [], []
    for (t0, t1, seg) in timeline:
        if seg['type'] != 'move' or _is_g0(seg):
            continue
        pts = np.asarray(seg['points'], float)
        if len(pts) < 2:
            continue
        n_seg = len(pts) - 1
        L_tot = float(np.hypot(*np.diff(pts, axis=0).T).sum())
        v = L_tot / max(t1 - t0, 1e-9) * 60.0
        for i in range(n_seg):
            a, b = pts[i], pts[i + 1]
            L = float(np.hypot(*(b - a)))
            if L < 1e-12:
                continue
            m = max(int(np.ceil(L / step)), 1)
            for j in range(m):
                u = (j + 0.5) / m
                C.append(a + (b - a) * u)
                T.append(t0 + (t1 - t0) * (i + u) / n_seg)
                LN.append(seg.get('line', -1)); V.append(v); DS.append(L / m); U.append((b - a) / L)
    if not C:
        return None
    return {'C': np.asarray(C), 'T': np.asarray(T), 'line': np.asarray(LN), 'v': np.asarray(V),
            'ds': np.asarray(DS), 'arc': np.cumsum(DS), 'u': np.asarray(U)}


def _boundary(tool_r, n=WEAR_NB):
    """Profili eşit aralıklı n noktaya böl; dış normaller (saat yönü tersine konturda)."""
    O = tool_outline(tool_r, 720)
    if poly_area(O) < 0:
        O = O[::-1]
    L = np.r_[0, np.cumsum(np.hypot(*np.diff(O, axis=0).T))]
    s = (np.arange(n) + 0.5) * L[-1] / n
    B = np.c_[np.interp(s, L, O[:, 0]), np.interp(s, L, O[:, 1])]
    k = np.clip(np.searchsorted(L, s) - 1, 0, len(O) - 2)
    tg = O[k + 1] - O[k]
    tg /= np.maximum(np.hypot(tg[:, 0], tg[:, 1]), 1e-12)[:, None]
    nb = np.c_[tg[:, 1], -tg[:, 0]]
    return B, nb, s


def _pair_table(M, N, smp, tool_r, sd, smax):
    """Kanal örneklerinin ışınlara giriş mesafeleri (malzeme tarafında, |s| ≤ smax) → ışın başına
    örnek sırasına göre birikimli en küçük değer. Döndürür: (anahtarlar, değerler, K)"""
    C = smp['C']
    K = len(C)
    I, KK, S = [], [], []
    poly = tool_shape(tool_r) is not None
    step = 12 if poly else 256
    reach = tool_reach(tool_r) + smax
    for a in range(0, K, step):
        Cc = C[a:a + step]
        lo, hi = Cc.min(0) - reach, Cc.max(0) + reach
        rid = np.where((M[:, 0] >= lo[0]) & (M[:, 0] <= hi[0]) & (M[:, 1] >= lo[1]) & (M[:, 1] <= hi[1]))[0]
        if not len(rid):
            continue
        s = tool_ray_matrix(M[rid], N[rid], Cc, tool_r)
        ok = np.isfinite(s) & (np.abs(s) <= smax) & ((s == 0) | (np.sign(s) == sd[rid, None]))
        r_, k_ = np.nonzero(ok)
        I.append(rid[r_]); KK.append(a + k_); S.append(np.abs(s[r_, k_]))
    if not I:
        return np.zeros(0, np.int64), np.zeros(0), K
    I, KK, S = np.concatenate(I), np.concatenate(KK), np.concatenate(S)
    key = I.astype(np.int64) * (K + 1) + KK
    o = np.argsort(key, kind='stable')
    key, I, S = key[o], I[o], S[o]
    cm = pd.Series(S).groupby(I).cummin().to_numpy()
    return key, cm, K


def _pair_query(tab, ray, kmax):
    """Işın ray için örnek no ≤ kmax olanların en küçük giriş mesafesi (yoksa inf)."""
    key, cm, K = tab
    if len(key) == 0:
        return np.full(len(ray), np.inf)
    q = ray.astype(np.int64) * (K + 1) + np.asarray(kmax, np.int64)
    pos = np.searchsorted(key, q, side='right') - 1
    ok = (pos >= 0) & (np.asarray(kmax) >= 0)
    pos_c = np.clip(pos, 0, len(key) - 1)
    ok &= (key[pos_c] // (K + 1)) == ray
    return np.where(ok, cm[pos_c], np.inf)


def _nearest_rays(M, Q):
    try:
        from scipy.spatial import cKDTree
        return cKDTree(M).query(Q)[1]
    except Exception:
        out = np.empty(len(Q), int)
        for a in range(0, len(Q), 2048):
            d2 = ((Q[a:a + 2048, None, :] - M[None]) ** 2).sum(-1)
            out[a:a + 2048] = np.argmin(d2, axis=1)
        return out


def _engage_setup(timelines, radii, mold_points, fin=None, env_limit=0.0):
    """Temas hesabının ortak hazırlığı (aşınma ve proses hesapları): kalıp ışınları, malzeme tarafı,
    gelen malzeme yüzeyi, takım örnekleri ve örnek→ışın giriş tabloları."""
    M, N, _ = mold_rays(mold_points)
    if len(M) < 2:
        raise ValueError("Kalıp noktası yetersiz.")
    total_T = max((tl[-1][1] for tl in timelines if tl), default=1.0)
    act = [bool(tl) and channel_active([s for _a, _b, s in tl]) for tl in timelines]
    samples = [(_wear_samples(timelines[i], radii[i]) if act[i] else None) for i in range(N_CH)]
    # malzeme tarafı (ışın başına): ön form varsa ondan, yoksa takımın bulunduğu taraf
    if fin is not None and len(fin.get('side', [])) == len(M):
        sd = np.asarray(fin['side'], int)
    else:
        sd = np.zeros(len(M), int)
        for i in range(N_CH):
            if samples[i] is not None:
                s_env = radius_envelope(M, N, samples[i]['C'], radii[i])
                m = (sd == 0) & np.isfinite(s_env) & (s_env != 0)
                sd[m] = np.sign(s_env[m]).astype(int)
        sd[sd == 0] = 1
    # başlangıç malzeme yüzeyi (kalıptan uzaklık)
    if fin is not None and len(fin.get('T', [])) == len(M) and np.isfinite(fin['T']).any():
        T0 = np.asarray(fin['T'], float)
        S0 = np.where(np.isfinite(T0), T0, float(np.nanmedian(T0)))
        src = "ön form kalınlığı (DXF)"
        # ön formun ulaşmadığı ışınlar: malzeme oraya ancak akarak gelir. Simülasyon varsa, duvarın
        # ulaşmadığı (malzemenin bittiği) ışınlarda gelen kalınlık 0 alınır.
        sm = fin.get('sim')
        if sm is not None and sm.get('ok') and sm.get('kind') != 'strip' and sm.get('final') is not None:
            stf = sm['final']
            a_, f_ = int(stf.get('start', 0)), int(stf.get('front', -1))
            idx = np.arange(len(M))
            S0 = np.where(~np.isfinite(T0) & ((idx < a_) | (idx > f_)), 0.0, S0)
            src += "; duvarın ulaşmadığı yerde malzeme yok"
    elif env_limit and env_limit > 0:
        S0 = np.full(len(M), float(env_limit)); src = f"ham malzeme kalınlığı {env_limit:g} mm"
    else:
        rr = max([float(r) for r, a_ in zip(radii, act) if a_] or [10.0])
        S0 = np.full(len(M), rr); src = f"takım yarıçapı ({rr:g} mm; ham kalınlık girilmedi)"
    smax = float(np.max(S0)) + 1.0
    tabs = [(_pair_table(M, N, samples[i], radii[i], sd, smax) if samples[i] is not None else None)
            for i in range(N_CH)]
    return {'M': M, 'N': N, 'sd': sd, 'S0': S0, 'src': src, 'samples': samples, 'tabs': tabs, 'T': total_T,
            'radii': radii}


def _engage_chunks(E, i, B, nb, chunk=64):
    """Kanal i için örnek öbekleri: her (örnek, profil noktası) çiftinde
    ray = en yakın kalıp ışını, sQ = noktanın kalıptan uzaklığı, Sin = gelen malzeme yüzeyi,
    h = noktanın gelen yüzeyin altında kaldığı derinlik (yalnız ön / kalıba bakan yüzde > 0),
    mold_face = nokta kalıba bakıyor mu. Diziler (örnek sayısı, profil noktası sayısı) biçiminde."""
    M, N, sd, S0, samples, tabs = E['M'], E['N'], E['sd'], E['S0'], E['samples'], E['tabs']
    smp = samples[i]
    nB = len(B)
    lag = 2.0 * tool_reach(E['radii'][i])
    Kn = len(smp['C'])
    kown = np.searchsorted(smp['arc'], smp['arc'] - lag, side='right') - 1
    nM = len(M)
    kE = np.where(np.hypot(*(M - M[-1]).T) > 1e-9)[0]
    kS = np.where(np.hypot(*(M - M[0]).T) > 1e-9)[0]
    tE = M[-1] - M[kE[-1]] if len(kE) else np.zeros(2)       # kalıbın son ve ilk ucundaki dışa doğru teğetler
    tS = M[0] - M[kS[0]] if len(kS) else np.zeros(2)
    for a0 in range(0, Kn, chunk):
        ks = np.arange(a0, min(a0 + chunk, Kn))
        Q = (smp['C'][ks][:, None, :] + B[None]).reshape(-1, 2)
        ray = _nearest_rays(M, Q)
        nd = N[ray] * sd[ray][:, None]
        dQ = Q - M[ray]
        # kalıba gerçek uzaklık (en yakın kalıp noktası), malzeme tarafında +; köşe / uç ışınlarının eğik
        # normalleri sonucu bozmasın. Kalıbın iki ucunun ötesine düşen noktalar hesaba katılmaz.
        sQ = np.hypot(dQ[:, 0], dQ[:, 1]) * np.where((dQ * nd).sum(1) < 0, -1.0, 1.0)
        beyond = ((ray == nM - 1) & ((dQ * tE).sum(1) > 0)) | ((ray == 0) & ((dQ * tS).sum(1) > 0))
        sQ = np.where(beyond, np.inf, sQ)
        kk = np.repeat(ks, nB)
        Sin = np.minimum(S0[ray], _pair_query(tabs[i], ray, kown[kk]))
        for c in range(N_CH):
            if c != i and tabs[c] is not None:
                kc = np.searchsorted(samples[c]['T'], smp['T'][kk], side='left') - 1
                Sin = np.minimum(Sin, _pair_query(tabs[c], ray, kc))
        with np.errstate(invalid='ignore'):
            h = Sin - np.maximum(sQ, 0.0)
        nbt = np.tile(nb, (len(ks), 1))
        mold_face = (nbt * nd).sum(1) < 0.05
        # temas yayı: kalıba bakan VE ilerleme yönüne bakan (ön) yüz; arka yüz malzemeden ayrılır
        face = mold_face & ((nbt * smp['u'][kk]).sum(1) > 0.0)
        with np.errstate(invalid='ignore'):
            ok = (h > 0) & face & (sQ > -0.5) & np.isfinite(h)
        h = np.where(ok, h, 0.0)
        sh = (len(ks), nB)
        yield ks, ray.reshape(sh), sQ.reshape(sh), Sin.reshape(sh), h.reshape(sh), mold_face.reshape(sh)


def compute_wear(timelines, radii, mold_points, fin=None, env_limit=0.0, a=1.0, b=0.5, v_ref=1000.0, K=0.0,
                 n_bins=160):
    """Kanal başına makara profili boyunca birikimli aşınma ve zaman–konum ısı haritası."""
    E = _engage_setup(timelines, radii, mold_points, fin, env_limit)
    total_T = E['T']
    res = []
    for i in range(N_CH):
        smp = E['samples'][i]
        if smp is None:
            res.append(None)
            continue
        B, nb, sb = _boundary(radii[i])
        nB = len(B)
        n_bins = int(min(120, max(20, len(smp['C']) // 14)))     # kutu başına ≥ ~14 örnek: şeritlenme olmasın
        W = np.zeros(nB)
        H = np.zeros((n_bins, nB))
        line_w = {}
        hmax = 0.0
        for ks, ray, sQ, Sin, h, mface in _engage_chunks(E, i, B, nb):
            hmax = max(hmax, float(h.max()) if h.size else 0.0)
            vt = (smp['v'][ks] / v_ref) ** b if b else np.ones(len(ks))
            dW = (h ** a) * (vt * smp['ds'][ks])[:, None]
            W += dW.sum(0)
            bins = np.minimum((smp['T'][ks] / max(total_T, 1e-9) * n_bins).astype(int), n_bins - 1)
            np.add.at(H, bins, dW)
            for ln, w_ in zip(smp['line'][ks], dW.sum(1)):
                if w_ > 0:
                    line_w[int(ln)] = line_w.get(int(ln), 0.0) + float(w_)
        res.append({'B': B, 'nb': nb, 's': sb, 'W': W, 'H': H, 'lines': line_w, 'hmax': hmax,
                    'T': total_T, 'K': K, 'shape': tool_shape(radii[i]) is not None})
    return res, E['src']


def _prof_angle(B, W=None):
    """Profil noktalarının merkeze göre açısı (°). W verilirse açılar, aşınan bölge ±180° sınırında
    bölünmeyecek şekilde kaydırılır (en büyük boş aralığın ortasından kesilir)."""
    ang = np.degrees(np.arctan2(B[:, 1], B[:, 0]))
    if W is None or W.max() <= 0:
        return ang
    on = np.sort(ang[W > 0.01 * W.max()])
    if len(on) < 2:
        return ang
    gaps = np.diff(np.r_[on, on[0] + 360.0])
    k = int(np.argmax(gaps))
    cut = on[k] + gaps[k] / 2.0                       # bu açının karşısında aşınma yok
    return (ang - cut) % 360.0 + cut - 360.0 * (cut > 0)


def wear_figures(i, w, color):
    """(profil ısı figürü, zaman–konum ısı haritası, profil boyunca birikimli aşınma) → HTML"""
    B, nb, W, H = w['B'], w['nb'], w['W'], w['H']
    Wm = float(W.max()) if W.max() > 0 else 1.0
    rel = W / Wm * 100.0
    norm = mcolors.Normalize(0, 100)
    # 1) profil: kenar rengi = göreli aşınma, kesikli = abartılı aşınmış profil
    fig = Figure(figsize=(6.2, 5.2), dpi=100); FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    Bc = np.vstack([B, B[:1]])
    ax.fill(Bc[:, 0], Bc[:, 1], color='#f2f2f2', zorder=0)
    seg = np.stack([Bc[:-1], Bc[1:]], axis=1)
    lc = LineCollection(seg, cmap=WEAR_CMAP, norm=norm, linewidths=5, capstyle='round', zorder=2)
    lc.set_array(np.r_[rel])
    ax.add_collection(lc)
    reach = float(np.max(np.hypot(B[:, 0], B[:, 1])))
    ex = 0.12 * reach
    Wp = B - nb * (W / Wm * ex)[:, None]
    Wp = np.vstack([Wp, Wp[:1]])
    ax.plot(Wp[:, 0], Wp[:, 1], color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3,
            label=f"Aşınmış profil (abartılı: en çok aşınan yer {ex:.1f} mm içeri)")
    ax.plot(Bc[:, 0], Bc[:, 1], color=INK2, linewidth=0.6, zorder=1)
    km = int(np.argmax(W))
    ax.plot([0], [0], '+', color=INK, markersize=9, markeredgewidth=1.4)
    ax.plot([0, B[km, 0]], [0, B[km, 1]], color=INK2, linewidth=0.8, zorder=1)
    ang = _prof_angle(B, W)
    ax.annotate(f"en çok aşınma\n{ang[km]:.0f}°", B[km], xytext=(B[km] * 0.45), fontsize=8, color=INK,
                ha='center', arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    ax.set_aspect('equal', adjustable='box')
    pad = reach * 0.45
    ax.set_xlim(B[:, 0].min() - pad, B[:, 0].max() + pad); ax.set_ylim(B[:, 1].min() - pad, B[:, 1].max() + pad)
    ax.grid(True, color=GRID, linewidth=0.8)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK2)
    ax.set_title(f"Kanal {i + 1} — makara profili üzerinde göreli aşınma", fontsize=10, color=INK)
    ax.legend(loc='lower left', fontsize=7, frameon=False)
    cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Göreli aşınma (%; en çok aşınan yer = 100)', fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.tight_layout()
    h1 = fig_to_svg_card(fig)
    # 2) ısı haritası: yatay = profil üzerindeki açı (merkeze göre), dikey = zaman
    on = W > 0.01 * Wm
    o = np.argsort(ang)
    ang_s, H_s, rel_s = ang[o], H[:, o], rel[o]
    if on.any():
        lo_a, hi_a = float(ang[on].min()) - 10, float(ang[on].max()) + 10
    else:
        lo_a, hi_a = -180.0, 180.0
    sel = (ang_s >= lo_a) & (ang_s <= hi_a)
    fig2 = Figure(figsize=(7.2, 4.6), dpi=100); FigureCanvasAgg(fig2)
    ax2 = fig2.add_subplot(111)
    Hm = float(H_s.max()) if H_s.max() > 0 else 1.0
    if sel.sum() >= 2:
        im = ax2.pcolormesh(ang_s[sel], (np.arange(H.shape[0]) + 0.5) * w['T'] / H.shape[0], H_s[:, sel] / Hm * 100,
                            cmap=WEAR_CMAP, vmin=0, vmax=100, shading='nearest')
        cb2 = fig2.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
        cb2.set_label('Aşınma hızı (göreli, %)', fontsize=8, color=INK)
        cb2.ax.tick_params(labelsize=7, colors=INK2)
    ax2.set_xlabel('Makara profili üzerindeki konum — merkeze göre açı (°)', fontsize=9, color=INK)
    ax2.set_ylabel('Zaman (s)', fontsize=9, color=INK)
    ax2.set_title(f"Kanal {i + 1} — aşınmanın zaman ve profil konumuna göre dağılımı", fontsize=10, color=INK)
    ax2.tick_params(labelsize=8, colors=INK2)
    for sp in ('top', 'right'):
        ax2.spines[sp].set_visible(False)
    fig2.tight_layout()
    h2 = fig_to_svg_card(fig2)
    # 3) profil boyunca birikimli aşınma (tek seri)
    fig3 = Figure(figsize=(7.2, 2.8), dpi=100); FigureCanvasAgg(fig3)
    ax3 = fig3.add_subplot(111)
    ax3.plot(ang_s[sel], rel_s[sel], color='#eb6834', linewidth=2)
    ax3.set_xlabel('Makara profili üzerindeki konum — merkeze göre açı (°)', fontsize=9, color=INK)
    ax3.set_ylabel('Göreli aşınma (%)', fontsize=9, color=INK)
    ax3.set_title(f"Kanal {i + 1} — toplam aşınmanın profil boyunca dağılımı", fontsize=10, color=INK)
    ax3.grid(True, axis='y', color=GRID, linewidth=0.8)
    ax3.set_ylim(0, 105)
    for sp in ('top', 'right'):
        ax3.spines[sp].set_visible(False)
    ax3.tick_params(labelsize=8, colors=INK2)
    fig3.tight_layout()
    h3 = fig_to_svg_card(fig3)
    return h1 + h2 + h3


# -------------------------------------------------------------
# AŞINMIŞ MAKARA PROFİLİ → DXF (0,0 = makara merkezi; makara DXF'i olarak geri yüklenebilir)
# -------------------------------------------------------------
def worn_profile(tool_r, w, depth_max, step=0.25):
    """Orijinal profili (köşeleri koruyarak) sık örnekler, her noktayı aşınma derinliği kadar içeri alır.
    depth_max: bu kanalın en çok aşınan noktasındaki derinlik (mm). Döndürür: (orijinal, aşınmış) kapalı poligonlar"""
    O = tool_outline(tool_r, 720)
    if poly_area(O) < 0:
        O = O[::-1]
    E = np.diff(O, axis=0)
    L = np.hypot(E[:, 0], E[:, 1])
    keep = L > 1e-9
    O = np.vstack([O[:-1][keep], O[:1]]); E = np.diff(O, axis=0); L = np.hypot(E[:, 0], E[:, 1])
    en = np.c_[E[:, 1], -E[:, 0]] / L[:, None]                 # kenar dış normalleri (saat yönü tersine kontur)
    per = float(L.sum())
    cum = np.r_[0, np.cumsum(L)]
    W = w['W'] / max(float(w['W'].max()), 1e-30)
    Ws = (np.roll(W, 2) + np.roll(W, 1) + W + np.roll(W, -1) + np.roll(W, -2)) / 5.0      # basamakları yumuşat
    dfun = lambda sv: np.interp(sv, w['s'], Ws * depth_max, period=per)
    pts = []
    n = len(L)
    for j in range(n):
        vn = en[j] + en[j - 1]                                  # köşe normali: komşu kenarların ortalaması
        vn = vn / max(np.hypot(*vn), 1e-12)
        pts.append(O[j] - vn * dfun(cum[j]))
        m = int(np.ceil(L[j] / step))
        for q in range(1, m):
            u = q / m
            pts.append(O[j] + E[j] * u - en[j] * dfun(cum[j] + L[j] * u))
    P = np.asarray(pts)
    return O, np.vstack([P, P[:1]])


def wear_dxf_files(res, radii, depth_mm, K):
    """Kanal başına aşınmış profil DXF'i (yalnız aşınmış kontur, 0,0 merkez) + karşılaştırma DXF'i.
    Ölçek: depth_mm > 0 ise en çok aşınan kanalın en derin noktası depth_mm; değilse K·W (µm)."""
    act = [(i, w) for i, w in enumerate(res) if w is not None and w['W'].max() > 0]
    if not act:
        raise ValueError("Aşınma yok: makara malzemeye girmiyor.")
    gmax = max(float(w['W'].max()) for _i, w in act)
    if depth_mm and depth_mm > 0:
        scale, how = depth_mm / gmax, f"en büyük aşınma derinliği {depth_mm:g} mm (girilen)"
    elif K and K > 0:
        scale, how = K / 1000.0, f"K = {K:g} µm ölçeği (en büyük derinlik {K * gmax / 1000.0:.4g} mm)"
    else:
        raise ValueError("Aşınma derinliği ölçeği yok: 'en büyük aşınma derinliği (mm)' ya da K gir.")
    out_dir = tempfile.mkdtemp(prefix='cnc_wear_')
    paths, lines = [], []
    cmp_w = _DxfWriter()
    for i, w in act:
        dmax = float(w['W'].max()) * scale
        O, Pw = worn_profile(radii[i], w, dmax)
        wr = _DxfWriter()
        wr.layer('MAKARA_ASINMIS', 1)
        wr.polyline('MAKARA_ASINMIS', Pw[:-1], closed=True)
        path = os.path.join(out_dir, f'makara_asinmis_kanal{i + 1}.dxf')
        with open(path, 'w', encoding='ascii', errors='replace') as f:
            f.write(wr.text())
        paths.append(path)
        cmp_w.layer(f'K{i + 1}_MAKARA_ORIJINAL', 7); cmp_w.layer(f'K{i + 1}_MAKARA_ASINMIS', 1)
        cmp_w.polyline(f'K{i + 1}_MAKARA_ORIJINAL', O[:-1], closed=True)
        cmp_w.polyline(f'K{i + 1}_MAKARA_ASINMIS', Pw[:-1], closed=True)
        lines.append(f"Kanal {i + 1}: en derin aşınma {dmax:.4g} mm → makara_asinmis_kanal{i + 1}.dxf")
    pc = os.path.join(out_dir, 'makara_asinma_karsilastirma.dxf')
    with open(pc, 'w', encoding='ascii', errors='replace') as f:
        f.write(cmp_w.text())
    paths.append(pc)
    msg = ("Aşınmış makara profilleri hazır — ölçek: " + how + ".\n" + "\n".join(lines) +
           "\nKanal dosyaları yalnız aşınmış konturu içerir (0,0 = makara merkezi): 'Makara profili (DXF)' alanına "
           "yüklersen simülasyon aşınmış makarayla yapılır. Karşılaştırma dosyasında orijinal ve aşınmış profil ayrı katmanlarda.")
    return paths, msg


# -------------------------------------------------------------
# PROSES HESAPLARI (teori) — IUL "Shear Forming and Flow Forming" ders notundaki bağıntılar
#  Flow forming, Thamasett (slayt 7.20–7.21):
#     b = Δs·cot α            temas boyu (ilerleme yönünde)      → burada profil geometrisinden ölçülür
#     Δh = f·Δs / b           devir başına ezme
#     l_d ≈ √(r_m,CD · Δh)    çevresel temas boyu
#     F_r ≈ b · l_d · k_fm/η  radyal kuvvet   ( = √(r_m,CD·f·cot α)·Δs·k_fm/η ),  0,06 < η < 0,24
#     F_a ≈ F_r · tan β       eksenel kuvvet
#     F_t ≈ f · Δs · k_fm/η   teğetsel kuvvet → tork ve güç
#  Akış yönü (7.17): 1/r_m,CD = 1/(2 r_makara) + 1/(2 r_kalıp),  1/r_m,LD = 1/(2 ρ_w);
#     r_m,CD > r_m,LD ise akış ağırlıklı eksenel; değilse çevresel akış → çap büyümesi / yığılma riski
#  Shear forming (7.33–7.38): sinüs kuralı s1 = s0·sin α, φ_v = 1/(√3·tan α), Kalpakcioglu F_t = k_f/√3·f·s0·cos α
#  Makine esnemesi (ders notunda parametre olarak geçer; bağıntı burada eklenmiştir): boşluk F_r/c kadar açılır.
# -------------------------------------------------------------
PROC_COLORS = {'Fr': '#2a78d6', 'Fa': '#eb6834', 'Ft': '#1baf7a'}


def _roll_med(x, w=7):
    return pd.Series(x).rolling(w, center=True, min_periods=1).median().to_numpy()


def compute_process(timelines, radii, mold_points, fin, env_limit, ri, n_rpm, kfm, eta, D_tool, z=1,
                    beta_deg=0.0, c_stiff=0.0, shear=False):
    """Takım yolu boyunca kuvvet / güç / akış yönü / esneme. ri: parça yarıçapı ekseni indeksi (0 = X, 1 = Y).
    Döndürür: (kanal sonuçları, gelen yüzey kaynağı, E)"""
    E = _engage_setup(timelines, radii, mold_points, fin, env_limit)
    M, N = E['M'], E['N']
    rT = max(float(D_tool), 1e-6) / 2.0
    cN = float(c_stiff) * 1000.0                      # kN/mm → N/mm
    out = []
    for i in range(N_CH):
        smp = E['samples'][i]
        if smp is None:
            out.append(None)
            continue
        B, nb, _sb = _boundary(radii[i], 360)
        Kn = len(smp['C'])
        s1 = np.full(Kn, np.nan); s0 = np.full(Kn, np.nan); bw = np.zeros(Kn); tipray = np.zeros(Kn, int)
        for ks, ray, sQ, Sin, h, mface in _engage_chunks(E, i, B, nb):
            rows = np.arange(len(ks))
            g = np.where(mface & (sQ > -0.5), np.maximum(sQ, 0.0), np.inf)
            tip = np.argmin(g, axis=1)
            gmin = g[rows, tip]
            s1[ks] = np.where(np.isfinite(gmin), gmin, np.nan)
            s0k = Sin[rows, tip]
            s0[ks] = s0k
            proj = (B[None, :, :] * smp['u'][ks][:, None, :]).sum(2)
            cont = (h > 0) & (sQ <= s0k[:, None] + 1e-6)
            pmax = np.where(cont, proj, -np.inf).max(1)
            bw[ks] = np.where(np.isfinite(pmax), np.clip(pmax - proj[rows, tip], 0.0, None), 0.0)
            tipray[ks] = ray[rows, tip]
        ds = np.clip(np.nan_to_num(s0 - s1), 0.0, None)
        eng = (ds > 1e-3) & (bw > 1e-3) & np.isfinite(s1)
        f = smp['v'] / max(float(n_rpm) * max(int(z), 1), 1e-9)          # mm / devir (makara başına)
        rM = np.abs(M[tipray, ri])
        rmCD = 2.0 * rT * rM / np.maximum(rT + rM, 1e-9)
        rmLD = 2.0 * float(radii[i])
        with np.errstate(divide='ignore', invalid='ignore'):
            dh = np.where(eng, f * ds / np.maximum(bw, 1e-9), 0.0)
            ld = np.sqrt(rmCD * dh)
            Fr = np.where(eng, bw * ld * kfm / eta, 0.0)
            Ft = np.where(eng, f * ds * kfm / eta, 0.0)
            alpha = np.where(eng, np.arctan2(ds, np.maximum(bw, 1e-9)), 0.0)
            beta = np.radians(beta_deg) if beta_deg and beta_deg > 0 else np.minimum(alpha, np.pi / 4)
            Fa = Fr * np.tan(beta)
            # makine esnemesi: F ∝ Δs kabulüyle kapalı çözüm  δ = Δs·x/(1+x),  x = F_r/(c·Δs)
            x = np.where(eng & (cN > 0), Fr / np.maximum(cN * ds, 1e-9), 0.0)
            dlt = np.where(eng, ds * x / (1.0 + x), 0.0)
            eps = np.where(eng, ds / np.maximum(s0, 1e-9), 0.0)
            phi = np.where(eng & (s1 > 1e-6), np.log(np.maximum(s0, 1e-9) / np.maximum(s1, 1e-9)), 0.0)
        rc = rM + np.nan_to_num(s1)
        P = Ft * 2 * np.pi * (rc / 1000.0) * float(n_rpm) / 60.0 * max(int(z), 1)      # W (tüm makaralar)
        r = {'t': smp['T'], 'line': smp['line'], 'eng': eng, 's0': s0, 's1': s1, 'ds': ds, 'b': bw, 'f': f,
             'Fr': _roll_med(Fr), 'Fa': _roll_med(Fa), 'Ft': _roll_med(Ft), 'P': _roll_med(P), 'dlt': _roll_med(dlt),
             'alpha': alpha, 'eps': eps, 'phi': phi, 'flow': rmCD / max(rmLD, 1e-9), 'rM': rM, 'tipray': tipray,
             'Mt': _roll_med(Ft * rc / 1000.0), 'z': max(int(z), 1)}
        if shear and fin is not None and len(fin.get('T', [])) == len(M):
            sinA = np.abs(N[tipray, 1 - ri]); cosA = np.abs(N[tipray, ri])
            t0 = np.nan_to_num(np.asarray(fin['T'], float)[tipray])
            with np.errstate(divide='ignore', invalid='ignore'):
                r['sine'] = np.where(eng & (t0 * sinA > 1e-6), s1 / (t0 * sinA), np.nan)     # 1 = sinüs kuralı
                r['FtK'] = np.where(eng, kfm / np.sqrt(3.0) * f * t0 * cosA, 0.0)
                r['phiV'] = np.where(eng & (sinA > 1e-6), cosA / (np.sqrt(3.0) * sinA), np.nan)
        out.append(r)
    return out, E['src'], E


def process_deflection_rays(E, res):
    """Işın başına esneme payı δ: o ışında en küçük boşluğu veren (son şekli belirleyen) takım örneğinden."""
    nM = len(E['M'])
    best = np.full(nM, np.inf); d = np.full(nM, np.nan)
    for r in res:
        if r is None:
            continue
        for k in np.where(r['eng'])[0]:
            j = r['tipray'][k]
            if r['s1'][k] < best[j]:
                best[j], d[j] = r['s1'][k], r['dlt'][k]
    have = np.where(np.isfinite(d))[0]
    if len(have) == 0:
        return np.zeros(nM)
    return np.interp(np.arange(nM), have, d[have])


def process_figures(i, r, c_stiff):
    """Kanal i için: kuvvetler, kalınlık (gelen / programlanan / esnemeli), güç."""
    t = r['t']
    te = t[r['eng']]
    xlim = (max(float(te.min()) - 1.0, float(t.min())), min(float(te.max()) + 1.0, float(t.max()))) if len(te) else None
    def base(title, ylab, h=3.0):
        fig = Figure(figsize=(7.4, h), dpi=100); FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=10, color=INK); ax.set_xlabel('Zaman (s)', fontsize=9, color=INK)
        ax.set_ylabel(ylab, fontsize=9, color=INK); ax.grid(True, axis='y', color=GRID, linewidth=0.8)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK2)
        if xlim:
            ax.set_xlim(*xlim)
        return fig, ax
    fig, ax = base(f"Kanal {i + 1} — şekillendirme kuvvetleri (Thamasett), makara başına", 'Kuvvet (kN)')
    for key, lab in (('Fr', 'F_r radyal'), ('Fa', 'F_a eksenel'), ('Ft', 'F_t teğetsel')):
        y = r[key] / 1000.0
        ax.plot(t, y, color=PROC_COLORS[key], linewidth=2, label=lab)
        k = int(np.argmax(y))
        if y[k] > 0:
            ax.annotate(f"{y[k]:.1f}", (t[k], y[k]), xytext=(0, 4), textcoords='offset points', ha='center',
                        fontsize=8, color=INK)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22); ax.legend(fontsize=8, frameon=False, loc='upper left', ncol=3)
    fig.tight_layout(); h1 = fig_to_svg_card(fig)
    fig, ax = base(f"Kanal {i + 1} — makara altındaki kalınlık", 'Kalınlık (mm)')
    e = r['eng']
    s0 = np.where(e, r['s0'], np.nan); s1 = np.where(e, r['s1'], np.nan)
    ax.plot(t, s0, color=INK2, linewidth=1.4, label='Gelen kalınlık s₀')
    ax.plot(t, s1, color='#2a78d6', linewidth=2, label='Programlanan boşluk s₁')
    if c_stiff and c_stiff > 0:
        ax.plot(t, s1 + np.where(e, r['dlt'], np.nan), color='#eb6834', linewidth=2, label='Esnemeyle tahmini s₁')
    ax.set_ylim(bottom=0); ax.legend(fontsize=8, frameon=False, loc='lower left', ncol=3)
    fig.tight_layout(); h2 = fig_to_svg_card(fig)
    fig, ax = base(f"Kanal {i + 1} — şekillendirme gücü ({r['z']} makara toplamı)", 'Güç (kW)', 2.6)
    ax.plot(t, r['P'] / 1000.0, color='#2a78d6', linewidth=2)
    ax.set_ylim(bottom=0)
    fig.tight_layout(); h3 = fig_to_svg_card(fig)
    return h1 + h2 + h3


# ===== Hücre 20 =====
# -------------------------------------------------------------
# ANİMASYON SAHNESİ (canlı oynatıcı ve MP4 aynı kodu kullanır)
# -------------------------------------------------------------
def channel_active(segs):
    """Kanalın programında gerçek bir hareket / bekleme var mı (boş kanallar çizime ve sınırlara girmez)."""
    return any(s.get('type') in ('move', 'dwell') for s in segs)


def _scene_bounds(channels_segments, start_points, radii, mold_points, extra=None):
    all_x, all_y = [p[0] for p in mold_points], [p[1] for p in mold_points]
    act = [channel_active(s) for s in channels_segments]
    for P in (extra or []):                                  # ön form / ürün
        P = np.asarray(P, float)
        if len(P):
            all_x.extend(P[:, 0]); all_y.extend(P[:, 1])
    for segs, (sx, sy), a in zip(channels_segments, start_points, act):
        if not a:
            continue
        all_x.append(sx); all_y.append(sy)
        for seg in segs:
            if seg['type'] == 'move':
                all_x.extend(p[0] for p in seg['orig_points'])
                all_y.extend(p[1] for p in seg['orig_points'])
    ra = [tool_reach(r) for r, a in zip(radii, act) if a] or [0.0]
    margin = max(10.0, max(ra) * 2)
    return min(all_x) - margin, max(all_x) + margin, min(all_y) - margin, max(all_y) + margin


def build_scene(channels_segments, start_points, radii, mold_points, xname='X', yname='Y'):
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
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    if len(mold_points) > 1:
        ax.plot([p[0] for p in mold_points], [p[1] for p in mold_points],
                color='black', linewidth=1.0, linestyle='-', label='Kalıp (sabit)')

    sc = {'fig': fig, 'canvas': canvas, 'starts': list(start_points), 'contact_lines': [], 'tool_circles': [],
          'spoke_lines': [], 'contact_dots': [], 'tool_centers': []}
    for i in range(N_CH):
        (cline,) = ax.plot([], [], color=CHANNEL_COLORS[i], linewidth=2.2, label=f'Kanal {i+1} — kontur')
        _ol = tool_outline(radii[i])
        circ = MplPolygon(_ol, closed=True, facecolor=CHANNEL_COLORS[i], alpha=0.22,
                          edgecolor=CHANNEL_COLORS[i], linewidth=1.8)
        circ._tool_ol = _ol
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
        if timelines[i]:
            center_pos = get_position(timelines[i], t, 'points')
            contact_pos = get_position(timelines[i], t, 'orig_points')
        else:                                       # hareketsiz kanal: başlangıç noktasında dursun
            center_pos = contact_pos = tuple(sc['starts'][i])
        trail = get_trail(timelines[i], t, 'orig_points')
        if trail:
            sc['contact_lines'][i].set_data([p[0] for p in trail], [p[1] for p in trail])
        else:
            sc['contact_lines'][i].set_data([], [])
        sc['tool_circles'][i].set_xy(sc['tool_circles'][i]._tool_ol + np.asarray(center_pos, float))
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


def register_anim(channels_segments, start_points, radii, mold_points, timelines, total_time, fps, speed,
                  xname='X', yname='Y'):
    sc = build_scene(channels_segments, start_points, radii, mold_points, xname, yname)
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


def mp4_generator(channels_segments, start_points, radii, mold_points, timelines, total_time, fps, path,
                  xname='X', yname='Y'):
    """Her ~5 karede bir (kare_no, toplam) üretir; bittiğinde dosya yazılmış olur."""
    sc = build_scene(channels_segments, start_points, radii, mold_points, xname, yname)
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
def config_to_json_dict(cfg, fps, speed, ac_start, ac_end, ac_step, opts, preform=None, tools=None):
    data = {'channels': [], 'mold': dict(cfg['mold']),
            'settings': {'contact_mode': cfg['contact_mode'], 'g0_feed': cfg['g0_feed'], 'fps': fps, 'speed': speed},
            'autocad': {'start': ac_start, 'end': ac_end, 'step': ac_step},
            'plot': {'label_font': opts['label_font'], 'label_shift': opts['label_shift'],
                     'xname': opts['xname'], 'yname': opts['yname'],
                     'show_env': opts['show_env'], 'env_limit': opts['env_limit']},
            'preform': {'use': opts['pf_use'], 'dx': opts['pf_dx'], 'dy': opts['pf_dy'],
                        'model': opts['pf_model'], 'radial': opts['pf_radial'], 'sim': opts['pf_sim'],
                        'selected': opts['pf_layers'],
                        'name': (preform or {}).get('name'),
                        'layers': {k: [[[round(x, 4), round(y, 4)] for x, y in pl] for pl in v]
                                   for k, v in ((preform or {}).get('layers') or {}).items()}}}
    for idx, ch in enumerate(cfg['channels']):
        data['channels'].append({'channel_id': idx + 1, 'x0': ch['x0'], 'y0': ch['y0'], 'tr': ch['tr'],
                                 'fd': ch['fd'], 'y_offset': ch['y_offset'], 'code': ch['code'],
                                 'dia_mode': ch['dia_mode'], 'dia_axis': ch['dia_axis']})
        te = (tools or {}).get(str(idx))
        if te and te.get('shape'):
            data['channels'][-1]['tool_dxf'] = {'name': te.get('name'), 'shape': te['shape']}
    return data


def json_dict_to_updates(data, gr_update):
    """JSON sözlüğünü bileşen güncellemelerine çevirir. Sıra, arayüzdeki all_inputs ile aynıdır:
    N_CFG cfg + fps + speed + 3 DXF çember zamanı + N_OPT grafik seçeneği. JSON'da olmayan alanlar mevcut değerini korur."""
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
    ups.append(pick(st, 'g0_feed'))
    ups.append(pick(st, 'fps'))
    ups.append(pick(st, 'speed'))
    ac = data.get('autocad', {}) or {}
    for key in ('start', 'end', 'step'):
        ups.append(pick(ac, key))
    pl = data.get('plot', {}) or {}
    for key in ('label_font', 'label_shift', 'xname', 'yname', 'show_env', 'env_limit'):
        ups.append(pick(pl, key))
    pf = data.get('preform', {}) or {}
    for key in ('use', 'dx', 'dy'):
        ups.append(pick(pf, key))
    if pf.get('layers'):
        st = {'name': pf.get('name') or 'ön form', 'layers': pf['layers']}
        sel = [k for k in (pf.get('selected') or []) if k in st['layers']] or list(st['layers'])
        ups.append(gr_update(choices=preform_layer_choices(st), value=sel))
    else:
        ups.append(gr_update())
    for key in ('model', 'radial', 'sim'):
        ups.append(pick(pf, key))
    if pf.get('layers'):
        ups.append(st)                                  # gr.State: değer doğrudan verilir
    else:
        ups.append(gr_update())                         # State değişmez
    # makara profilleri: JSON'da kanal bilgisi varsa profiller JSON'daki gibi olur (olmayan kanal: daire)
    if chans:
        ups.append({str(k): chans[k]['tool_dxf'] for k in range(min(N_CH, len(chans)))
                    if isinstance(chans[k], dict) and chans[k].get('tool_dxf')})
    else:
        ups.append(gr_update())
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
button.rec{background:#c0392b;color:#fff;border-color:#c0392b}
input[type=range]{vertical-align:middle}
#main{flex:1 1 auto;min-height:0;display:flex}
#cwrap{flex:1 1 auto;min-width:0;position:relative}
#cv{position:absolute;left:0;top:0;width:100%;height:100%}
#side{width:290px;flex:0 0 290px;overflow:auto;border-left:1px solid #ddd;padding:10px;box-sizing:border-box;font-size:12px;background:#fafafa}
#side .ttl{font-weight:bold;font-size:14px;margin-bottom:2px}
#side .sub{color:#666;font-size:11px;margin-bottom:8px}
#dt{width:100%;border-collapse:collapse;margin-bottom:10px}
#dt th{font-size:11px;color:#555;text-align:right;padding:2px 3px;border-bottom:1px solid #ccc}
#dt th:first-child,#dt td:first-child{text-align:left}
#dt td{padding:5px 3px;text-align:right;border-bottom:1px solid #eee;font-variant-numeric:tabular-nums;vertical-align:top}
#dt .cur{font-size:16px;font-weight:bold}
#dt .tt{display:block;color:#888;font-size:10px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}
#measb{width:100%;margin-bottom:6px}
#mstat{font-size:12px;line-height:1.4}
#tm{flex:1;min-width:140px}
#cncrow{flex:0 0 auto;display:flex;gap:8px;padding:8px;background:#1b1f24;border-top:1px solid #ddd;overflow-x:auto}
.cnc{flex:1 1 0;min-width:290px;background:#0d1117;color:#d8dee9;border-radius:6px;padding:6px 8px;font-family:Consolas,'Courier New',monospace;font-size:12px}
.cnc .hdr{display:flex;justify-content:space-between;gap:8px;font-weight:bold;border-bottom:1px solid #2d333b;padding-bottom:3px;margin-bottom:4px}
.cnc .hdr .st{color:#f2b600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cnc .body{display:flex;gap:10px}
.cnc .prog{flex:1 1 auto;min-width:0}
.cnc .ln{white-space:pre;overflow:hidden;text-overflow:ellipsis;padding:1px 3px;border-radius:2px;color:#8b949e;min-height:15px}
.cnc .ln.cur{background:#f2b600;color:#000;font-weight:bold}
.cnc .pos{flex:0 0 auto}
.cnc table{border-collapse:collapse}
.cnc th{font-size:10px;color:#8b949e;font-weight:normal;text-align:right;padding:0 5px}
.cnc td{padding:1px 5px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.cnc td.k{text-align:left;color:#8b949e}
.cnc td.big{font-size:15px;font-weight:bold;color:#e6edf3}
@media (max-width:760px){#cncrow{flex-direction:column}.cnc{min-width:0}}
@media (max-width:760px){#main{flex-direction:column}#side{width:auto;flex:0 0 auto;max-height:46%;border-left:0;border-top:1px solid #ddd}}
</style></head><body>
<div class="bar">
 <button id="play" class="main">▶ Oynat</button><button id="rst">⟲ Başa</button>
 <span>Hız:</span><input id="spd" type="range" min="0.1" max="30" step="0.1" value="1" style="width:150px">
 <b id="spdv" style="min-width:46px">1.0×</b>
 <span id="presets"></span>
 <label><input type="checkbox" id="loop" checked> Döngü</label>
 <label><input type="checkbox" id="envc"> Radüs izi</label>
 <button id="fs">⛶ Tam Ekran</button>
</div>
<div class="bar"><input id="tm" type="range" min="0" max="10000" value="0"><b id="tv"></b></div>
<div id="main">
 <div id="cwrap"><canvas id="cv"></canvas></div>
 <div id="side">
  <div class="ttl">Takım – Kalıp Mesafesi</div>
  <div class="sub">Temas noktası → kalıp konturu (mm), canlı</div>
  <table id="dt"><thead><tr><th>Kanal</th><th>Anlık</th><th>Min</th><th>Maks</th></tr></thead><tbody id="dtb"></tbody></table>
  <button id="measb" class="main">📏 Ölçümü Başlat</button>
  <div id="mstat">Ölçüm yapılmadı. “Ölçümü Başlat” ile başlat, “Ölçümü Bitir” ile sabitle.</div>
  <div id="simstat" style="display:none;margin-top:12px;padding-top:8px;border-top:1px solid #ddd;font-size:12px;line-height:1.5"></div>
 </div>
</div>
<div id="cncrow"></div>
<script>
(function(){
var D=__DATA__;
var cv=document.getElementById('cv'), ctx=cv.getContext('2d');
var playB=document.getElementById('play'), rstB=document.getElementById('rst'), spd=document.getElementById('spd'),
    spdv=document.getElementById('spdv'), loopC=document.getElementById('loop'), fsB=document.getElementById('fs'),
    tm=document.getElementById('tm'), tv=document.getElementById('tv'), pre=document.getElementById('presets'),
    measB=document.getElementById('measb'), mstat=document.getElementById('mstat'), dtb=document.getElementById('dtb');
var t=0, playing=false, last=0, speed=Math.min(Math.max(D.speed||1,0.1),30), W=0, H=0;
var xmin=D.b[0], xmax=D.b[1], ymin=D.b[2], ymax=D.b[3], NC=D.ch.length;
function hex2rgba(h,a){var n=parseInt(h.slice(1),16);return 'rgba('+(n>>16&255)+','+(n>>8&255)+','+(n&255)+','+a+')';}
function nice(r){var e=Math.pow(10,Math.floor(Math.log10(r))), f=r/e; return (f<1.5?1:f<3.5?2:f<7.5?5:10)*e;}
function lerp(a,k,f){return a[k]+(a[k+1]-a[k])*f;}
function fmt(v){return (typeof v==='number' && isFinite(v))?v.toFixed(3):'—';}
function fmtT(v){return v.toFixed(2);}

// ---- kalıp mesafesi ----
function dpt(x,y){
  var M=D.mold; if(M.length<2) return NaN; var best=1e18;
  for(var i=0;i<M.length-1;i++){
    var ax=M[i][0], ay=M[i][1], bx=M[i+1][0]-ax, by=M[i+1][1]-ay, l2=bx*bx+by*by;
    var q=l2>0?((x-ax)*bx+(y-ay)*by)/l2:0; q=q<0?0:(q>1?1:q);
    var dx=ax+q*bx-x, dy=ay+q*by-y, dd=dx*dx+dy*dy; if(dd<best) best=dd;}
  return Math.sqrt(best);
}
function pos(i,key,k,f){var c=D.ch[i]; return [lerp(c[key[0]],k,f), lerp(c[key[1]],k,f)];}
function kf(){var u=t/D.dt, k=Math.min(Math.floor(u),D.n-1), f=u-k; if(k<0){k=0;f=0;} return [k,f];}
function curDist(i){var q=kf(), p=pos(i,['px','py'],q[0],q[1]); return dpt(p[0],p[1]);}

var meas={on:false,has:false,last:0,tlo:0,thi:0,mn:[],mx:[],tmn:[],tmx:[]};
function measReset(){meas.mn=[];meas.mx=[];meas.tmn=[];meas.tmx=[]; for(var i=0;i<NC;i++){meas.mn.push(Infinity);meas.mx.push(-Infinity);meas.tmn.push(0);meas.tmx.push(0);}}
function measPush(i,v,tt){ if(!isFinite(v)||(D.act&&!D.act[i])) return; if(v<meas.mn[i]){meas.mn[i]=v;meas.tmn[i]=tt;} if(v>meas.mx[i]){meas.mx[i]=v;meas.tmx[i]=tt;} }
function measStep(){
  if(!meas.on) return;
  var lo=Math.min(meas.last,t), hi=Math.max(meas.last,t);
  var k0=Math.max(0,Math.floor(lo/D.dt)), k1=Math.min(D.n-1,Math.floor(hi/D.dt));
  for(var k=k0;k<=k1;k++){                       // aralıktaki tüm örnekler + aralarında 3 ara adım (hız farketmez)
    for(var s=0;s<4;s++){ var tt=(k+s/4)*D.dt; if(tt<lo-1e-9||tt>hi+1e-9) continue;
      for(var i=0;i<NC;i++){ var c=D.ch[i]; measPush(i,dpt(lerp(c.px,k,s/4),lerp(c.py,k,s/4)),tt); } } }
  for(var i=0;i<NC;i++) measPush(i,curDist(i),t);
  meas.last=t; meas.tlo=Math.min(meas.tlo,t); meas.thi=Math.max(meas.thi,t);
}
function startMeas(){ measReset(); meas.on=true; meas.has=false; meas.last=t; meas.tlo=t; meas.thi=t;
  for(var i=0;i<NC;i++) measPush(i,curDist(i),t);
  measB.textContent='⏹ Ölçümü Bitir'; measB.className='rec'; ui(); }
function stopMeas(why){ if(!meas.on) return; measStep(); meas.on=false; meas.has=true;
  measB.textContent='📏 Ölçümü Başlat'; measB.className='main'; meas.why=why||''; ui(); }
measB.onclick=function(){ if(meas.on) stopMeas(''); else startMeas(); };

// tablo satırları
for(var qi=0;qi<NC;qi++){
  var tr=document.createElement('tr');
  tr.innerHTML='<td><span class="sw" style="background:'+D.col[qi]+'"></span>K'+(qi+1)+'</td>'+
    '<td class="cur" id="cd'+qi+'">—</td><td id="mn'+qi+'">—</td><td id="mx'+qi+'">—</td>';
  dtb.appendChild(tr);
}
measReset();
function cellHtml(v,tt){ return isFinite(v)?(fmt(v)+'<span class="tt">@'+fmtT(tt)+' s</span>'):'—'; }
function updatePanel(){
  for(var i=0;i<NC;i++){
    document.getElementById('cd'+i).textContent=(D.act&&!D.act[i])?'boş':fmt(curDist(i));
    document.getElementById('mn'+i).innerHTML=cellHtml(meas.mn[i],meas.tmn[i]);
    document.getElementById('mx'+i).innerHTML=cellHtml(meas.mx[i],meas.tmx[i]);
  }
  if(meas.on){ mstat.innerHTML='<span style="color:#c0392b;font-weight:bold">● Ölçüm sürüyor</span><br>Aralık: '+fmtT(meas.tlo)+' – '+fmtT(meas.thi)+' s'; }
  else if(meas.has){ mstat.innerHTML='<span style="color:#1e8449;font-weight:bold">✔ Ölçüm tamamlandı (sabitlendi)</span><br>Aralık: '+fmtT(meas.tlo)+' – '+fmtT(meas.thi)+' s'+(meas.why?'<br><i>'+meas.why+'</i>':''); }
}

// ---- RADÜS İZİ (form yüzeyi): kalıp noktalarından normal yönünde ışın, takım dairesine ilk giriş ----
var EV=D.env||null, envC=document.getElementById('envc'), NM=0, CP=(D.shp&&D.shp.some(function(v){return !!v;}))?64:256, Mx,My,Nx,Ny, envCk=[], envCache=[];
if(EV){
  envC.checked=!!EV.show; NM=EV.M.length/2;
  Mx=new Float64Array(NM); My=new Float64Array(NM); Nx=new Float64Array(NM); Ny=new Float64Array(NM);
  for(var q0=0;q0<NM;q0++){Mx[q0]=EV.M[2*q0];My[q0]=EV.M[2*q0+1];Nx[q0]=EV.N[2*q0];Ny[q0]=EV.N[2*q0+1];}
} else { envC.checked=false; envC.parentNode.style.display='none'; }
envC.onchange=function(){draw();};
function polyRayS(ox,oy,nx,ny,P){   // Python _poly_ray_s: içerideyse 0, değilse doğrultudaki en yakın kesişim
  var best=Infinity, cnt=0, n=P.length/2-1;
  for(var e=0;e<n;e++){ var ax=P[2*e]-ox, ay=P[2*e+1]-oy, dx=P[2*e+2]-P[2*e], dy=P[2*e+3]-P[2*e+1], den=nx*dy-ny*dx;
    if(Math.abs(den)<=1e-12) continue; var sv=(ax*dy-ay*dx)/den, tv=(ax*ny-ay*nx)/den;
    if(tv<0||tv>=1) continue; if(sv>0) cnt++; if(Math.abs(sv)<Math.abs(best)) best=sv; }
  return (cnt%2===1)?0:best; }
function applySamples(arr,ch,a,b){
  var SH=D.shp?D.shp[ch]:null;
  if(SH){ var C0=EV.S[ch].c, RR0=(D.rch[ch]+EV.lim[ch])*(D.rch[ch]+EV.lim[ch]);
    for(var k=a;k<b;k++){ var cx=C0[2*k], cy=C0[2*k+1];
      for(var i=0;i<NM;i++){ var dx=Mx[i]-cx, dy=My[i]-cy; if(dx*dx+dy*dy>RR0) continue;
        var s0=polyRayS(dx,dy,Nx[i],Ny[i],SH); if(Math.abs(s0)<Math.abs(arr[i])) arr[i]=s0; } }
    return; }
  var C=EV.S[ch].c, r2=D.r[ch]*D.r[ch], RR=(D.r[ch]+EV.lim[ch])*(D.r[ch]+EV.lim[ch]);
  for(var k=a;k<b;k++){ var cx=C[2*k], cy=C[2*k+1];
    for(var i=0;i<NM;i++){ var dx=Mx[i]-cx, dy=My[i]-cy, d2=dx*dx+dy*dy; if(d2>RR) continue;
      var ct=d2-r2, s; if(ct<=0){s=0;} else { var bb=Nx[i]*dx+Ny[i]*dy, disc=bb*bb-ct; if(disc<0) continue;
        var sq=Math.sqrt(disc); s=bb>0?(-bb+sq):(-bb-sq); }
      if(Math.abs(s)<Math.abs(arr[i])) arr[i]=s; } }
}
if(EV){ for(var ch0=0;ch0<NC;ch0++){ var K=EV.S[ch0].t.length, arr0=new Float64Array(NM).fill(Infinity), cps=[arr0.slice()];
  for(var k0=0;k0<K;k0+=CP){ applySamples(arr0,ch0,k0,Math.min(K,k0+CP)); cps.push(arr0.slice()); }
  envCk.push(cps); envCache.push({j:-1,a:null}); } }
function upperBound(A,x){var lo=0,hi=A.length; while(lo<hi){var m=(lo+hi)>>1; if(A[m]<=x) lo=m+1; else hi=m;} return lo;}
function envAt(ch){ var j=upperBound(EV.S[ch].t,t+1e-9), c=Math.floor(j/CP), ca=envCache[ch];
  if(ca.j===j) return ca.a; var a=envCk[ch][c].slice(); applySamples(a,ch,c*CP,j); ca.j=j; ca.a=a; return a; }
// ön form (varsa): ışın başına lokal malzeme kalınlığı (T) ve malzemenin olduğu taraf (SD)
var HASPF=!!(EV&&EV.T), STRIP=!!(EV&&EV.mode==='strip'), SHEAR=!!(EV&&(EV.mode==='shear'||EV.mode==='strip')), Ta=null, SD=null, TI=null;
if(HASPF){ Ta=new Float64Array(EV.T); SD=new Int8Array(EV.sd); if(SHEAR) TI=new Float64Array(EV.Ti); }
// dinamik form simülasyonu (hacim sabit)
var SM=(EV&&EV.sim)?EV.sim:null, SIMD=!!SM, SW0=null, SDS=null, SDSS=null, SIMP=null, SIMEND=0;
var STS=(SIMD&&SM.kind==='strip'), SPA=null, SPB=null, SSIG=null, SV=null, STAU=null, SGD=null, SSR=null;
if(STS){ SPA=[]; SPB=[]; for(var q8=0;q8<SM.pax.length;q8++){ SPA.push([SM.pax[q8],SM.pay[q8]]); SPB.push([SM.pbx[q8],SM.pby[q8]]); }
  SSIG=SM.sig; SV=SM.V; STAU=new Float64Array(SM.tau); SSR=new Float64Array(SM.sigR); }
if(SIMD && !STS){ SW0=new Float64Array(SM.W0); SDS=new Int8Array(SM.sd); SDSS=new Float64Array(SM.ds); SIMP=[];
  for(var q9=0;q9<SM.poly.length;q9+=2) SIMP.push([SM.poly[q9],SM.poly[q9+1]]);
  SIMEND=SIMP.reduce(function(a,p){return SM.dir>0?Math.max(a,p[SM.k]):Math.min(a,p[SM.k]);}, SM.dir>0?-Infinity:Infinity); }
if(STS){ SW0=new Float64Array(SM.W0); SDS=new Int8Array(SM.sd); SDSS=new Float64Array(SM.dsc); }
function okv(i,v,L){
  if(!isFinite(v)) return false;
  if(SIMD){ if(v!==0 && ((v>0)!==(SDS[i]>0))) return false; if(SM.tR && !(SM.tR[i]>=0 && SM.tR[i]<=t+1e-9)) return false;
    return Math.abs(v)<=(SM.kind==='strip'?SM.thr:(SM.lim?SM.lim[i]:SM.tmax))+1e-9; }
  if(STRIP){ if(v!==0 && ((v>0)!==(SD[i]>0))) return false; return Math.abs(v)<=EV.thr+1e-9; }
  if(HASPF){ var T=Ta[i]; if(!(T>0)) return false; if(v!==0 && ((v>0)!==(SD[i]>0))) return false;
    return SHEAR ? Math.abs(v)<=T+1e-9 : Math.abs(v)<T-1e-6; }
  return Math.abs(v)<=L;
}
function runs(ok,px,py,brk){ var out=[], cur=[];
  for(var i=0;i<NM;i++){ if(!ok(i)){ if(cur.length>1) out.push(cur); cur=[]; continue; }
    if(cur.length){ var j=cur[cur.length-1]; if(Math.hypot(px(i)-px(j),py(i)-py(j))>brk){ if(cur.length>1) out.push(cur); cur=[]; } }
    cur.push(i); }
  if(cur.length>1) out.push(cur); return out; }
function polyPath(run,fx,fy,X,Y,back){ for(var q=0;q<run.length;q++){var i=run[q]; if(q) ctx.lineTo(X(fx(i)),Y(fy(i))); else ctx.moveTo(X(fx(i)),Y(fy(i)));}
  if(back){ for(var q2=run.length-1;q2>=0;q2--){var i2=run[q2]; ctx.lineTo(X(back[0](i2)),Y(back[1](i2)));} ctx.closePath(); } }
function simInterp(Vq,A){ var V=SM.V, n=V.length;
  if(Vq<=V[0]) return A[0]; if(Vq>=V[n-1]) return A[n-1];
  var j=upperBound(V,Vq), i0=j-1, f=(V[j]>V[i0])?(Vq-V[i0])/(V[j]-V[i0]):0; return A[i0]+(A[j]-A[i0])*f; }
function remapV(p,src,tgt){ var ri=SM.ri, ax=1-ri, o=[p[0],p[1]]; o[ax]+=tgt[ax]-src[ax];
  var d=tgt[ri]*tgt[ri]-src[ri]*src[ri], r=o[ri], sg=(r<0)?-1:1; o[ri]=sg*Math.sqrt(Math.max(r*r+d,0)); return o; }
function clipHalf(P,k,c,d){ var out=[], n=P.length; if(n>1&&P[0][0]===P[n-1][0]&&P[0][1]===P[n-1][1]) n--;
  for(var i=0;i<n;i++){ var cur=P[i], prev=P[(i-1+n)%n], cin=d*(cur[k]-c)>=0, pin=d*(prev[k]-c)>=0;
    if(cin!==pin){ var tt=(c-prev[k])/(cur[k]-prev[k]); out.push([prev[0]+tt*(cur[0]-prev[0]), prev[1]+tt*(cur[1]-prev[1])]); }
    if(cin) out.push(cur); }
  return out.length>=3?out:null; }
function simVat(c){ var cs=SM.c, V=SM.V, d=SM.dir, n=cs.length, q=d*c;
  if(q<=d*cs[0]) return V[0]; if(q>=d*cs[n-1]) return V[n-1];
  var lo=0, hi=n-1; while(hi-lo>1){ var m=(lo+hi)>>1; if(d*cs[m]<=q) lo=m; else hi=m; }
  var x0=d*cs[lo], x1=d*cs[hi], f=(x1>x0)?(q-x0)/(x1-x0):0; return V[lo]+(V[hi]-V[lo])*f; }
function nearRay(p){ var best=Infinity, k=0; for(var i=0;i<NM;i++){ var dx=p[0]-Mx[i], dy=p[1]-My[i], d2=dx*dx+dy*dy; if(d2<best){best=d2;k=i;} } return k; }
function moldPen(p){ var k=nearRay(p), L=NM-1;     // Python _mold_pen: kalıp uçlarının ötesi serbest
  if(k===L && (p[0]-Mx[L])*(Mx[L]-Mx[L-1])+(p[1]-My[L])*(My[L]-My[L-1])>0) return [Infinity,k];
  if(k===0 && (p[0]-Mx[0])*(Mx[0]-Mx[1])+(p[1]-My[0])*(My[0]-My[1])>0) return [Infinity,k];
  return [(p[0]-Mx[k])*Nx[k]*SDS[k]+(p[1]-My[k])*Ny[k]*SDS[k], k]; }
function moldPenMin(P){ var L=NM-1, mn=Infinity;   // Python _mold_pen_all(...).min()
  for(var q=0;q<P.length;q++){ var p=P[q], sg=moldSigned(p);
    var dE=Math.hypot(p[0]-Mx[L],p[1]-My[L]), dS=Math.hypot(p[0]-Mx[0],p[1]-My[0]);
    var bE=((p[0]-Mx[L])*(Mx[L]-Mx[L-1])+(p[1]-My[L])*(My[L]-My[L-1])>0) && dE<=dS;
    var bS=((p[0]-Mx[0])*(Mx[0]-Mx[1])+(p[1]-My[0])*(My[0]-My[1])>0) && dS<dE;
    if(!(bE||bS) && sg<mn) mn=sg; }
  return mn; }
function polyVol(P){ var ri=SM.ri, A=0, cx=0, cy=0, n=P.length;   // Python poly_volume (kapalı poligon)
  for(var i=0;i<n-1;i++){ var cr=P[i][0]*P[i+1][1]-P[i+1][0]*P[i][1]; A+=cr; cx+=(P[i][0]+P[i+1][0])*cr; cy+=(P[i][1]+P[i+1][1])*cr; }
  A*=0.5; if(Math.abs(A)<1e-12) return 0; cx/=(6*A); cy/=(6*A); return 2*Math.PI*Math.abs(A)*Math.abs(ri===1?cy:cx); }
function bendTail(cc, useLo, src, outer, tgt, Vtarget, remP, outP){   // Python _bend_tail
  if(!remP || remP.length<3) return [remP,outP];
  if(moldPenMin(remP)>=-0.01) return [remP,outP];
  var cs=SM.c, d=SM.dir, Is=[], Os=[];
  for(var j=0;j<cs.length;j++) if(d*cs[j]>d*cc){ var lo=[SM.lox[j],SM.loy[j]], hi=[SM.hix[j],SM.hiy[j]]; if(useLo){Is.push(lo);Os.push(hi);} else {Is.push(hi);Os.push(lo);} }
  if(Is.length<2) return [remP,outP];
  var I=[src].concat(Is).map(function(p){return remapV(p,src,tgt);}), O=[outer].concat(Os).map(function(p){return remapV(p,src,tgt);});
  var n=I.length, In=[I[0].slice()], th=[0], t=0, Rb=SM.Rb||20;
  for(var j2=1;j2<n;j2++){ var sx=I[j2][0]-I[j2-1][0], sy=I[j2][1]-I[j2-1][1], L=Math.hypot(sx,sy), pv=In[j2-1];
    if(L<1e-12){ In.push(pv.slice()); th.push(t); continue; }
    var ct=Math.cos(t), st=Math.sin(t), rs=[ct*sx-st*sy, st*sx+ct*sy], cand=[pv[0]+rs[0],pv[1]+rs[1]], pk=moldPen(cand);
    if(pk[0]<0){ var k=pk[1], nx=Nx[k]*SDS[k], ny=Ny[k]*SDS[k], pr=[cand[0]-pk[0]*nx,cand[1]-pk[0]*ny], dx=pr[0]-pv[0], dy=pr[1]-pv[1];
      var need=Math.atan2(rs[0]*dy-rs[1]*dx, rs[0]*dx+rs[1]*dy), lim=L/Rb, use=Math.min(Math.max(need,-lim),lim);
      t+=use; var cu=Math.cos(use), su=Math.sin(use); cand=[pv[0]+cu*rs[0]-su*rs[1], pv[1]+su*rs[0]+cu*rs[1]];
      var p2=moldPen(cand); if(p2[0]<0){ var k2=p2[1]; cand=[cand[0]-p2[0]*Nx[k2]*SDS[k2], cand[1]-p2[0]*Ny[k2]*SDS[k2]]; } }
    In.push(cand); th.push(t); }
  var Wr=[], ri=SM.ri;
  for(var q=0;q<n;q++){ var wx=O[q][0]-I[q][0], wy=O[q][1]-I[q][1], c0=Math.cos(th[q]), s0=Math.sin(th[q]), sc=Math.abs(I[q][ri])/Math.max(Math.abs(In[q][ri]),1e-6);
    Wr.push([(c0*wx-s0*wy)*sc,(s0*wx+c0*wy)*sc]); }
  var build=function(fv){ var On=[], P=[]; for(var q=0;q<n;q++) On.push([In[q][0]+Wr[q][0]*fv, In[q][1]+Wr[q][1]*fv]);
    for(var a=0;a<n;a++) P.push(In[a]); for(var b=n-1;b>=0;b--) P.push(On[b]); P.push(In[0]); return [P,On]; };
  var fa=1.0, fb=1.02, va=polyVol(build(fa)[0])-Vtarget, vb=polyVol(build(fb)[0])-Vtarget;
  for(var it=0;it<6;it++){ if(Math.abs(vb)<1e-9*Math.max(Vtarget,1)||vb===va) break; var fn=fb-vb*(fb-fa)/(vb-va); fa=fb; va=vb; fb=fn; vb=polyVol(build(fb)[0])-Vtarget; }
  var r=build(fb); return [r[0],[r[1][0]]]; }
function simState(SS){
  var front=-1, a0=-1, U=new Float64Array(NM);
  for(var i=0;i<NM;i++){ var u=Infinity; for(var c=0;c<NC;c++){ var v=SS[c][i]; if(okv(i,v,0)) u=Math.min(u,Math.abs(v)); } U[i]=u; if(isFinite(u)){ front=i; if(a0<0) a0=i; } }
  if(front<0) return {front:-1, Vc:0};
  var ri=SM.ri, k=SM.k, ca=(k===0)?Mx[a0]:My[a0], V0=simVat(ca), Vt=SM.Vtot-V0, end=front, exh=false, W=[], CUM=[], cum=0;
  var fixed=(V0>0)?clipHalf(SIMP,k,ca,-SM.dir):null;
  for(var i2=a0;i2<=front;i2++){ var w=isFinite(U[i2])?U[i2]:SW0[i2]; if(!(w>0)) w=0;
    var rm=(ri===0)?(Mx[i2]+Nx[i2]*SDS[i2]*w/2):(My[i2]+Ny[i2]*SDS[i2]*w/2);
    cum+=2*Math.PI*Math.abs(rm)*w*SDSS[i2]; W.push(w); CUM.push(cum); }
  if(cum>Vt){ var kc=0; while(kc<CUM.length && CUM[kc]<Vt) kc++;      // Python: searchsorted(cum, Vt)
    var m=Math.max(kc,1); end=a0+m; exh=true; W=W.slice(0,m+1); cum=Vt; }
  var st={front:end, start:a0, W:W, exh:exh, Vc:cum, V0:V0, fixed:fixed};
  if(exh) return st;
  var Vq=V0+cum, cc=simInterp(Vq,SM.c), lo=[simInterp(Vq,SM.lox),simInterp(Vq,SM.loy)], hi=[simInterp(Vq,SM.hix),simInterp(Vq,SM.hiy)];
  var nx=Nx[end]*SDS[end], ny=Ny[end]*SDS[end];
  var inner=(lo[0]*nx+lo[1]*ny<=hi[0]*nx+hi[1]*ny)?lo:hi, outer=(inner===lo)?hi:lo, tgt=[Mx[end],My[end]];
  var rem=clipHalf(SIMP,SM.k,cc,SM.dir);
  var remP=rem?rem.map(function(p){return remapV(p,inner,tgt);}):null, outP=[remapV(outer,inner,tgt)];
  if(remP){ var bt=bendTail(cc, inner===lo, inner, outer, tgt, SM.Vtot-V0-cum, remP, outP); st.bent=(bt[0]!==remP); remP=bt[0]; outP=bt[1]; }
  st.rem=remP;
  var wo=W[end-a0]; st.bridge=[tgt,[Mx[end]+Nx[end]*SDS[end]*wo,My[end]+Ny[end]*SDS[end]*wo],outP[0]];
  st.T=[tgt[0]-inner[0],tgt[1]-inner[1]]; st.c=cc;
  return st; }
function lerp1(xq,X,Yv){ var n=X.length; if(xq<=X[0]) return Yv[0]; if(xq>=X[n-1]) return Yv[n-1];
  var j=upperBound(X,xq), i0=j-1, f=(X[j]>X[i0])?(xq-X[i0])/(X[j]-X[i0]):0; return Yv[i0]+(Yv[j]-Yv[i0])*f; }
function stSec(sg){ return [[lerp1(sg,SSIG,SM.pax),lerp1(sg,SSIG,SM.pay)],[lerp1(sg,SSIG,SM.pbx),lerp1(sg,SSIG,SM.pby)]]; }
function stPoly(s0,s1){ var lo=Math.min(s0,s1), hi=Math.max(s0,s1); if(hi-lo<1e-9) return null;
  var a0=stSec(lo), a1=stSec(hi), A=[a0[0]], B=[a0[1]];
  for(var j=0;j<SSIG.length;j++){ if(SSIG[j]>lo && SSIG[j]<hi){ A.push(SPA[j]); B.push(SPB[j]); } }
  A.push(a1[0]); B.push(a1[1]); var P=A.concat(B.reverse()); P.push(A[0]); return P; }
function moldSigned(p){ var best=Infinity, k=0; for(var i=0;i<NM;i++){ var dx=p[0]-Mx[i], dy=p[1]-My[i], d2=dx*dx+dy*dy; if(d2<best){best=d2;k=i;} }
  return (p[0]-Mx[k])*Nx[k]*SDS[k]+(p[1]-My[k])*Ny[k]*SDS[k]; }
function attachPiece(P,src,tgt){ if(!P) return {P:null,tg:tgt}; var ri=SM.ri, tg=[tgt[0],tgt[1]], st=Math.max(Math.floor(P.length/160),1), sub=[];
  for(var j=0;j<P.length;j+=st) sub.push(P[j]);
  for(var it=0;it<4;it++){ var pen=-Infinity; for(var q=0;q<sub.length;q++){ var v=-moldSigned(remapV(sub[q],src,tg)); if(v>pen) pen=v; }
    if(pen<=0.01) break; tg[ri]+=(pen+0.02)*(tg[ri]>=0?1:-1); }
  return {P:P.map(function(p){return remapV(p,src,tg);}), tg:tg}; }
function stripState(SS){
  var Vt=SM.Vtot, ri=SM.ri, sig=SSIG, V=SV, SIGEND=sig[sig.length-1], ARC=SM.arc;
  var U=new Float64Array(NM), F=new Uint8Array(NM);
  for(var i=0;i<NM;i++){ var u=Infinity; for(var c=0;c<NC;c++){ var v=SS[c][i]; if(okv(i,v,0)) u=Math.min(u,Math.abs(v)); }
    U[i]=u; F[i]=(STAU[i]>=0 && STAU[i]<=t+1e-9 && isFinite(u))?1:0; }
  var runs=[], a=-1, last=-1;
  for(var i1=0;i1<NM;i1++){ if(!F[i1]) continue; if(a<0){a=i1;} else if(ARC[i1]-ARC[last]>6.0){ runs.push([a,last]); a=i1; } last=i1; }
  if(a>=0) runs.push([a,last]);
  var out={polys:[], walls:[], pieces:[], Vwall:0, exh:false, elong:{}};
  if(!runs.length){ out.pieces=[stPoly(0,SIGEND)]; out.polys=out.pieces.slice(); return out; }
  var info=[];
  runs.forEach(function(r){
    var A_=r[0], B_=r[1], W=[], dV=[], tmin=Infinity;
    for(var k=A_;k<=B_;k++){ var w=F[k]?Math.min(U[k],SW0[k]):SW0[k]; if(!(w>0)) w=0;
      var rm=(ri===0)?(Mx[k]+Nx[k]*SDS[k]*w/2):(My[k]+Ny[k]*SDS[k]*w/2);
      W.push(w); dV.push(2*Math.PI*Math.abs(rm)*w*SDSS[k]); if(F[k] && STAU[k]<tmin) tmin=STAU[k]; }
    var ties=[]; for(var k2=A_;k2<=B_;k2++){ if(F[k2] && STAU[k2]<=tmin+1e-6) ties.push(k2); }
    var n=ties[Math.floor(ties.length/2)], sn=SSR[n], Vn=lerp1(sn,sig,V), VL=0, VR=0;
    for(var k3=A_;k3<=B_;k3++){ if(k3<n) VL+=dV[k3-A_]; else VR+=dV[k3-A_]; }
    var lo=A_, hi=B_, exL=false, exR=false;
    if(VL>Vn){ var cl=0, kk=0; for(var k4=n-1;k4>=A_;k4--){ cl+=dV[k4-A_]; if(cl>=Vn) break; kk++; } lo=Math.max(A_,n-1-kk); VL=Vn; exL=true; }
    if(VR>Vt-Vn){ var cr=0, kr=0; for(var k5=n;k5<=B_;k5++){ cr+=dV[k5-A_]; if(cr>=Vt-Vn) break; kr++; } hi=Math.min(B_,n+kr); VR=Vt-Vn; exR=true; }
    var Pout=[], Mk=[]; for(var k6=lo;k6<=hi;k6++){ var w6=W[k6-A_]; Pout.push([Mx[k6]+Nx[k6]*SDS[k6]*w6, My[k6]+Ny[k6]*SDS[k6]*w6]); Mk.push([Mx[k6],My[k6]]); }
    var wall=Pout.concat(Mk.reverse()); wall.push(Pout[0]);
    var sa=lerp1(Vn-VL,V,sig), sb=lerp1(Vn+VR,V,sig);
    info.push({lo:lo,hi:hi,n:n,sa:sa,sb:sb,sn:sn,Pout:Pout,exL:exL,exR:exR,eL:(ARC[n]-ARC[lo])-(sn-sa),eR:(ARC[hi]-ARC[n])-(sb-sn)});
    out.walls.push(wall); out.Vwall+=VL+VR; out.exh=out.exh||exL||exR; });
  var secIO=function(sg,k){ var sc=stSec(sg), nx=Nx[k]*SDS[k], ny=Ny[k]*SDS[k];
    return (sc[0][0]*nx+sc[0][1]*ny<=sc[1][0]*nx+sc[1][1]*ny)?[sc[0],sc[1]]:[sc[1],sc[0]]; };
  var bridge=function(k,Pk,inner,outer){ return [[Mx[k],My[k]],Pk,outer,inner,[Mx[k],My[k]]]; };
  var br=[], pcs=[], f0=info[0], fl=info[info.length-1];
  if(!f0.exL && f0.sa>1e-6){ var io=secIO(f0.sa,f0.lo), r0=attachPiece(stPoly(0,f0.sa),io[0],[Mx[f0.lo],My[f0.lo]]);
    if(r0.P){ pcs.push(r0.P); br.push(bridge(f0.lo,f0.Pout[0],remapV(io[0],io[0],r0.tg),remapV(io[1],io[0],r0.tg))); } }
  for(var q=0;q<info.length-1;q++){ var f1=info[q], f2=info[q+1];
    if(f2.sa-f1.sb>1e-6){ var pm=stPoly(f1.sb,f2.sa); if(pm){ pcs.push(pm); var i1_=secIO(f1.sb,f1.hi), i2_=secIO(f2.sa,f2.lo);
      br.push(bridge(f1.hi,f1.Pout[f1.Pout.length-1],i1_[0],i1_[1])); br.push(bridge(f2.lo,f2.Pout[0],i2_[0],i2_[1])); } } }
  if(!fl.exR && SIGEND-fl.sb>1e-6){ var io2=secIO(fl.sb,fl.hi), r1=attachPiece(stPoly(fl.sb,SIGEND),io2[0],[Mx[fl.hi],My[fl.hi]]);
    if(r1.P){ pcs.push(r1.P); br.push(bridge(fl.hi,fl.Pout[fl.Pout.length-1],remapV(io2[0],io2[0],r1.tg),remapV(io2[1],io2[0],r1.tg))); } }
  out.pieces=pcs; out.polys=out.walls.concat(br).concat(pcs);
  var ax=1-ri, an=(ax===0)?'X':'Y', up=((ax===0?Mx[NM-1]:My[NM-1])>=(ax===0?Mx[0]:My[0]));
  out.elong[an+(up?'−':'+')+' ucu']=f0.eL; out.elong[an+(up?'+':'−')+' ucu']=fl.eR;
  return out; }
var XY0=null;
function bboxOf(Ps){ var a=[Infinity,Infinity,-Infinity,-Infinity]; Ps.forEach(function(P){ if(!P) return; P.forEach(function(p){ if(p[0]<a[0])a[0]=p[0]; if(p[1]<a[1])a[1]=p[1]; if(p[0]>a[2])a[2]=p[0]; if(p[1]>a[3])a[3]=p[1]; }); }); return a; }
function xyLines(Ps){   // Python sim_xy: X ve Y eksenlerinde parça boyu, ön form → şimdi
  if(!XY0) XY0=bboxOf(STS?[SPA,SPB]:[SIMP]);
  var b=bboxOf(Ps); if(!isFinite(b[0])) b=XY0;
  var x0=XY0[2]-XY0[0], y0=XY0[3]-XY0[1], x1=b[2]-b[0], y1=b[3]-b[1], sg=function(v){return (v>=0?'+':'')+v.toFixed(2);};
  window.__XY={x0:x0,x1:x1,y0:y0,y1:y1};
  return '<br>'+D.xn+' boyu: '+x0.toFixed(2)+' → '+x1.toFixed(2)+' mm (<b>'+sg(x1-x0)+'</b>)'+
         '<br>'+D.yn+' boyu: '+y0.toFixed(2)+' → '+y1.toFixed(2)+' mm (<b>'+sg(y1-y0)+'</b>)'; }
function drawStrip(X,Y,st){
  var fillP=function(P){ if(!P||P.length<3) return; ctx.beginPath(); P.forEach(function(p,q){ if(q) ctx.lineTo(X(p[0]),Y(p[1])); else ctx.moveTo(X(p[0]),Y(p[1])); }); ctx.closePath();
    ctx.fillStyle='rgba(61,111,182,0.62)'; ctx.fill(); ctx.strokeStyle='#16325c'; ctx.lineWidth=1.1; ctx.stroke(); };
  st.polys.forEach(fillP);
  var box=document.getElementById('simstat'); box.style.display='block';
  var h='<b>Simülasyon — sac (hacim sabit)</b><br>Ön form hacmi: '+Math.round(SM.Vtot).toLocaleString('tr-TR')+' mm³<br>Kalıba sıvanan: %'+(st.Vwall/SM.Vtot*100).toFixed(1);
  var ks=Object.keys(st.elong); if(!ks.length) h+='<br>Makara henüz saca girmedi.';
  ks.forEach(function(k){ h+='<br>'+k+' uzaması: <b>'+st.elong[k].toFixed(2)+' mm</b>'; });
  h+=xyLines(st.polys);
  if(st.exh) h+='<br><span style="color:#c0392b;font-weight:bold">Malzeme bitti (bir uçta).</span>';
  box.innerHTML=h; }
function drawSim(X,Y,st){
  var fillP=function(P){ if(!P||P.length<3) return; ctx.beginPath(); P.forEach(function(p,q){ if(q) ctx.lineTo(X(p[0]),Y(p[1])); else ctx.moveTo(X(p[0]),Y(p[1])); }); ctx.closePath();
    ctx.fillStyle='rgba(61,111,182,0.62)'; ctx.fill(); ctx.strokeStyle='#16325c'; ctx.lineWidth=1.1; ctx.stroke(); };
  var box=document.getElementById('simstat'); box.style.display='block';
  if(st.front<0){ fillP(SIMP); box.innerHTML='<b>Simülasyon</b><br>Makara henüz malzemeye girmedi.<br>Ön form hacmi: '+Math.round(SM.Vtot).toLocaleString('tr-TR')+' mm³'; return; }
  if(st.fixed) fillP(st.fixed);
  var wall=[]; for(var i=st.start;i<=st.front;i++){ var w=st.W[i-st.start]; wall.push([Mx[i]+Nx[i]*SDS[i]*w, My[i]+Ny[i]*SDS[i]*w]); }
  for(var i3=st.front;i3>=st.start;i3--) wall.push([Mx[i3],My[i3]]);
  fillP(wall); if(st.bridge) fillP(st.bridge); if(st.rem) fillP(st.rem);
  var endNew;
  if(st.rem){ endNew=st.rem.reduce(function(a,p){return SM.dir>0?Math.max(a,p[SM.k]):Math.min(a,p[SM.k]);}, SM.dir>0?-Infinity:Infinity); }
  else { endNew=wall.reduce(function(a,p){return SM.dir>0?Math.max(a,p[SM.k]):Math.min(a,p[SM.k]);}, SM.dir>0?-Infinity:Infinity); }
  var el=SM.dir*(endNew-SIMEND), kn=(SM.k===0)?'X':'Y', fm=function(v){return Math.round(v).toLocaleString('tr-TR');};
  window.__SIM={el:el, Vc:st.Vc, V0:st.V0, start:st.start, front:st.front, exh:st.exh, rem:st.rem||null};
  box.innerHTML='<b>Simülasyon (hacim sabit)</b><br>Ön form hacmi: '+fm(SM.Vtot)+' mm³'+
    '<br>Yerinde kalan (makara değmedi): '+fm(st.V0)+' mm³'+
    '<br>Kalıba sıvanan duvar: '+fm(st.Vc)+' mm³ (%'+(st.Vc/SM.Vtot*100).toFixed(1)+')'+
    '<br>Uç uzaması ('+kn+' yönünde): <b>'+el.toFixed(2)+' mm</b>'+xyLines([st.fixed,wall,st.bridge,st.rem])+
    (st.bent?'<br><span style="color:#b9770e">Kalan ön form kalıba değdi — dışa doğru büküldü.</span>':'')+
    (st.exh?'<br><span style="color:#c0392b;font-weight:bold">Malzeme bitti — duvar burada sona eriyor.</span>':''); }
function drawEnv(X,Y){
  if(!EV||!envC.checked) return;
  var SS=[]; for(var c0=0;c0<NC;c0++) SS.push(envAt(c0));
  var F=null, BRK=Math.max(2,0.5*(EV.medT||EV.lim[0]));
  if(SIMD){
    ctx.strokeStyle='#9a9a9a'; ctx.lineWidth=1.0; ctx.setLineDash([8,3,2,3]);
    (EV.pf||[]).forEach(function(pl){ ctx.beginPath(); for(var q=0;q<pl.length;q+=2){ if(q) ctx.lineTo(X(pl[q]),Y(pl[q+1])); else ctx.moveTo(X(pl[q]),Y(pl[q+1])); } ctx.stroke(); });
    ctx.setLineDash([]);
    if(STS){ var __st=stripState(SS); window.__ST=__st; drawStrip(X,Y,__st); } else drawSim(X,Y,simState(SS));
    if(SHEAR && !STS){ ctx.strokeStyle='#6a3d9a'; ctx.lineWidth=1.4; ctx.setLineDash([4,2]);
      var ixs=function(i){return Mx[i]+Nx[i]*SDS[i]*TI[i];}, iys=function(i){return My[i]+Ny[i]*SDS[i]*TI[i];};
      runs(function(i){return Ta[i]>0 && TI[i]>1e-6;},ixs,iys,BRK).forEach(function(r){ctx.beginPath(); polyPath(r,ixs,iys,X,Y,null); ctx.stroke();});
      ctx.setLineDash([]); }
  }
  if(HASPF && !SIMD){
    F=new Float64Array(NM);
    for(var i=0;i<NM;i++){ var T=Ta[i]; if(!(T>0)){F[i]=NaN;continue;} var u=SHEAR?Infinity:T;
      for(var c1=0;c1<NC;c1++){ var v=SS[c1][i]; if(okv(i,v,0)) u=Math.min(u,Math.abs(v)); } F[i]=isFinite(u)?(STRIP?Math.min(u,T):u):NaN; }
    var fx=function(i){return Mx[i]+Nx[i]*SD[i]*F[i];}, fy=function(i){return My[i]+Ny[i]*SD[i]*F[i];};
    var mx=function(i){return Mx[i];}, my=function(i){return My[i];};
    var has=function(i){return Ta[i]>0 && isFinite(F[i]);};
    var RF=runs(has,fx,fy,BRK);
    ctx.fillStyle='rgba(47,127,216,0.38)'; RF.forEach(function(r){ctx.beginPath(); polyPath(r,fx,fy,X,Y,[mx,my]); ctx.fill();});
    if(SHEAR){
      var ix=function(i){return Mx[i]+Nx[i]*SD[i]*TI[i];}, iy=function(i){return My[i]+Ny[i]*SD[i]*TI[i];};
      var ov=function(i){return has(i) && F[i]<TI[i]-0.01;}, un=function(i){return has(i) && F[i]>TI[i]+0.01;};
      ctx.fillStyle='rgba(214,39,40,0.32)'; runs(ov,fx,fy,BRK).forEach(function(r){ctx.beginPath(); polyPath(r,ix,iy,X,Y,[fx,fy]); ctx.fill();});
      ctx.fillStyle='rgba(255,127,14,0.32)'; runs(un,fx,fy,BRK).forEach(function(r){ctx.beginPath(); polyPath(r,fx,fy,X,Y,[ix,iy]); ctx.fill();});
      ctx.strokeStyle='#6a3d9a'; ctx.lineWidth=1.6; ctx.setLineDash([4,2]);
      runs(function(i){return Ta[i]>0 && TI[i]>1e-6;},ix,iy,BRK).forEach(function(r){ctx.beginPath(); polyPath(r,ix,iy,X,Y,null); ctx.stroke();});
      ctx.setLineDash([]);
    } else {
      var ox=function(i){return Mx[i]+Nx[i]*SD[i]*Ta[i];}, oy=function(i){return My[i]+Ny[i]*SD[i]*Ta[i];};
      var frm=function(i){return Ta[i]>0 && F[i]<Ta[i]-1e-6;};
      ctx.fillStyle='rgba(214,39,40,0.28)'; runs(frm,fx,fy,BRK).forEach(function(r){ctx.beginPath(); polyPath(r,ox,oy,X,Y,[fx,fy]); ctx.fill();});
    }
    ctx.strokeStyle='#7f7f7f'; ctx.lineWidth=1.3; ctx.setLineDash([8,3,2,3]);
    (EV.pf||[]).forEach(function(pl){ ctx.beginPath(); for(var q=0;q<pl.length;q+=2){ if(q) ctx.lineTo(X(pl[q]),Y(pl[q+1])); else ctx.moveTo(X(pl[q]),Y(pl[q+1])); } ctx.stroke(); });
    ctx.setLineDash([]);
  }
  for(var ch=0;ch<NC;ch++){ (function(ch){ var s=SS[ch], L=EV.lim[ch], col=EV.col[ch];
    var ex=function(i){return Mx[i]+Nx[i]*s[i];}, ey=function(i){return My[i]+Ny[i]*s[i];};
    var ok=function(i){return okv(i,s[i],L);};
    var brk=HASPF?BRK:Math.max(2,0.5*L);
    // yön değişiminde de böl
    var R=runs(ok,ex,ey,brk), R2=[];
    R.forEach(function(r){ var cur=[r[0]]; for(var q=1;q<r.length;q++){ var a=s[r[q]], b=s[r[q-1]];
        if(a!==0&&b!==0&&((a>0)!==(b>0))){ if(cur.length>1) R2.push(cur); cur=[]; } cur.push(r[q]); } if(cur.length>1) R2.push(cur); });
    R2.forEach(function(r){
      if(!HASPF){ ctx.beginPath(); polyPath(r,ex,ey,X,Y,[function(i){return Mx[i];},function(i){return My[i];}]); ctx.fillStyle=hex2rgba(col,0.32); ctx.fill(); }
      ctx.beginPath(); polyPath(r,ex,ey,X,Y,null); ctx.setLineDash([6,3]); ctx.strokeStyle=col; ctx.lineWidth=1.8; ctx.stroke(); ctx.setLineDash([]); });
  })(ch); }
  if(HASPF && !SIMD){ ctx.strokeStyle='#1f3a5f'; ctx.lineWidth=2.2;
    runs(function(i){return Ta[i]>0 && isFinite(F[i]);},function(i){return Mx[i]+Nx[i]*SD[i]*F[i];},function(i){return My[i]+Ny[i]*SD[i]*F[i];},BRK)
      .forEach(function(r){ ctx.beginPath(); polyPath(r,function(i){return Mx[i]+Nx[i]*SD[i]*F[i];},function(i){return My[i]+Ny[i]*SD[i]*F[i];},X,Y,null); ctx.stroke(); }); }
}

// ---- CNC EKRANI: program satırları, eksen konumu, kalan mesafe ----
var CN=D.cnc||null, cncRow=document.getElementById('cncrow'), cards=[];
var MODE=['G0','G1','G2','G3','G4','WAIT','G1'];
function el(tag,cls,txt){var e=document.createElement(tag); if(cls) e.className=cls; if(txt!==undefined) e.textContent=txt; return e;}
if(!CN){ cncRow.style.display='none'; } else {
  CN.forEach(function(c,i){
    var card=el('div','cnc'), hdr=el('div','hdr'), body=el('div','body'), prog=el('div','prog'), pos=el('div','pos');
    var ttl=el('span','', 'KANAL '+(i+1)); ttl.style.color=D.col[i]==='#0072B2'?'#58a6ff':'#ff9b6b';
    var st=el('span','st',''); hdr.appendChild(ttl); hdr.appendChild(st);
    var lines=[]; for(var q=0;q<6;q++){var d=el('div','ln',''); prog.appendChild(d); lines.push(d);}
    var tb=el('table'), th=el('tr'); ['', 'Konum', 'Kalan'].forEach(function(h){th.appendChild(el('th','',h));}); tb.appendChild(th);
    function row(lbl){var r=el('tr'); var k=el('td','k',lbl), a=el('td','big',''), b=el('td','',''); r.appendChild(k); r.appendChild(a); r.appendChild(b); tb.appendChild(r); return [a,b];}
    var rx=row('X'+(c.fx===2?' Ø':'')), ry=row('Y'+(c.fy===2?' Ø':''));
    var rl=el('tr'), lk=el('td','k','Yol'), lv=el('td','',''), lv2=el('td','','');
    rl.appendChild(lk); rl.appendChild(lv); rl.appendChild(lv2); tb.appendChild(rl);
    pos.appendChild(tb); body.appendChild(prog); body.appendChild(pos); card.appendChild(hdr); card.appendChild(body); cncRow.appendChild(card);
    cards.push({st:st, lines:lines, rx:rx, ry:ry, lk:lk, lv:lv, lv2:lv2});
  });
}
function f3(v){return (Math.abs(v)<5e-4?0:v).toFixed(3);}
function blockAt(B,tt){var lo=0,hi=B.length-1,ans=-1; while(lo<=hi){var m=(lo+hi)>>1; if(B[m][0]<=tt+1e-9){ans=m;lo=m+1;} else hi=m-1;} return ans;}
function updateCNC(){
  if(!CN) return;
  CN.forEach(function(c,i){
    if(D.act && !D.act[i]){ cards[i].st.textContent='BOŞ PROGRAM'; return; }
    var cd=cards[i], B=c.B, k=blockAt(B,t), L=c.L;
    var b=(k>=0)?B[k]:null, done=(!b)||(k===B.length-1&&t>b[1]+1e-6);
    var cur=-1, px=0, py=0, rx=0, ry=0, rem=0, status='', remLbl='Yol', remTxt='';
    if(!b){ status='—'; if(c.p0){ px=c.p0[0]; py=c.p0[1]; } }
    else {
      cur=(b[2]===null)?-1:b[2];
      var code=b[3], f=(b[1]>b[0])?Math.min(Math.max((t-b[0])/(b[1]-b[0]),0),1):1; if(done) f=1;
      if(code===2||code===3){ var a=b[11]+(b[12]-b[11])*f; px=b[8]+b[10]*Math.cos(a); py=b[9]+b[10]*Math.sin(a); rem=Math.abs(b[12]-b[11])*b[10]*(1-f); }
      else { px=b[4]+(b[6]-b[4])*f; py=b[5]+(b[7]-b[5])*f; rem=Math.hypot(b[6]-px,b[7]-py); }
      rx=(b[6]-px)*c.fx; ry=(b[7]-py)*c.fy;
      if(done){ status='PROGRAM SONU'; rem=0; rx=0; ry=0; }
      else if(code===5){ status='WAIT – diğer kanallar bekleniyor'; remLbl='Bekleme'; remTxt=(b[1]-t).toFixed(2)+' s'; }
      else if(code===4){ status='G4 bekleme'; remLbl='Bekleme'; remTxt=(b[1]-t).toFixed(2)+' s'; }
      else { status=MODE[code]+(code===6?' (köşe geçişi)':'')+(b[13]>0?'   F'+(+b[13].toFixed(1)):''); }
    }
    cd.st.textContent=status;
    var start=Math.max(0,Math.min(cur-2,L.length-6));
    for(var q=0;q<6;q++){ var idx=start+q, d=cd.lines[q];
      if(idx<L.length){ var n='N'+(idx+1); while(n.length<5) n+=' '; d.textContent=n+L[idx]; } else d.textContent='';
      d.className=(idx===cur&&!done)?'ln cur':'ln'; }
    cd.rx[0].textContent=f3(px*c.fx); cd.ry[0].textContent=f3((py-c.yo)*c.fy);
    cd.rx[1].textContent=f3(rx); cd.ry[1].textContent=f3(ry);
    cd.lk.textContent=remLbl; cd.lv.textContent=''; cd.lv2.textContent=remTxt||f3(rem);
  });
}

function setSpeed(v){speed=v; spd.value=v; spdv.textContent=v.toFixed(1)+'×';
  var bs=pre.getElementsByTagName('button'); for(var i=0;i<bs.length;i++){bs[i].className=(Math.abs(parseFloat(bs[i].getAttribute('data-v'))-v)<1e-9)?'on':'';}}
[0.5,1,2,5,10,20].forEach(function(v){var b=document.createElement('button'); b.textContent=v+'×'; b.setAttribute('data-v',v);
  b.onclick=function(){setSpeed(v);}; pre.appendChild(b);});
function resize(){var dpr=window.devicePixelRatio||1, r=cv.getBoundingClientRect(); W=r.width; H=r.height;
  cv.width=Math.max(10,Math.round(W*dpr)); cv.height=Math.max(10,Math.round(H*dpr)); ctx.setTransform(dpr,0,0,dpr,0,0); draw();}
function draw(){
  ctx.clearRect(0,0,W,H);
  var L=62,B=46,T=34,R=Math.min(165,W*0.26);
  var pw=W-L-R, ph=H-T-B; if(pw<20||ph<20) return;
  var sc=Math.min(pw/(xmax-xmin), ph/(ymax-ymin));
  var ox=L+(pw-sc*(xmax-xmin))/2, oy=T+(ph-sc*(ymax-ymin))/2;
  function X(x){return ox+(x-xmin)*sc;} function Y(y){return oy+(ymax-y)*sc;}
  ctx.font='bold 15px Arial'; ctx.fillStyle='#222'; ctx.textAlign='center'; ctx.fillText('CNC Simülasyonu',ox+sc*(xmax-xmin)/2,20);
  var st=nice(Math.max(xmax-xmin,ymax-ymin)/8);
  ctx.save(); ctx.beginPath(); ctx.rect(ox,oy,sc*(xmax-xmin),sc*(ymax-ymin)); ctx.clip();
  ctx.strokeStyle='rgba(0,0,0,0.12)'; ctx.lineWidth=1; ctx.beginPath();
  for(var gx=Math.ceil(xmin/st)*st; gx<=xmax; gx+=st){ctx.moveTo(X(gx),oy); ctx.lineTo(X(gx),oy+sc*(ymax-ymin));}
  for(var gy=Math.ceil(ymin/st)*st; gy<=ymax; gy+=st){ctx.moveTo(ox,Y(gy)); ctx.lineTo(ox+sc*(xmax-xmin),Y(gy));}
  ctx.stroke();
  ctx.strokeStyle='#000'; ctx.lineWidth=1.4; ctx.beginPath();
  D.mold.forEach(function(p,i){ if(i===0) ctx.moveTo(X(p[0]),Y(p[1])); else ctx.lineTo(X(p[0]),Y(p[1])); }); ctx.stroke();
  drawEnv(X,Y);
  var q=kf(), k=q[0], f=q[1];
  D.ch.forEach(function(c,i){
    if(D.act && !D.act[i]) return;
    var col=D.col[i];
    ctx.strokeStyle=col; ctx.lineWidth=2.3; ctx.beginPath();
    var pen=false, m=c.m[k];
    for(var j=0;j<m;j++){var p=c.F[j]; if(p===null){pen=false; continue;}
      if(!pen){ctx.moveTo(X(p[0]),Y(p[1])); pen=true;} else ctx.lineTo(X(p[0]),Y(p[1]));}
    var hx=c.hx[k], hy=c.hy[k];
    if(hx!==null){ if(c.hx[k+1]!==null && c.hx[k+1]!==undefined){hx=lerp(c.hx,k,f); hy=lerp(c.hy,k,f);}
      if(pen) ctx.lineTo(X(hx),Y(hy)); }
    ctx.stroke();
    var cx=lerp(c.cx,k,f), cy=lerp(c.cy,k,f), px=lerp(c.px,k,f), py=lerp(c.py,k,f);
    ctx.beginPath();
    if(D.shp && D.shp[i]){ var SP=D.shp[i]; ctx.moveTo(X(cx+SP[0]),Y(cy+SP[1])); for(var e2=2;e2<SP.length;e2+=2) ctx.lineTo(X(cx+SP[e2]),Y(cy+SP[e2+1])); ctx.closePath(); }
    else ctx.arc(X(cx),Y(cy),D.r[i]*sc,0,6.2832);
    ctx.fillStyle=hex2rgba(col,0.22); ctx.fill();
    ctx.strokeStyle=col; ctx.lineWidth=1.8; ctx.stroke();
    ctx.lineWidth=1.2; ctx.beginPath(); ctx.moveTo(X(cx),Y(cy)); ctx.lineTo(X(px),Y(py)); ctx.stroke();
    ctx.lineWidth=1.6; ctx.beginPath(); ctx.moveTo(X(cx)-6,Y(cy)); ctx.lineTo(X(cx)+6,Y(cy)); ctx.moveTo(X(cx),Y(cy)-6); ctx.lineTo(X(cx),Y(cy)+6); ctx.stroke();
    ctx.fillStyle=col; ctx.beginPath(); ctx.arc(X(px),Y(py),4,0,6.2832); ctx.fill();
  });
  ctx.restore();
  ctx.strokeStyle='#444'; ctx.lineWidth=1; ctx.strokeRect(ox,oy,sc*(xmax-xmin),sc*(ymax-ymin));
  ctx.fillStyle='#333'; ctx.font='11px Arial'; ctx.textAlign='center';
  for(var tx=Math.ceil(xmin/st)*st; tx<=xmax; tx+=st){ctx.fillText(+tx.toFixed(6),X(tx),oy+sc*(ymax-ymin)+14);}
  ctx.textAlign='right';
  for(var ty=Math.ceil(ymin/st)*st; ty<=ymax; ty+=st){ctx.fillText(+ty.toFixed(6),ox-5,Y(ty)+4);}
  // eksen adları
  ctx.fillStyle='#111'; ctx.font='bold 13px Arial'; ctx.textAlign='center';
  ctx.fillText(D.xn||'X', ox+sc*(xmax-xmin)/2, oy+sc*(ymax-ymin)+34);
  ctx.save(); ctx.translate(14, oy+sc*(ymax-ymin)/2); ctx.rotate(-Math.PI/2); ctx.fillText(D.yn||'Y',0,0); ctx.restore();
  ctx.textAlign='left'; ctx.font='bold 12px Arial'; ctx.fillStyle='#111';
  ctx.fillText('t = '+t.toFixed(2)+' / '+D.T.toFixed(2)+' s',ox+6,oy+16);
  var lgx=W-R+14, lgy=T+4; ctx.font='11px Arial';
  function leg(col,txt,y){ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.beginPath(); ctx.moveTo(lgx,y); ctx.lineTo(lgx+22,y); ctx.stroke(); ctx.fillStyle='#222'; ctx.fillText(txt,lgx+28,y+4);}
  leg('#000','Kalıp (sabit)',lgy+8);
  for(var qq=0;qq<D.col.length;qq++) leg(D.col[qq],'Kanal '+(qq+1)+' — kontur',lgy+28+qq*20);
  if(EV&&envC.checked){ for(var qe=0;qe<NC;qe++){ ctx.setLineDash([6,3]); leg(EV.col[qe],'Kanal '+(qe+1)+' — radüs izi',lgy+28+(D.col.length+qe)*20); ctx.setLineDash([]); }
    if(HASPF && SIMD){ var ys=lgy+28+(D.col.length+NC)*20; ctx.setLineDash([8,3,2,3]); leg('#9a9a9a','Ön form (ilk hali)',ys); ctx.setLineDash([]);
      ctx.fillStyle='rgba(61,111,182,0.75)'; ctx.fillRect(lgx,ys+16,22,8); ctx.fillStyle='#222'; ctx.fillText('Ürün (simülasyon)',lgx+28,ys+24);
      if(SHEAR && !STRIP){ ctx.setLineDash([4,2]); leg('#6a3d9a','İdeal (t₀·sin α)',ys+40); ctx.setLineDash([]); } }
    else if(HASPF){ var yb=lgy+28+(D.col.length+NC)*20; ctx.setLineDash([8,3,2,3]); leg('#7f7f7f','Ön form (DXF)',yb); ctx.setLineDash([]);
      leg('#1f3a5f',SHEAR?'Gerçek son yüzey':'Son form yüzeyi',yb+20);
      var box=function(c,txt,y){ ctx.fillStyle=c; ctx.fillRect(lgx,y,22,8); ctx.fillStyle='#222'; ctx.fillText(txt,lgx+28,y+8); };
      box('rgba(47,127,216,0.55)','Et (kalıp–yüzey)',yb+36);
      if(SHEAR){ ctx.setLineDash([4,2]); leg('#6a3d9a',STRIP?'Sac kalınlığı (inceltmesiz)':'İdeal (t₀·sin α)',yb+60); ctx.setLineDash([]);
        box('rgba(214,39,40,0.5)',STRIP?'İnceltme':'Aşırı inceltme',yb+72); if(!STRIP) box('rgba(255,127,14,0.5)','Eksik inceltme',yb+88); }
      else box('rgba(214,39,40,0.45)','Akıtılan malzeme',yb+52); } }
}
function ui(){tm.value=Math.round(t/D.T*10000); tv.textContent=t.toFixed(2)+' / '+D.T.toFixed(2)+' s'; updatePanel(); updateCNC();}
function setT(nt){t=nt; measStep(); draw(); ui();}
function setPlaying(p){playing=p; playB.textContent=p?'⏸ Duraklat':'▶ Oynat'; if(p){ if(t>=D.T) setT(0); last=performance.now(); requestAnimationFrame(frame);} }
function frame(now){ if(!playing) return; var dtt=(now-last)/1000; last=now; var nt=t+dtt*speed;
  if(nt>=D.T){ t=D.T; measStep(); if(meas.on) stopMeas('Süre sonuna ulaşıldı.');
    if(loopC.checked){t=0;} else {setPlaying(false);} }
  else { t=nt; measStep(); }
  draw(); ui(); if(playing) requestAnimationFrame(frame);}
playB.onclick=function(){setPlaying(!playing);};
rstB.onclick=function(){ if(meas.on) stopMeas('Başa dönüldü.'); setT(0); };
spd.oninput=function(){setSpeed(parseFloat(spd.value));};
tm.oninput=function(){setT(tm.value/10000*D.T);};
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


_CODE_G = {'line': 1, 'fillet': 6}


def build_block_table(segs, timeline, total_time, start=None):
    """CNC ekranı için blok tablosu. Her satır:
    [t0, t1, satır, kod, sx, sy, ex, ey, cx, cy, r, a0, a1, F]
    kod: 0=G0 1=G1 2=G2 3=G3 4=G4 5=WAIT 6=köşe geçişi. Koordinatlar programlanan (kompanzasyonsuz)."""
    r4 = lambda v: round(float(v), 4)
    entries = list(timeline)
    blocks, ei, t_prev, last = [], 0, 0.0, (tuple(start) if start is not None else None)
    for seg in segs:
        if seg['type'] == 'wait':
            t0 = t_prev
            t1 = max(entries[ei][0] if ei < len(entries) else total_time, t0)
            p = last or (0.0, 0.0)
            blocks.append([r4(t0), r4(t1), seg.get('line'), 5, r4(p[0]), r4(p[1]), r4(p[0]), r4(p[1]),
                           0, 0, 0, 0, 0, 0])
            t_prev = t1
            continue
        if ei >= len(entries):
            break
        t0, t1, s_ = entries[ei]
        ei += 1
        ps = s_.get('prog_start') or (last or (0.0, 0.0))
        pe = s_.get('prog_end') or ps
        arc = [0, 0, 0, 0, 0]
        if s_['type'] == 'dwell':
            code, F = 4, 0
        else:
            F = s_.get('feed', 0.0)
            if _is_g0(s_):
                code = 0
            elif s_.get('kind') == 'arc':
                code = 2 if s_.get('clockwise') else 3
                if s_.get('prog_arc'):
                    arc = [r4(v) for v in s_['prog_arc']]
            else:
                code = _CODE_G.get(s_.get('kind'), 1)
        blocks.append([r4(t0), r4(t1), s_.get('line'), code, r4(ps[0]), r4(ps[1]), r4(pe[0]), r4(pe[1])]
                      + arc + [round(float(F), 1)])
        t_prev, last = t1, pe
    return blocks


def player_cnc_data(cfg, channels_segments, timelines, total_time, start_points=None):
    out = []
    for i in range(N_CH):
        ch = cfg['channels'][i]
        dia = ch.get('dia_mode') == 'diameter'
        ax = ch.get('dia_axis', 'X')
        out.append({'L': (ch['code'] or '').strip().split('\n'),
                    'B': build_block_table(channels_segments[i], timelines[i], total_time,
                                           start_points[i] if start_points else (ch['x0'], ch['y0'])),
                    'fx': 2 if (dia and 'X' in ax) else 1, 'fy': 2 if (dia and 'Y' in ax) else 1,
                    'yo': round(float(ch.get('y_offset', 0.0)), 4),
                    'p0': [round(float(v), 4) for v in (start_points[i] if start_points else (ch['x0'], ch['y0']))]})
    return out


def player_env_data(timelines, radii, mold_points, limits, show, max_samples=20000, form=None):
    M, Nrm, _ = mold_rays(mold_points)
    if len(M) == 0:
        return None
    r4 = lambda a: [round(float(v), 4) for v in np.asarray(a).ravel()]
    S = []
    for i in range(N_CH):
        C, T = tool_samples(timelines[i], radii[i])
        if len(C) > max_samples:                                    # çok uzun programlarda seyrelt
            idx = np.linspace(0, len(C) - 1, max_samples).round().astype(int)
            C, T = C[idx], T[idx]
        S.append({'c': r4(C), 't': [float(v) for v in T]})
    out = {'M': r4(M), 'N': [round(float(v), 6) for v in Nrm.ravel()], 'S': S,
           'lim': [float(limits[i] if limits[i] > 0 else radii[i]) for i in range(N_CH)],
           'col': CHANNEL_ENV_COLORS[:N_CH], 'show': bool(show)}
    fin = (form or {}).get('final')
    if fin is not None and len(fin['T']) == len(M):
        out['mode'] = fin.get('mode', 'flow')
        if fin.get('thr') is not None:
            out['thr'] = float(fin['thr'])
        if fin.get('Ti') is not None:
            Ti = np.asarray(fin['Ti'], float)
            out['Ti'] = [round(float(v), 4) if np.isfinite(v) else -1.0 for v in Ti]
        T = np.asarray(fin['T'], float)
        has = np.isfinite(T)
        out['T'] = [round(float(v), 4) if h else -1.0 for v, h in zip(T, has)]
        out['sd'] = [int(v) for v in fin['side']]
        tmax = float(np.max(T[has])) if has.any() else 1.0
        _sm0 = fin.get('sim') or {}
        _sl = float(np.max(_sm0['lim'])) if _sm0.get('ok') and _sm0.get('lim') is not None else 0.0
        out['lim'] = [max(tmax, float(fin.get('thr') or 0.0), _sl)] * N_CH   # takım örneklerini elemek için üst sınır
        out['medT'] = float(np.median(T[has])) if has.any() else 1.0
        out['pf'] = [[round(float(v), 4) for v in np.asarray(A).ravel()] for A, _cl in form['pf_chains']]
        sm = fin.get('sim')
        if sm and sm.get('ok') and sm.get('kind') == 'strip':
            r5 = lambda a: [round(float(v), 5) for v in np.asarray(a, float).ravel()]
            stp = sm['strip']
            out['sim'] = {'kind': 'strip', 'W0': r5(sm['W0']), 'sd': [int(v) for v in sm['side']],
                          'dsc': [float(v) for v in sm['dsc']], 'arc': [float(v) for v in fin['profile']['arc']],
                          'ri': sm['ri'], 'thr': float(sm['thr']), 'Vtot': float(stp['Vtot']),
                          'tau': [float(v) if np.isfinite(v) else -1.0 for v in sm['tau']], 'sigR': [float(v) for v in sm['sigR']],
                          'sig': [float(v) for v in stp['sig']], 'V': [float(v) for v in stp['V']],
                          'pax': [float(v) for v in stp['pA'][:, 0]], 'pay': [float(v) for v in stp['pA'][:, 1]],
                          'pbx': [float(v) for v in stp['pB'][:, 0]], 'pby': [float(v) for v in stp['pB'][:, 1]]}
        elif sm and sm.get('ok'):
            r5 = lambda a: [round(float(v), 5) for v in np.asarray(a, float).ravel()]
            out['sim'] = {'W0': r5(sm['W0']), 'sd': [int(v) for v in sm['side']], 'ds': r5(fin['ds']),
                          'ri': sm['ri'], 'k': sm['k'], 'dir': sm['dir'], 'tmax': sm['tmax'], 'Vtot': sm['Vtot'], 'Rb': float(sm.get('Rb', 20.0)),
                          'lim': r5(sm['lim']) if sm.get('lim') is not None else None,
                          'tR': ([float(v) if np.isfinite(v) else -1.0 for v in sm['tR']]
                                 if sm.get('tR') is not None else None),
                          'c': r5(sm['c']), 'V': [float(v) for v in sm['V']],
                          'lox': r5(sm['lo'][:, 0]), 'loy': r5(sm['lo'][:, 1]),
                          'hix': r5(sm['hi'][:, 0]), 'hiy': r5(sm['hi'][:, 1]), 'poly': r5(sm['poly'])}
    return out


def build_player_data(channels_segments, start_points, radii, mold_points, timelines, total_time,
                      speed0=1.0, rate=30, xname='X', yname='Y', env=None, cnc=None, extra=None):
    xmin, xmax, ymin, ymax = _scene_bounds(channels_segments, start_points, radii, mold_points, extra)
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
            if tl:
                cp = get_position(tl, t, 'points')
                pp = get_position(tl, t, 'orig_points')
            else:                                   # hareketsiz kanal: başlangıç noktasında dursun
                cp = pp = start_points[i]
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
    return {'T': float(total_time), 'n': n, 'dt': dt, 'b': [r3(xmin), r3(xmax), r3(ymin), r3(ymax)],
            'mold': [[r3(p[0]), r3(p[1])] for p in mold_points], 'r': [r3(r) for r in radii],
            'shp': [([round(float(v), 4) for v in tool_shape(r).ravel()] if tool_shape(r) is not None else None) for r in radii],
            'rch': [r3(tool_reach(r)) for r in radii],
            'col': CHANNEL_COLORS[:N_CH], 'speed': float(speed0), 'xn': xname, 'yn': yname, 'ch': chans,
            'env': env, 'cnc': cnc, 'act': [bool(channel_active(s)) for s in channels_segments]}


def build_player_html(data, height=960):
    doc = PLAYER_TEMPLATE.replace('__DATA__', json.dumps(data, separators=(',', ':')))
    return (f'<iframe srcdoc="{html.escape(doc, quote=True)}" sandbox="allow-scripts" allow="fullscreen" '
            f'style="width:100%;height:{height}px;border:1px solid #ddd;border-radius:8px;background:#fff"></iframe>')


# ===== Hücre 22 =====
import gradio as gr

N_ANIM_OUT = 14   # run_all çıktı sayısı


def _split_vals(vals):
    """vals = N_CFG cfg + fps + speed + ac_start + ac_end + ac_step + N_OPT grafik seçeneği"""
    cfg = cfg_from_values(vals[:N_CFG])
    fps = int(_num(vals[N_CFG], 20))
    speed = _num(vals[N_CFG + 1], 1.0) or 1.0
    ac_s, ac_e, ac_st = (_num(vals[N_CFG + 2], 0.0), _num(vals[N_CFG + 3], 0.0), _num(vals[N_CFG + 4], 0.1))
    opts = opts_from_values(vals[N_CFG + 5:N_CFG + 5 + N_OPT])
    k = N_CFG + 5 + N_OPT
    opts['pf_state'] = vals[k] if len(vals) > k and isinstance(vals[k], dict) else None
    opts['tool_state'] = vals[k + 1] if len(vals) > k + 1 and isinstance(vals[k + 1], dict) else {}
    apply_tool_shapes(cfg, opts['tool_state'])
    return cfg, fps, speed, ac_s, ac_e, ac_st, opts


def _plot_kw(opts):
    return dict(xname=opts['xname'], yname=opts['yname'],
                label_font=opts['label_font'], label_shift=opts['label_shift'])


def _preform(opts, cfg=None):
    """Kullanılacak ön form (yüklü ve 'kullan' işaretliyse): zincirler + model + radyal eksen."""
    st = opts.get('pf_state')
    if not opts.get('pf_use') or not st or not st.get('layers'):
        return None
    chains, rnote = preform_chains(st, opts.get('pf_layers'), opts.get('pf_dx', 0.0), opts.get('pf_dy', 0.0),
                                   with_note=True)
    if not chains:
        return None
    radial, why = resolve_radial(opts.get('pf_radial', 'auto'), cfg or {})
    radial, warn = radial_guard(chains, radial)
    if warn:
        why = 'otomatik düzeltildi'
    return {'chains': chains, 'model': opts.get('pf_model', 'strip'), 'radial': radial, 'radial_why': why,
            'radial_warn': warn, 'repair_note': rnote, 'sim': bool(opts.get('pf_sim'))}


def _pf_note(opts, cfg, form):
    if not (opts.get('pf_state') and opts.get('pf_use')):
        return ""
    if not (form and form.get('final')):
        return "\nÖn form seçili ama kullanılabilir çizgi bulunamadı (katman seçimini kontrol et)."
    fin = form['final']
    _pp = _preform(opts, cfg) or {}
    _pw = "".join("\n" + x for x in (_pp.get('repair_note', ''), _pp.get('radial_warn', '')) if x)
    return _pw + _pf_note_body(opts, cfg, form, fin)


THICK_HINT_MM = 10.0     # sac modelinde bu kalınlığın üstü: kalın (döküm / dövme) ön form uyarısı


def _pf_note_body(opts, cfg, form, fin):
    if fin.get('mode') == 'strip':
        pf = _preform(opts, cfg)
        if not any(cl for _A, cl in form['pf_chains']):
            return "\nUYARI: Sac ön form modeli için DXF kapalı kontur olmalı."
        stp = fin.get('strip') or {}
        return (f"\nÖn form: sac modeli (sıvama + inceltme), parça yarıçapı {pf['radial']} ekseni. "
                f"Sac kalınlığı DXF'ten: {np.min(stp.get('t', [np.nan])):.2f} – {np.max(stp.get('t', [np.nan])):.2f} mm "
                f"(ortanca {stp.get('tmed', float('nan')):.2f}). Hacim {stp.get('Vtot', 0):,.0f} mm³."
                + (f"\nUYARI: Ön form kalın (en kalın yer {np.max(stp.get('t', [0])):.1f} mm) — sac modeli ince sac içindir. "
                   "Döküm / dövme gibi kalın ön formlarda 'Kalıp üzerinde (flow forming)' modelini seç."
                   if np.max(stp.get('t', [0])) > THICK_HINT_MM else ""))
    if fin.get('mode') == 'shear':
        pf = _preform(opts, cfg)
        closed = any(cl for _A, cl in form['pf_chains'])
        msg = (f"\nÖn form: shear forming (sinüs kuralı t₀·sin α), radyal eksen {pf['radial']} ({pf['radial_why']}). "
               "Kalınlık yalnız DXF'teki ön formdan alındı (Et Kalınlığı sekmesi).")
        if not closed:
            msg += "\nUYARI: Shear forming için ön form KAPALI kontur olmalı (eksenel kalınlık okunamadı)."
        elif not np.isfinite(fin['T']).any():
            msg += "\nUYARI: Ön formun radyal aralığı kalıpla örtüşmüyor — radyal ekseni kontrol et."
        return msg
    return "\nÖn form: kalıp üzerinde modeli — radüs izi lokal malzeme kalınlığıyla sınırlandı (Et Kalınlığı sekmesi)."


def _envelopes(timelines, radii, mold_pts, opts, force=False, cfg=None):
    """Radüs izi + (varsa) ön forma göre son form. Gösterim kapalı, ön form yok ve zorlanmadıysa hesaplanmaz."""
    pf = _preform(opts, cfg)
    if (not opts['show_env'] and not force and pf is None) or len(mold_pts) < 2:
        return None
    return compute_form(timelines, radii, mold_pts, opts['env_limit'], pf)


def _static_plots(segs, starts, radii, mold_pts, opts, form=None):
    show = bool(form) and opts['show_env']
    comp = fig_to_svg_card(plot_comparison_fig(segs, starts, radii, mold_pts, opts['xname'], opts['yname'],
                                               envelopes=(form['ch'] if show else None),
                                               form=(form if form and form.get('final') else None)))
    if len(mold_pts) > 1:
        ind = [fig_to_svg_card(plot_channel_fig(i, segs, starts, radii, mold_pts, **_plot_kw(opts),
                                                envelope=(form['ch'][i] if show else None),
                                                pf_chains=(form['pf_chains'] if form else None)))
               for i in range(N_CH)]
    else:
        ind = ["<p>Kalıp noktası yetersiz.</p>"] * N_CH
    return comp, ind


# -------------------------------------------------------------
# 1) Ana çalıştırma: animasyon + analizler + hızlı oynatıcı
# -------------------------------------------------------------
def run_all(*vals):
    try:
        cfg, fps, speed, ac_s, ac_e, ac_st, opts = _split_vals(vals)
        segs, starts, radii, mold_pts, note = build_channels_cfg(cfg)
        timelines, total_time = synchronize(segs)
        if total_time <= 0:
            total_time = 1.0

        token, n_frames = register_anim(segs, starts, radii, mold_pts, timelines, total_time, fps, speed,
                                        opts['xname'], opts['yname'])
        frame0 = render_frame_image(token, 0)

        form = _envelopes(timelines, radii, mold_pts, opts, cfg=cfg)
        comp_html, ind_html = _static_plots(segs, starts, radii, mold_pts, opts, form)
        table = distance_table(segs, mold_pts)
        th_html, th_md, th_df = thickness_profile(form)

        try:
            env_d = (player_env_data(timelines, radii, mold_pts, [opts['env_limit']] * N_CH, opts['show_env'],
                                     form=form) if len(mold_pts) > 1 else None)
            cnc_d = player_cnc_data(cfg, segs, timelines, total_time, starts)
            player_html = build_player_html(build_player_data(segs, starts, radii, mold_pts, timelines,
                                                              total_time, speed0=speed,
                                                              xname=opts['xname'], yname=opts['yname'],
                                                              env=env_d, cnc=cnc_d, extra=_form_extra(form)))
        except Exception:
            player_html = "<p>Hızlı oynatıcı hazırlanamadı:</p><pre>" + html.escape(traceback.format_exc()) + "</pre>"

        status = (note + "\n" if note else "") + \
                 (f"Simülasyon hazır. Toplam süre: {total_time:.2f} sn — animasyon kare sayısı: {n_frames}")
        status += _pf_note(opts, cfg, form)
        _sm = ((form or {}).get('final') or {}).get('sim') if form else None
        if _sm is not None:
            if _sm.get('ok'):
                q = sim_summary(_sm, _sm['final'])
                if q.get('elong_groups') is not None:
                    el = ", ".join(f"{k} uzaması {v:.2f} mm" for k, v in q['elong_groups'].items())
                else:
                    el = f"uç uzaması {q['elong']:.2f} mm ({q['axis']} yönünde)"
                status += (f"\nSimülasyon: hacim sabit (fark %{q['err']:.2f}), {el or 'şekillenme yok'}"
                           + (" — malzeme bitti." if q['exhausted'] else ". Hızlı Oynatıcıda adım adım izle.")
                           + "\nEksenlere göre uzama: " + sim_xy_text(_sm, _sm['final'], (opts['xname'], opts['yname'])))
            else:
                status += "\nSimülasyon çalışmadı: " + _sm.get('why', '')
        return (frame0,
                gr.update(minimum=0, maximum=n_frames, value=0),
                gr.update(value=playback_interval(fps, speed), active=False),
                token,
                comp_html, ind_html[0], ind_html[1], ind_html[2],
                table, player_html, th_html, th_md, th_df, status)
    except Exception:
        err = "Simülasyon hazırlanırken hata oluştu:\n" + traceback.format_exc()
        return tuple([gr.update()] * (N_ANIM_OUT - 1) + [err])


def redraw_plots(*vals):
    """Yazı boyutu / kaydırma / eksen adı / radüs izi ayarı değişince yalnızca statik grafikleri yeniden çizer."""
    try:
        cfg, fps, speed, ac_s, ac_e, ac_st, opts = _split_vals(vals)
        segs, starts, radii, mold_pts, note = build_channels_cfg(cfg)
        timelines, total_time = synchronize(segs)
        form = _envelopes(timelines, radii, mold_pts, opts, cfg=cfg)
        comp, ind = _static_plots(segs, starts, radii, mold_pts, opts, form)
        th_html, th_md, th_df = thickness_profile(form)
        return comp, ind[0], ind[1], ind[2], th_html, th_md, th_df, "Grafikler güncellendi."
    except Exception:
        return (gr.update(),) * 7 + ("Grafikler yeniden çizilirken hata oluştu:\n" + traceback.format_exc(),)


def upload_preform(file_obj):
    """Ön form DXF yüklenince: state + katman listesi + durum mesajı."""
    if file_obj is None:
        return None, gr.update(choices=[], value=[]), "Ön form kaldırıldı."
    try:
        path = file_obj if isinstance(file_obj, str) else getattr(file_obj, 'name', None) or file_obj['name']
        st, msg = load_preform_file(path)
        try:
            _ch, _rn = preform_chains(st, None, with_note=True)
            if _rn:
                msg += "\n" + _rn
            elif any(cl for _A, cl in _ch):
                msg += "\nKontur temiz: kapalı kontur bulundu, onarım gerekmedi."
        except Exception:
            pass
        return st, gr.update(choices=preform_layer_choices(st), value=list(st['layers'])), \
            msg + "\nKatman seçimini kontrol et, sonra 'Çalıştır'a bas."
    except Exception as e:
        return gr.update(), gr.update(), f"Ön form DXF okunamadı: {e}"


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
        opts = opts_from_values(vals[N_CFG + 1:N_CFG + 1 + N_OPT])
        _ts = vals[N_CFG + 1 + N_OPT] if len(vals) > N_CFG + 1 + N_OPT else {}
        apply_tool_shapes(cfg, _ts if isinstance(_ts, dict) else {})
        segs, starts, radii, mold_pts, note = build_channels_cfg(cfg)
        timelines, total_time = synchronize(segs)
        if total_time <= 0:
            total_time = 1.0
        out_dir = tempfile.mkdtemp(prefix='cnc_mp4_')
        path = os.path.join(out_dir, 'cnc_simulation.mp4')
        yield gr.update(), gr.update(), (note + "\n" if note else "") + "MP4 hazırlanıyor..."
        for done, total in mp4_generator(segs, starts, radii, mold_pts, timelines, total_time, mp4_fps, path,
                                         opts['xname'], opts['yname']):
            yield gr.update(), gr.update(), f"MP4 kareleri oluşturuluyor: {done}/{total}"
        yield path, path, f"Tamamlandı: cnc_simulation.mp4 ({total_time:.2f} sn, {mp4_fps} FPS). Aşağıdan indirebilirsin."
    except Exception:
        yield gr.update(), gr.update(), "MP4 oluşturulurken hata oluştu:\n" + traceback.format_exc()


# -------------------------------------------------------------
# 4) DXF çıktısı (AutoCAD'de doğrudan açılır)
# -------------------------------------------------------------
def make_dxf(*vals):
    """vals = all_inputs + [seçili DXF katmanları]"""
    try:
        layers = vals[-1] or []
        if not layers:
            return None, "DXF için en az bir katman seç."
        cfg, fps, speed, ac_s, ac_e, ac_st, opts = _split_vals(vals[:-1])
        segs, starts, radii, mold_pts, note = build_channels_cfg(cfg)
        timelines, total_time = synchronize(segs)
        if total_time <= 0:
            total_time = 1.0
        envs = (_envelopes(timelines, radii, mold_pts, opts, force=True, cfg=cfg)
                if {'envelope', 'preform', 'final'} & set(layers) else None)
        end_used = ac_e if ac_e > ac_s else total_time
        txt, summary = build_dxf_text(layers, segs, radii, mold_pts, timelines, total_time,
                                      ac_s, end_used, ac_st, envs)
        out_dir = tempfile.mkdtemp(prefix='cnc_dxf_')
        path = os.path.join(out_dir, 'cnc_simulasyon.dxf')
        with open(path, 'w', encoding='ascii', newline='\r\n') as f:
            f.write(txt)
        msg = ((note + "\n") if note else "") + "DXF hazır: cnc_simulasyon.dxf (birim mm). Katmanlar: " + "; ".join(summary)
        if 'circles' in layers:
            msg += f"\nÇemberler: {ac_s:g}–{end_used:.2f} sn, adım {ac_st:g} sn."
        return path, msg
    except ValueError as ve:
        return None, f"Hata: {ve}"
    except Exception:
        return None, "DXF oluşturulurken hata oluştu:\n" + traceback.format_exc()


# -------------------------------------------------------------
# 5) JSON kaydet / yükle + Python varsayılan kodu
# -------------------------------------------------------------
def export_json(*vals):
    try:
        cfg, fps, speed, ac_s, ac_e, ac_st, opts = _split_vals(vals)
        data = config_to_json_dict(cfg, fps, speed, ac_s, ac_e, ac_st, opts, opts.get('pf_state'),
                                   opts.get('tool_state'))
        out_dir = tempfile.mkdtemp(prefix='cnc_cfg_')
        path = os.path.join(out_dir, 'gcode_simulator_config.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return path, "Konfigürasyon 'gcode_simulator_config.json' olarak hazırlandı — yukarıdaki dosya kutusundan indirebilirsin."
    except Exception:
        return None, "JSON dışa aktarılırken hata oluştu:\n" + traceback.format_exc()


def import_json(file_obj):
    n_out = N_CFG + 2 + 3 + N_OPT + 1 + 1 + 1
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


# -------------------------------------------------------------
# 6) Makara profili DXF, kalıp DXF → G-kodu, takım aşınması
# -------------------------------------------------------------
def _file_path(f):
    if f is None:
        return None
    return f if isinstance(f, str) else getattr(f, 'name', None) or f['name']


def make_upload_tool(k):
    def _up(file_obj, state, tr):
        state = dict(state or {})
        path = _file_path(file_obj)
        if path is None:
            state.pop(str(k), None)
            return state, "", f"Kanal {k + 1}: makara profili kaldırıldı — takım yarıçaplı daire olarak hesaplanır."
        try:
            entry, msg = load_tool_dxf(path, tr)
            state[str(k)] = entry
            return state, tool_preview_html(entry, tr, CHANNEL_COLORS[k]), f"Kanal {k + 1}: " + msg
        except Exception as e:
            return state, "", f"Kanal {k + 1}: makara DXF'i okunamadı: {e}"
    return _up


def make_tool_previews(state, *trs):
    state = state or {}
    return tuple(tool_preview_html(state.get(str(k)), trs[k], CHANNEL_COLORS[k]) if state.get(str(k)) else ""
                 for k in range(N_CH))


def upload_mold_dxf(file_obj, face, radial_sel, rev, *vals):
    """vals = cfg_comps + [pf_radial]"""
    path = _file_path(file_obj)
    if path is None:
        return (gr.update(),) * 5 + ("Kalıp DXF'i seçilmedi.",)
    try:
        cfg = cfg_from_values(vals[:N_CFG])
        if radial_sel in ('X', 'Y'):
            radial, why = radial_sel, 'seçildi'
        else:
            radial, why = resolve_radial(vals[N_CFG] if len(vals) > N_CFG else 'auto', cfg)
        m, msg = dxf_to_mold(path, face or 'outer', radial, bool(rev))
        return (gr.update(value=m['x0']), gr.update(value=m['y0']), gr.update(value=m['code']),
                gr.update(value='radius'), gr.update(value='X', visible=False), msg)
    except Exception as e:
        return (gr.update(),) * 5 + (f"Kalıp DXF'i çevrilemedi: {e}",)


def export_mold_json(*vals):
    try:
        cfg = cfg_from_values(vals[:N_CFG])
        data = {'type': 'kalip', 'mold': dict(cfg['mold'])}
        out_dir = tempfile.mkdtemp(prefix='cnc_mold_')
        path = os.path.join(out_dir, 'kalip.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return path, ("Kalıp 'kalip.json' olarak hazırlandı. Daha sonra 'JSON Ayar Dosyası Yükle' ile yüklersen "
                      "yalnız kalıp alanları değişir, diğer ayarlar korunur.")
    except Exception:
        return None, "Kalıp JSON'u oluşturulamadı:\n" + traceback.format_exc()


def _wear_compute(vals):
    """vals = all_inputs + [a, b, v_ref, K] → (cfg, radii, sonuçlar, gelen yüzey kaynağı, not, (a, b, vref, K))"""
    a, b, vref, Kc = (_num(vals[-4], 1.0), _num(vals[-3], 0.5), _num(vals[-2], 1000.0) or 1000.0, _num(vals[-1], 0.0))
    cfg, fps, speed, ac_s, ac_e, ac_st, opts = _split_vals(vals[:-4])
    segs, starts, radii, mold_pts, note = build_channels_cfg(cfg)
    timelines, total_time = synchronize(segs)
    if len(mold_pts) < 2:
        raise ValueError("Kalıp noktası yetersiz.")
    form = _envelopes(timelines, radii, mold_pts, opts, force=True, cfg=cfg) if _preform(opts, cfg) else None
    fin = (form or {}).get('final')
    res, src = compute_wear(timelines, radii, mold_pts, fin, opts['env_limit'], a, b, vref, Kc)
    return cfg, radii, res, src, note, (a, b, vref, Kc)


def run_wear_dxf(*vals):
    """vals = all_inputs + [a, b, v_ref, K, en büyük aşınma derinliği (mm)]"""
    try:
        depth = _num(vals[-1], 0.0)
        cfg, radii, res, src, note, (a, b, vref, Kc) = _wear_compute(vals[:-1])
        paths, msg = wear_dxf_files(res, radii, depth, Kc)
        return paths, msg
    except ValueError as e:
        return None, f"Aşınmış profil DXF'i oluşturulamadı: {e}"
    except Exception:
        return None, "Aşınmış profil DXF'i oluşturulurken hata oluştu:\n" + traceback.format_exc()


def run_wear(*vals):
    """vals = all_inputs + [a, b, v_ref, K]"""
    try:
        try:
            cfg, radii, res, src, note, (a, b, vref, Kc) = _wear_compute(vals)
        except ValueError as e:
            return "", str(e), None, f"Aşınma hesaplanamadı: {e}"
        html_parts, rows, md = [], [], []
        md.append(f"**Denklem:** her makara profili noktası için her adımda  dW = K · h^{a:g} · (F / {vref:g})^{b:g} · Δs  \n"
                  f"h = noktanın gelen malzeme yüzeyinin altında kaldığı derinlik (süpürdüğü kalınlık, mm), "
                  f"F = ilerleme hızı (mm/dk), Δs = adımda alınan yol (mm). Gelen yüzey: {src} ile önceki pasların / "
                  f"diğer kanalların bıraktığı yüzeyin küçüğü.")
        for i, w in enumerate(res):
            if w is None:
                continue
            Wm = float(w['W'].max())
            if Wm <= 0:
                html_parts.append(f"<h4>Kanal {i + 1}</h4><p>Makara malzemeye girmiyor — aşınma yok.</p>")
                continue
            html_parts.append(f"<h4>Kanal {i + 1}{' — DXF makara profili' if w['shape'] else ' — daire (R' + format(float(radii[i]), 'g') + ')'}</h4>")
            html_parts.append(wear_figures(i, w, CHANNEL_COLORS[i]))
            ang = _prof_angle(w['B'], w['W'])
            km = int(np.argmax(w['W']))
            band = ang[w['W'] >= 0.5 * Wm]
            line = (f"**Kanal {i + 1}:** en çok aşınan nokta merkeze göre **{ang[km]:.0f}°**; "
                    f"aşınmanın yarıdan fazlası {band.min():.0f}° … {band.max():.0f}° arasında. "
                    f"En derin temas (süpürülen kalınlık) {w['hmax']:.2f} mm. Toplam aşınma indeksi {w['W'].sum():.4g}.")
            if Kc > 0:
                line += f" Tahmini en büyük aşınma: **{Kc * Wm:.4g} µm** (K = {Kc:g})."
            md.append(line)
            tot = sum(w['lines'].values()) or 1.0
            for ln, v in sorted(w['lines'].items(), key=lambda kv: -kv[1])[:12]:
                _cl = cfg['channels'][i]['code'].split('\n')          # segment satır no 0'dan başlar
                code_ln = _cl[ln].strip() if 0 <= ln < len(_cl) else ''
                rows.append({'Kanal': i + 1, 'Satır': ln + 1, 'Program': code_ln, 'Aşınma payı (%)': round(v / tot * 100, 1)})
        md.append("Bu bir **yaklaşım**: a, b ve K gerçek aşınma ölçümleriyle kalibre edilmeli. Şekiller göreli dağılımı gösterir; "
                  "K girilirse en büyük değer µm olarak tahmin edilir. Gelen yüzey ön formun **ilk** halinden alınır: "
                  "simülasyonda malzemenin akıp gittiği yerlerde makaranın yan yüzlerindeki temas fazla görünebilir.")
        df = pd.DataFrame(rows) if rows else None
        return "".join(html_parts) or "<p>Aktif kanal yok.</p>", "\n\n".join(md), df, (note + "\n" if note else "") + "Aşınma hesaplandı."
    except Exception:
        return "", "", None, "Aşınma hesaplanırken hata oluştu:\n" + traceback.format_exc()


def run_process(*vals):
    """vals = all_inputs + [devir, k_fm, η, makara çapı, makara sayısı, β, makine rijitliği]"""
    try:
        n_rpm, kfm, eta, Dt = _num(vals[-7], 0.0), _num(vals[-6], 0.0), _num(vals[-5], 0.15), _num(vals[-4], 0.0)
        z, beta, cst = int(max(_num(vals[-3], 1.0), 1)), _num(vals[-2], 0.0), max(_num(vals[-1], 0.0), 0.0)
        if n_rpm <= 0 or kfm <= 0 or Dt <= 0 or not (0 < eta <= 1):
            return "", "Devir, akma gerilmesi ve makara çapı sıfırdan büyük; η 0 ile 1 arasında olmalı.", None, \
                "Proses hesabı yapılamadı: girdileri kontrol et."
        cfg, fps, speed, ac_s, ac_e, ac_st, opts = _split_vals(vals[:-7])
        segs, starts, radii, mold_pts, note = build_channels_cfg(cfg)
        timelines, total_time = synchronize(segs)
        if len(mold_pts) < 2:
            return "", "Kalıp noktası yetersiz.", None, "Proses hesabı yapılamadı: kalıp yok."
        pf = _preform(opts, cfg)
        form = _envelopes(timelines, radii, mold_pts, opts, force=True, cfg=cfg) if pf else None
        fin = (form or {}).get('final')
        radial = pf['radial'] if pf else resolve_radial(opts.get('pf_radial', 'auto'), cfg)[0]
        ri = 0 if radial == 'X' else 1
        shear = bool(fin) and fin.get('mode') == 'shear'
        res, src, E = compute_process(timelines, radii, mold_pts, fin, opts['env_limit'], ri, n_rpm, kfm, eta, Dt,
                                      z, beta, cst, shear)
        md = [f"**Girdiler:** n = {n_rpm:g} dev/dk, k_fm = {kfm:g} MPa, η = {eta:g}, makara çapı {Dt:g} mm, "
              f"kanal başına {z} makara, parça yarıçapı ekseni {radial}"
              + (f", makine rijitliği {cst:g} kN/mm" if cst > 0 else ", makine rijit kabul edildi")
              + f". Gelen kalınlık: {src}.",
              "**Bağıntılar (Thamasett):** Δh = f·Δs/b,  l_d ≈ √(r_m,CD·Δh),  **F_r ≈ b·l_d·k_fm/η**,  "
              "**F_a ≈ F_r·tan β**,  **F_t ≈ f·Δs·k_fm/η**;  f = F/(n·z) devir başına ilerleme, Δs = s₀ − s₁, "
              "b = temas boyu (profilden ölçülür; b = Δs·cot α)."
              + (" β girilmedi: β = α (etkin giriş açısı, en fazla 45°) alındı." if not beta > 0 else "")]
        if not (0.06 <= eta <= 0.24):
            md.append(f"Not: ders notunda η için 0,06 – 0,24 aralığı veriliyor; girilen {eta:g} bu aralığın dışında.")
        html_parts, rows = [], []
        for i, r in enumerate(res):
            if r is None:
                continue
            e = r['eng']
            if not e.any():
                html_parts.append(f"<h4>Kanal {i + 1}</h4><p>Makara malzemeye girmiyor — kuvvet yok.</p>")
                continue
            html_parts.append(f"<h4>Kanal {i + 1}</h4>" + process_figures(i, r, cst))
            kF = int(np.argmax(r['Fr']))
            fl = r['flow'][e]
            line = (f"**Kanal {i + 1}:** en büyük F_r **{r['Fr'].max() / 1000:.1f} kN** (t = {r['t'][kF]:.1f} s, satır {int(r['line'][kF]) + 1}), "
                    f"F_a {r['Fa'].max() / 1000:.1f} kN, F_t {r['Ft'].max() / 1000:.2f} kN, "
                    f"tork {r['Mt'].max():.0f} N·m, güç **{r['P'].max() / 1000:.1f} kW**. "
                    f"En büyük Δs {r['ds'][e].max():.2f} mm (ezme %{r['eps'][e].max() * 100:.0f}, φ = {r['phi'][e].max():.2f}), "
                    f"f = {r['f'][e].min():.2f}–{r['f'][e].max():.2f} mm/dev, etkin giriş açısı α ≈ {np.degrees(np.median(r['alpha'][e])):.0f}°.")
            if fl.min() < 1.0:
                line += (f" **Akış yönü:** r_m,CD / r_m,LD en az {fl.min():.2f} (< 1) — temasın bir kısmında çevresel akış baskın: "
                         "çap büyümesi / yığılma riski (makara burun yarıçapı büyük ya da makara / kalıp çapı küçük).")
            else:
                line += f" **Akış yönü:** r_m,CD / r_m,LD ≥ {fl.min():.1f} (> 1) — akış ağırlıklı eksenel."
            if cst > 0:
                line += f" Makine esnemesi: boşluk en fazla **{r['dlt'][e].max():.3f} mm** açılıyor."
            if 'sine' in r:
                sr = r['sine'][e & np.isfinite(r['sine'])]
                if len(sr):
                    q = float(np.median(sr))
                    line += (f" **Sinüs kuralı:** boşluk / (s₀·sin α) ortanca {q:.2f}"
                             + (" — boşluk fazla: buruşma riski." if q > 1.05 else
                                " — boşluk dar: kenar (flanş) oluşumu / yığılma." if q < 0.95 else " — sinüs kuralına uygun.")
                             + f" Kalpakcioglu F_t en büyük {np.nanmax(r['FtK']) / 1000:.2f} kN, φ_v ortanca {np.nanmedian(r['phiV'][e]):.2f}.")
            md.append(line)
            for ln in sorted(set(int(v) for v in r['line'][e])):
                m = e & (r['line'] == ln)
                cl = cfg['channels'][i]['code'].split('\n')
                row = {'Kanal': i + 1, 'Satır': ln + 1, 'Program': cl[ln].strip() if 0 <= ln < len(cl) else '',
                       'F_r (kN)': round(float(r['Fr'][m].max()) / 1000, 2), 'F_a (kN)': round(float(r['Fa'][m].max()) / 1000, 2),
                       'F_t (kN)': round(float(r['Ft'][m].max()) / 1000, 3), 'Güç (kW)': round(float(r['P'][m].max()) / 1000, 2),
                       'f (mm/dev)': round(float(np.median(r['f'][m])), 3), 'Δs (mm)': round(float(r['ds'][m].max()), 2),
                       'Ezme (%)': round(float(r['eps'][m].max()) * 100, 1), 'α (°)': round(float(np.degrees(np.median(r['alpha'][m]))), 1),
                       'r_CD/r_LD': round(float(r['flow'][m].min()), 2)}
                if cst > 0:
                    row['Esneme (mm)'] = round(float(r['dlt'][m].max()), 3)
                rows.append(row)
        sm = (fin or {}).get('sim') if fin else None
        if cst > 0 and sm is not None and sm.get('ok') and sm.get('final') is not None:
            d = process_deflection_rays(E, res)
            if sm.get('kind') == 'strip':
                st2 = strip_state(fin, sm, np.where(fin['formed'], np.nan_to_num(fin['F']) + d, np.inf), fin['formed'])
            else:
                st2 = sim_state(fin, sm, sm['F'] + d, sm['formed'])
            nm = (opts['xname'], opts['yname'])
            md.append("**Esnemenin simülasyona etkisi** (boşluk F_r/c kadar açılmış halde yeniden hesaplandı):  \n"
                      f"rijit makine — {sim_xy_text(sm, sm['final'], nm)}  \n"
                      f"esneme dahil — {sim_xy_text(sm, st2, nm)}")
        elif cst > 0:
            md.append("Esnemenin uzamaya etkisi için ön form yükleyip simülasyonu aç.")
        md.append("Bunlar **analitik yaklaşımlardır** (ideal plastik malzeme, sürtünme ve geri yaylanma η içinde): "
                  "kuvvetlerin mertebesini ve satırlar arası karşılaştırmayı verir. k_fm ve η'yı makinede ölçülen "
                  "kuvvet ya da akımla ayarla. Bağıntılar eksenel ilerlemeli flow forming içindir; radyal dalma "
                  "satırlarında ve shear / sac modelinde yaklaşık kalır.")
        df = pd.DataFrame(rows) if rows else None
        return ("".join(html_parts) or "<p>Aktif kanal yok.</p>", "\n\n".join(md), df,
                (note + "\n" if note else "") + "Proses hesapları tamamlandı.")
    except Exception:
        return "", "", None, "Proses hesaplanırken hata oluştu:\n" + traceback.format_exc()


def make_defaults_code(*vals):
    cfg = cfg_from_values(vals[:N_CFG])
    chans = pprint.pformat(cfg['channels'], sort_dicts=False, width=110)
    mold = pprint.pformat(cfg['mold'], sort_dicts=False, width=110)
    return (f"DEFAULT_CHANNELS = {chans}\n\nDEFAULT_MOLD = {mold}\n\n"
            f"DEFAULT_CONTACT_MODE = {cfg['contact_mode']!r}\n\nDEFAULT_G0_FEED = {cfg['g0_feed']!r}")


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
                "mesafe tablosunu, DXF çıktısını ve MP4'ü kullan.")

    with gr.Row():
        # ---------------- SOL: girdiler ----------------
        with gr.Column(scale=4, min_width=380):
            cfg_comps = []
            dia_pairs = []
            tool_files, tool_prevs, tool_trs = [], [], []
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
                        with gr.Accordion("Makara profili (DXF)", open=False):
                            gr.Markdown("Makara kesitini **kapalı kontur** olarak çiz; DXF'teki **0,0 = makara merkezi** "
                                        "(CNC'nin kontrol ettiği nokta). Yüklenirse radüs izi, temas noktası, mesafe ölçümü "
                                        "ve çizimler bu profille yapılır; G41/G42 kompanzasyonu yukarıdaki takım yarıçapıyla yapılır.")
                            tool_file = gr.File(label=f"Kanal {k+1} makara DXF", file_types=[".dxf"], file_count="single")
                            tool_prev = gr.HTML()
                        tool_files.append(tool_file); tool_prevs.append(tool_prev); tool_trs.append(tr)

            with gr.Accordion("Kalıp (Sabit Path)", open=True):
                m_x0 = gr.Number(value=DEFAULT_MOLD['x0'], label="Kalıp Başlangıç X")
                m_y0 = gr.Number(value=DEFAULT_MOLD['y0'], label="Kalıp Başlangıç Y")
                m_code = gr.Textbox(value=DEFAULT_MOLD['code'], lines=5, label="Kalıp G-Kodu")
                m_dmode = gr.Radio(DIA_MODE_CHOICES, value=DEFAULT_MOLD['dia_mode'], label="Kalıp: girdiğim değer türü",
                                   info="Çap seçilirse kalıp G-kodu koordinatları ve başlangıç değeri 2'ye bölünür.")
                m_daxis = gr.Radio(DIA_AXIS_CHOICES, value=DEFAULT_MOLD['dia_axis'], label="Kalıp: çap hangi eksende?",
                                   visible=(DEFAULT_MOLD['dia_mode'] == 'diameter'))
                dia_pairs.append((m_dmode, m_daxis))
                with gr.Accordion("Kalıbı DXF'ten oluştur", open=False):
                    gr.Markdown("Kalıp kesitini DXF olarak yükle: çizgiler doğru + yay olarak **kalıp G-koduna** (G1 / G2 / G3 R) "
                                "çevrilip yukarıdaki alanlara yazılır. Açık çizgi olduğu gibi alınır; kapalı konturda seçilen yüz alınır. "
                                "Saklamak için **Kalıbı JSON olarak indir** — sonra 'JSON Ayar Dosyası Yükle' ile geri yükle.")
                    mold_file = gr.File(label="Kalıp DXF yükle", file_types=[".dxf"], file_count="single")
                    with gr.Row():
                        mold_face = gr.Dropdown(MOLD_FACE_CHOICES, value='outer', label="Kapalı konturda kullanılacak yüz")
                        mold_radial = gr.Dropdown([('Otomatik (ön form ayarı)', 'auto'), ('X', 'X'), ('Y', 'Y')], value='auto',
                                                  label="Parça yarıçapı ekseni")
                        mold_rev = gr.Checkbox(value=False, label="Yönü ters çevir")
                    mold_json_btn = gr.Button("💾 Kalıbı JSON olarak indir")
                    mold_json_out = gr.File(label="Kalıp JSON", interactive=False)
                contact_mode = gr.Dropdown(choices=CONTACT_MODE_CHOICES, value=DEFAULT_CONTACT_MODE,
                                           label="Çizilen yol")
                cfg_comps += [m_x0, m_y0, m_code, m_dmode, m_daxis, contact_mode]

            g0_feed = gr.Number(value=DEFAULT_G0_FEED, label="G0 (hızlı hareket) hızı (mm/dk)",
                                info="G0 satırında F yazılmamışsa bu hız kullanılır.")
            cfg_comps.append(g0_feed)

            with gr.Accordion("Grafik eksen adları", open=False):
                with gr.Row():
                    x_name = gr.Textbox(value=DEFAULT_XNAME, label="Yatay eksen adı", max_lines=1)
                    y_name = gr.Textbox(value=DEFAULT_YNAME, label="Dikey eksen adı", max_lines=1)
                gr.Markdown("Varsayılan X ve Y. Grafiklerde görünen adı değiştirir (G-kodundaki X/Y harfleri aynı kalır). "
                            "Enter'a basınca grafikler güncellenir; animasyon / oynatıcı / MP4'e bir sonraki çalıştırmada yansır.")

            with gr.Accordion("Radüs izi (form yüzeyi)", open=True):
                gr.Markdown("Takım dairesinin kalıba bakan zarfı: takım ilerlemese, sadece yanaşsa bile uç radüsünün "
                            "bıraktığı iz. Kalıp ile bu iz arasındaki taralı bölge inceltilmiş malzemedir.")
                with gr.Row():
                    show_env = gr.Checkbox(value=DEFAULT_SHOW_ENV, label="Radüs izini göster")
                    env_limit = gr.Number(value=DEFAULT_ENV_LIMIT, label="İz sınırı / ham malzeme kalınlığı (mm)",
                                          info="Kalıba bundan uzak kalan yerlerde iz çizilmez. 0 = takım yarıçapı.")

            with gr.Accordion("Ön form (DXF) — lokal malzeme kalınlığı", open=False):
                gr.Markdown("Ön formun kesitini **kalıp koordinatlarında**, **kapalı kontur** (iç + dış yüzey) olarak "
                            "DXF'e çizip yükle. **Shear forming** modelinde kalınlık yalnız DXF'ten alınır: her yarıçapta "
                            "ön formun eksen yönündeki kalınlığı t₀ okunur, malzeme aynı yarıçapta kalıba sıvanır ve "
                            "ideal et kalınlığı **t₀·sin α** olur (α: kalıp yüzeyi ile dönme ekseni arası açı). Gerçek et "
                            "kalınlığı takım boşluğudur; fark aşırı / eksik inceltme olarak gösterilir. "
                            "Okunan nesneler: LINE, POLYLINE, LWPOLYLINE, ARC, CIRCLE, ELLIPSE, SPLINE (bloklar için EXPLODE).")
                pf_file = gr.File(label="Ön form DXF yükle", file_types=[".dxf"], file_count="single")
                pf_layers = gr.CheckboxGroup(choices=[], value=[], label="Kullanılacak DXF katmanları")
                pf_model = gr.Dropdown(PF_MODEL_CHOICES, value='strip', label="Ön form modeli")
                pf_radial = gr.Dropdown(PF_RADIAL_CHOICES, value='auto', label="Parça yarıçapı ekseni",
                                        info="Hacim 2π·yarıçap·alan ile hesaplanır. Dönme ekseni bu eksenin 0 çizgisidir.")
                pf_sim = gr.Checkbox(value=False, label="Simülasyon modu: ön form adım adım şekillensin (hacim sabit)",
                                     info="Makaranın geçtiği yer kalıba sıvanıp takım boşluğu kalınlığına iner; artan hacim "
                                          "kalan malzemeyi ileri iterek uçta uzama olarak görünür. Hızlı Oynatıcıda canlı izlenir.")
                with gr.Row():
                    pf_use = gr.Checkbox(value=True, label="Ön formu kullan")
                    pf_dx = gr.Number(value=0.0, label="Ön form kaydırma X (mm)")
                    pf_dy = gr.Number(value=0.0, label="Ön form kaydırma Y (mm)")
            pf_state = gr.State(None)
            tool_state = gr.State({})

            with gr.Row():
                fps_sl = gr.Slider(5, 40, value=20, step=1, label="FPS")
                speed_sl = gr.Slider(0.2, 5.0, value=1.0, step=0.2, label="Hız Çarpanı")

            run_btn = gr.Button("▶ Simülasyonu ve Analizleri Çalıştır", variant="primary")

            with gr.Accordion("Konfigürasyon Yönetimi (JSON)", open=True):
                gr.Markdown("3 kanalın ve kalıbın tüm ayarlarını (başlangıç, yarıçap, F, Y offset, G-kodları, "
                            "çizilen yol modu, G0 hızı, FPS/hız, DXF çember zaman ayarları, grafik ve radüs izi ayarları) "
                            "tek JSON dosyası olarak kaydet / yükle.")
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
                                "1× = gerçek zamanlı simülasyon süresi. Sağdaki panelde takım–kalıp mesafesi canlı görünür; "
                                "**Ölçümü Başlat / Bitir** ile iki an arasındaki en küçük ve en büyük mesafe sabitlenir. "
                                "Alttaki CNC ekranında her kanalın çalışan program satırı (vurgulu), eksen konumu ve "
                                "bloğun bitmesine kalan mesafe görünür. Üst çubuktaki **Radüs izi** ile form yüzeyi "
                                "anlık olarak açılıp kapatılır.")
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
                    with gr.Row():
                        lbl_font = gr.Slider(4, 20, value=DEFAULT_LABEL_FONT, step=1, label="Mesafe yazı boyutu")
                        lbl_shift = gr.Slider(-60, 60, value=DEFAULT_LABEL_SHIFT, step=0.5,
                                              label="Tüm yazıları kaydır (mm): + yukarı / − aşağı")
                    redraw_btn = gr.Button("Grafikleri yeniden çiz")
                    with gr.Tabs():
                        with gr.Tab("Kanal 1"):
                            ind1 = gr.HTML()
                        with gr.Tab("Kanal 2"):
                            ind2 = gr.HTML()
                        with gr.Tab("Kanal 3"):
                            ind3 = gr.HTML()

                with gr.Tab("📊 Et Kalınlığı"):
                    gr.Markdown("Ön form yüklüyse, kalıp konturu boyunca et kalınlığı (kalıba dik). Shear forming modelinde: "
                                "ön form eksenel kalınlığı t₀, ideal t₀·sin α ve gerçek kalınlık (3 kanalın en dar takım "
                                "boşluğu) ile sapma; + değer sinüs kuralından kalın (eksik inceltme), − değer ince (aşırı inceltme).")
                    thick_html = gr.HTML()
                    thick_md = gr.Markdown()
                    thick_df = gr.DataFrame(label="Kalınlık tablosu (~1 mm aralıkla)")

                with gr.Tab("📏 Mesafe Tablosu"):
                    gr.Markdown("Her hareket segmentinin başlangıç / orta / bitiş noktasının kalıp konturuna olan "
                                "en kısa mesafesi (G0 hariç).")
                    table_out = gr.DataFrame()

                with gr.Tab("📐 DXF Çıktısı"):
                    gr.Markdown("Seçilen katmanları **DXF** dosyası olarak verir (AutoCAD R12 formatı, birim mm). "
                                "Her kanal ve her çizim türü ayrı katmandadır (örn. `K1_KONTUR`, `K2_RADUS_IZI`, "
                                "`K3_CEMBER`, `KALIP`); AutoCAD'de katmanları ayrı ayrı açıp kapatabilirsin.")
                    dxf_layers = gr.CheckboxGroup(DXF_LAYER_CHOICES, value=DXF_DEFAULT_LAYERS, label="Katmanlar")
                    gr.Markdown("**Zaman adımlı takım çemberleri:** seçilen aralıkta her adımda takım merkezine "
                                "takım yarıçapında çember. Bitiş 0 (veya Başlangıç'tan küçük) ise toplam süre kullanılır.")
                    with gr.Row():
                        ac_start = gr.Number(value=0.0, label="Başlangıç Zamanı (s)")
                        ac_end = gr.Number(value=0.0, label="Bitiş Zamanı (s)  [0 = otomatik]")
                        ac_step = gr.Number(value=0.1, label="Zaman Adımı (s)")
                    dxf_btn = gr.Button("📐 DXF Oluştur", variant="primary")
                    dxf_file = gr.File(label="DXF indir", interactive=False)

                with gr.Tab("📐 Proses Hesapları"):
                    gr.Markdown("Takım yolu boyunca **şekillendirme kuvvetleri, tork, güç, akış yönü ve makine esnemesi** — "
                                "Thamasett bağıntılarıyla (F_r ≈ b·l_d·k_fm/η, F_a ≈ F_r·tan β, F_t ≈ f·Δs·k_fm/η). "
                                "Δs ve temas boyu simülasyondaki makara–malzeme temasından alınır. Makine rijitliği girilirse "
                                "boşluğun F_r/c kadar açıldığı kabul edilir ve uzama buna göre yeniden hesaplanır.")
                    with gr.Row():
                        pr_n = gr.Number(value=300.0, label="İş mili devri n (dev/dk)")
                        pr_kf = gr.Number(value=300.0, label="Ortalama akma gerilmesi k_fm (MPa)")
                        pr_eta = gr.Number(value=0.15, label="Verim η (0,06 – 0,24)")
                        pr_D = gr.Number(value=250.0, label="Makara çapı D (mm)")
                    with gr.Row():
                        pr_z = gr.Number(value=1, precision=0, label="Kanal başına makara sayısı z")
                        pr_beta = gr.Number(value=0.0, label="β (°) — 0: giriş açısı α alınır")
                        pr_c = gr.Number(value=0.0, label="Makine rijitliği c (kN/mm) — 0: rijit")
                    pr_btn = gr.Button("📐 Proses hesaplarını yap", variant="primary")
                    pr_md = gr.Markdown()
                    pr_html = gr.HTML()
                    pr_df = gr.DataFrame(label="Program satırlarına göre en büyük değerler")

                with gr.Tab("🔥 Takım Aşınması"):
                    gr.Markdown("Makara profilinin her noktasının ne kadar malzeme süpürdüğüne, ne kadar yol aldığına ve "
                                "hangi hızla gittiğine göre **yaklaşık** aşınma:  **dW = K · hᵃ · (F / F_ref)ᵇ · Δs**. "
                                "h: profil noktasının gelen malzeme yüzeyinin altında kaldığı derinlik (mm), F: ilerleme hızı, "
                                "Δs: alınan yol. Sonuç profil üzerinde ısı haritası, zaman–konum ısı haritası ve program satırı "
                                "payları olarak gösterilir. Ön form yüklüyse gelen yüzey ön formdan alınır.")
                    with gr.Row():
                        wear_a = gr.Number(value=1.0, label="a — kalınlık üssü")
                        wear_b = gr.Number(value=0.5, label="b — hız üssü")
                        wear_vref = gr.Number(value=1000.0, label="F_ref (mm/dk)")
                        wear_K = gr.Number(value=0.0, label="K (µm / mm^(a+1)) — 0: yalnız göreli")
                    wear_btn = gr.Button("🔥 Aşınmayı hesapla", variant="primary")
                    wear_md = gr.Markdown()
                    wear_html = gr.HTML()
                    wear_df = gr.DataFrame(label="Aşınmaya en çok katkı veren program satırları")
                    gr.Markdown("**Aşınmış makara profili (DXF):** profilin her noktası aşınması oranında içeri alınır. "
                                "Ölçek için en çok aşınan noktanın derinliğini gir (0 bırakırsan K kullanılır). "
                                "Kanal dosyaları 0,0 merkezlidir; makara DXF'i olarak geri yüklenebilir.")
                    with gr.Row():
                        wear_depth = gr.Number(value=0.5, label="En büyük aşınma derinliği (mm) — 0: K'ya göre")
                        wear_dxf_btn = gr.Button("📐 Aşınmış profili DXF indir")
                    wear_dxf_out = gr.File(label="Aşınmış makara profili DXF", interactive=False, file_count="multiple")

    # ---------------- OLAYLAR ----------------
    opt_comps = [lbl_font, lbl_shift, x_name, y_name, show_env, env_limit,
                 pf_use, pf_dx, pf_dy, pf_layers, pf_model, pf_radial, pf_sim] # sıra: opts_from_values
    all_inputs = cfg_comps + [fps_sl, speed_sl, ac_start, ac_end, ac_step] + opt_comps + [pf_state, tool_state]

    run_btn.click(
        run_all, inputs=all_inputs,
        outputs=[anim_img, frame_sl, play_timer, anim_token, comp_html, ind1, ind2, ind3,
                 table_out, player_out, thick_html, thick_md, thick_df, status])

    # statik grafikleri yeniden çiz (yazı boyutu / kaydırma / eksen adı / radüs izi)
    redraw_outputs = [comp_html, ind1, ind2, ind3, thick_html, thick_md, thick_df, status]
    for _c in (lbl_font, lbl_shift):
        _c.release(redraw_plots, inputs=all_inputs, outputs=redraw_outputs)
    for _c in (x_name, y_name, env_limit, pf_dx, pf_dy):
        _c.submit(redraw_plots, inputs=all_inputs, outputs=redraw_outputs)
    for _c in (show_env, pf_use, pf_layers, pf_model, pf_radial, pf_sim):
        _c.input(redraw_plots, inputs=all_inputs, outputs=redraw_outputs)

    # ön form DXF
    pf_file.upload(upload_preform, inputs=pf_file, outputs=[pf_state, pf_layers, status])
    pf_file.clear(lambda: upload_preform(None), outputs=[pf_state, pf_layers, status])
    redraw_btn.click(redraw_plots, inputs=all_inputs, outputs=redraw_outputs)

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
    mp4_btn.click(make_mp4, inputs=cfg_comps + [mp4_fps] + opt_comps + [tool_state],
                  outputs=[video_out, mp4_file, status])

    # DXF
    dxf_btn.click(make_dxf, inputs=all_inputs + [dxf_layers], outputs=[dxf_file, status])

    # JSON
    json_btn.click(export_json, inputs=all_inputs, outputs=[json_file_out, status])
    json_upload.upload(import_json, inputs=json_upload, outputs=all_inputs + [status]).then(
        make_tool_previews, inputs=[tool_state] + tool_trs, outputs=tool_prevs)

    # makara profili DXF (kanal başına)
    for _k in range(N_CH):
        tool_files[_k].upload(make_upload_tool(_k), inputs=[tool_files[_k], tool_state, tool_trs[_k]],
                              outputs=[tool_state, tool_prevs[_k], status])
        tool_files[_k].clear(lambda st, tr, _k=_k: make_upload_tool(_k)(None, st, tr),
                             inputs=[tool_state, tool_trs[_k]], outputs=[tool_state, tool_prevs[_k], status])

    # kalıp DXF → G-kodu, kalıp JSON
    mold_file.upload(upload_mold_dxf, inputs=[mold_file, mold_face, mold_radial, mold_rev] + cfg_comps + [pf_radial],
                     outputs=[m_x0, m_y0, m_code, m_dmode, m_daxis, status])
    mold_json_btn.click(export_mold_json, inputs=cfg_comps, outputs=[mold_json_out, status])

    # takım aşınması
    wear_btn.click(run_wear, inputs=all_inputs + [wear_a, wear_b, wear_vref, wear_K],
                   outputs=[wear_html, wear_md, wear_df, status])
    pr_btn.click(run_process, inputs=all_inputs + [pr_n, pr_kf, pr_eta, pr_D, pr_z, pr_beta, pr_c],
                 outputs=[pr_html, pr_md, pr_df, status])
    wear_dxf_btn.click(run_wear_dxf, inputs=all_inputs + [wear_a, wear_b, wear_vref, wear_K, wear_depth],
                       outputs=[wear_dxf_out, status])

    # Çap/yarıçap: eksen seçimi yalnızca "Çap" seçilince görünür
    for _dm, _da in dia_pairs:
        _dm.change(lambda m: gr.update(visible=(m == 'diameter')), inputs=_dm, outputs=_da)

    # Python varsayılan kodu
    defaults_btn.click(make_defaults_code, inputs=cfg_comps, outputs=defaults_out)

demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), auth=(APP_USER, APP_PASSWORD))
