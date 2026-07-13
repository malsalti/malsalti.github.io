"""Generate three original research-style website thumbnails.

All trajectories are deterministic illustrative simulations made for the
website. They are inspired by the research topics but do not reproduce data or
figures from published papers.
"""

from pathlib import Path

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "figures-preview"
W = H = 800

BG = HexColor("#FBFBF8")
TEXT = HexColor("#263238")
MUTED = HexColor("#7B898F")
GRID = HexColor("#DCE6E9")
ACCENT = HexColor("#36758A")
TEAL = HexColor("#2A8C82")
AMBER = HexColor("#D59A31")
PALE = HexColor("#EAF3F5")
NAVY = HexColor("#173E52")
ROSE = HexColor("#B86B65")


def new_page(path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=(W, H))
    c.setFillColor(BG)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(TEXT)
    c.setFont("Helvetica", 12)
    c.drawString(66, 744, title)
    c.setStrokeColor(GRID)
    c.setLineWidth(0.9)
    c.line(66, 730, 736, 730)
    return c


def finish(c, footer):
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 10)
    c.drawRightString(736, 38, footer)
    c.showPage()
    c.save()


def polyline(c, points, color, width=1.4, alpha=1.0, dash=None):
    if len(points) < 2:
        return
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setStrokeAlpha(alpha)
    if dash:
        c.setDash(*dash)
    p = c.beginPath()
    p.moveTo(*points[0])
    for point in points[1:]:
        p.lineTo(*point)
    c.drawPath(p, stroke=1, fill=0)
    c.restoreState()


def arrow_head(c, tip, direction, color, size=9, alpha=1.0):
    direction = np.asarray(direction, dtype=float)
    direction /= max(np.linalg.norm(direction), 1e-9)
    normal = np.array([-direction[1], direction[0]])
    base = np.asarray(tip) - size * direction
    left = base + 0.48 * size * normal
    right = base - 0.48 * size * normal
    p = c.beginPath()
    p.moveTo(*tip)
    p.lineTo(*left)
    p.lineTo(*right)
    p.close()
    c.saveState()
    c.setFillColor(color)
    c.setFillAlpha(alpha)
    c.drawPath(p, stroke=0, fill=1)
    c.restoreState()


def map_xy(x, y, box, xlim, ylim):
    x0, y0, x1, y1 = box
    px = x0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
    py = y0 + (y - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
    return np.asarray(px), np.asarray(py)


def axes(c, box, xlabel, ylabel, x_ticks=(), y_ticks=()):
    x0, y0, x1, y1 = box
    c.setStrokeColor(GRID)
    c.setLineWidth(0.8)
    c.line(x0, y0, x1, y0)
    c.line(x0, y0, x0, y1)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 10)
    c.drawCentredString((x0 + x1) / 2, y0 - 28, xlabel)
    c.saveState()
    c.translate(x0 - 38, (y0 + y1) / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()
    for pos, label in x_ticks:
        c.line(pos, y0 - 3, pos, y0 + 3)
        c.drawCentredString(pos, y0 - 15, label)
    for pos, label in y_ticks:
        c.line(x0 - 3, pos, x0 + 3, pos)
        c.drawRightString(x0 - 7, pos - 3, label)


def nonlinear_figure():
    path = OUT / "nonlinear-control-research-v1.pdf"
    c = new_page(path, "nonlinear trajectories and data-driven stabilization")

    box = (102, 112, 708, 686)
    xlim = (-2.2, 2.2)
    ylim = (-2.0, 2.0)

    # Potential-energy contours provide a restrained nonlinear background.
    levels = [0.32, 0.62, 1.02, 1.50, 2.05]
    for idx, level in enumerate(levels):
        top, bottom = [], []
        for x in np.linspace(-2.05, 2.05, 220):
            remaining = 2 * (level - 0.5 * x * x - 0.055 * x**4)
            if remaining >= 0:
                y = np.sqrt(remaining)
                top.append(map_xy(x, y, box, xlim, ylim))
                bottom.append(map_xy(x, -y, box, xlim, ylim))
        if top:
            points = [(float(a), float(b)) for a, b in top]
            points += [(float(a), float(b)) for a, b in bottom[::-1]]
            polyline(c, points + [points[0]], ACCENT, width=0.75, alpha=0.12 + 0.025 * idx)

    # Vector field for a controlled Duffing-type system.
    gx = np.linspace(-1.8, 1.8, 11)
    gy = np.linspace(-1.55, 1.55, 9)
    for x in gx:
        for y in gy:
            dx = y
            dy = -1.25 * x - 0.28 * x**3 - 1.05 * y
            norm = max(np.hypot(dx, dy), 1e-9)
            dx, dy = dx / norm, dy / norm
            px, py = map_xy(x, y, box, xlim, ylim)
            ex, ey = map_xy(x + 0.105 * dx, y + 0.105 * dy, box, xlim, ylim)
            polyline(c, [(px, py), (ex, ey)], MUTED, width=0.75, alpha=0.38)
            arrow_head(c, (ex, ey), (ex - px, ey - py), MUTED, size=3.8, alpha=0.38)

    def flow(initial, steps=180, dt=0.035):
        state = np.asarray(initial, dtype=float)
        result = [state.copy()]
        for _ in range(steps):
            x, y = state
            deriv = np.array([y, -1.25 * x - 0.28 * x**3 - 1.05 * y])
            mid = state + 0.5 * dt * deriv
            xm, ym = mid
            deriv_mid = np.array([ym, -1.25 * xm - 0.28 * xm**3 - 1.05 * ym])
            state = state + dt * deriv_mid
            result.append(state.copy())
        return np.asarray(result)

    initials = [(-1.75, 1.25), (-1.6, -1.35), (-0.75, 1.65), (1.8, 1.05), (1.65, -1.45), (0.6, -1.7)]
    colors = [NAVY, ACCENT, TEAL, NAVY, ACCENT, TEAL]
    for idx, (initial, color) in enumerate(zip(initials, colors)):
        tr = flow(initial)
        px, py = map_xy(tr[:, 0], tr[:, 1], box, xlim, ylim)
        pts = list(zip(px, py))
        polyline(c, pts, color, width=2.1 if idx in {1, 3} else 1.45, alpha=0.82)
        for j in range(0, len(pts), 18):
            c.setFillColor(color)
            c.setStrokeColor(BG)
            c.setLineWidth(0.55)
            c.circle(pts[j][0], pts[j][1], 2.8, stroke=1, fill=1)

    # Dashed data-certified region and equilibrium.
    t = np.linspace(0, 2 * np.pi, 180)
    rx, ry = 1.92 * np.cos(t), 1.55 * np.sin(t)
    px, py = map_xy(rx, ry, box, xlim, ylim)
    polyline(c, list(zip(px, py)), AMBER, width=1.3, alpha=0.72, dash=(5, 4))
    ox, oy = map_xy(0, 0, box, xlim, ylim)
    c.setFillColor(NAVY)
    c.setStrokeColor(BG)
    c.setLineWidth(1.2)
    c.circle(ox, oy, 6, stroke=1, fill=1)

    axes(c, box, "state coordinate", "state derivative")
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(498, 627, "data-certified region")
    c.setFillColor(ACCENT)
    c.setFont("Helvetica", 10)
    c.drawString(118, 93, "measured trajectories")
    finish(c, "data  ->  nonlinear structure  ->  stabilization")


def predictive_figure():
    path = OUT / "predictive-learning-research-v1.pdf"
    c = new_page(path, "prediction and learning from one offline experiment")

    top = (92, 438, 710, 690)
    axes(c, top, "prediction step", "output")
    x0, y0, x1, y1 = top

    # Constraint band.
    upper = map_xy(0, 1.1, top, (-8, 20), (-1.35, 1.35))[1]
    lower = map_xy(0, -1.1, top, (-8, 20), (-1.35, 1.35))[1]
    c.saveState()
    c.setFillColor(PALE)
    c.setFillAlpha(0.72)
    c.rect(x0, lower, x1 - x0, upper - lower, stroke=0, fill=1)
    c.restoreState()
    polyline(c, [(x0, upper), (x1, upper)], MUTED, width=0.8, alpha=0.5, dash=(4, 4))
    polyline(c, [(x0, lower), (x1, lower)], MUTED, width=0.8, alpha=0.5, dash=(4, 4))

    # Measured past and fan of candidate predictions.
    past_t = np.arange(-8, 1)
    past_y = 0.58 * np.sin(0.48 * past_t + 1.1) + 0.035 * past_t + 0.35
    px, py = map_xy(past_t, past_y, top, (-8, 20), (-1.35, 1.35))
    polyline(c, list(zip(px, py)), NAVY, width=2.0, alpha=0.9)
    for xv, yv in zip(px, py):
        c.setFillColor(NAVY)
        c.circle(xv, yv, 3.0, stroke=0, fill=1)

    future_t = np.arange(0, 21)
    start = past_y[-1]
    candidate_endpoints = []
    for idx, gain in enumerate(np.linspace(0.07, 0.24, 13)):
        pred = start * np.exp(-gain * future_t) + 0.13 * np.sin((0.32 + 0.012 * idx) * future_t) * np.exp(-0.09 * future_t)
        qx, qy = map_xy(future_t, pred, top, (-8, 20), (-1.35, 1.35))
        polyline(c, list(zip(qx, qy)), ACCENT, width=0.8, alpha=0.18 + idx * 0.014)
        candidate_endpoints.append((qx[-1], qy[-1]))

    selected = start * np.exp(-0.19 * future_t) + 0.07 * np.sin(0.39 * future_t) * np.exp(-0.10 * future_t)
    qx, qy = map_xy(future_t, selected, top, (-8, 20), (-1.35, 1.35))
    polyline(c, list(zip(qx, qy)), TEAL, width=2.5, alpha=0.96)
    for idx in range(0, len(qx), 3):
        c.setFillColor(TEAL)
        c.setStrokeColor(BG)
        c.setLineWidth(0.6)
        c.circle(qx[idx], qy[idx], 3.1, stroke=1, fill=1)

    present_x = map_xy(0, 0, top, (-8, 20), (-1.35, 1.35))[0]
    polyline(c, [(present_x, y0), (present_x, y1)], MUTED, width=0.9, alpha=0.6, dash=(3, 3))
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(110, 665, "measured past")
    c.drawString(482, 665, "candidate futures")

    # Lower panel: a Q-function landscape and off-policy updates.
    lower_box = (120, 102, 680, 374)
    center = np.array([515.0, 232.0])
    angle = np.deg2rad(18)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for idx, (rx, ry) in enumerate([(205, 115), (160, 88), (120, 65), (80, 43), (42, 23)]):
        theta = np.linspace(0, 2 * np.pi, 150)
        pts = np.column_stack([rx * np.cos(theta), ry * np.sin(theta)]) @ rot.T + center
        polyline(c, [tuple(p) for p in pts], ACCENT, width=0.9, alpha=0.17 + 0.055 * idx)

    # Offline data cloud on the left.
    rng = np.random.default_rng(29)
    cloud = rng.normal(size=(70, 2)) @ np.array([[46, 8], [-5, 36]]) + np.array([200, 230])
    for idx, (cx, cy) in enumerate(cloud):
        c.setFillColor(NAVY if idx % 3 else ACCENT)
        c.setFillAlpha(0.28 + 0.12 * (idx % 3))
        c.circle(cx, cy, 2.1, stroke=0, fill=1)
    c.setFillAlpha(1)

    updates = np.array([[228, 148], [294, 184], [352, 202], [403, 222], [446, 228], [479, 231], [515, 232]])
    polyline(c, [tuple(p) for p in updates], TEAL, width=2.2, alpha=0.95)
    for idx, (ux, uy) in enumerate(updates):
        c.setFillColor(AMBER if idx < 3 else TEAL)
        c.setStrokeColor(BG)
        c.setLineWidth(0.7)
        c.circle(ux, uy, 4.2, stroke=1, fill=1)
    arrow_head(c, tuple(updates[-1]), updates[-1] - updates[-2], TEAL, size=10)

    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(140, 337, "offline input-output data")
    c.drawString(474, 337, "learned cost geometry")
    c.setFillColor(NAVY)
    c.circle(center[0], center[1], 5.2, stroke=0, fill=1)
    finish(c, "offline data  ->  prediction / learning  ->  feedback")


def biomedical_figure():
    path = OUT / "biomedical-control-research-v1.pdf"
    c = new_page(path, "illustrative virtual-patient closed-loop responses")

    rng = np.random.default_rng(41)
    time = np.linspace(0, 120, 121)
    n_patients = 28
    target = 75.0
    bp = []
    infusion = []
    for _ in range(n_patients):
        initial = rng.uniform(43, 63)
        tau = rng.uniform(20, 43)
        oscillation = rng.uniform(0.8, 3.2) * np.exp(-time / rng.uniform(45, 78)) * np.sin(time / rng.uniform(8, 14) + rng.uniform(0, 1.8))
        response = target - (target - initial) * np.exp(-time / tau) + oscillation
        response += rng.normal(scale=0.22, size=time.size)
        gain = rng.uniform(0.12, 0.22)
        rate = np.clip(gain * (target - response) + rng.uniform(0.2, 0.7) * np.exp(-time / 55), 0, 7.5)
        bp.append(response)
        infusion.append(rate)
    bp = np.asarray(bp)
    infusion = np.asarray(infusion)

    top = (104, 398, 708, 688)
    axes(c, top, "time [min]", "mean arterial pressure [mmHg]")
    x0, y0, x1, y1 = top
    band_low = map_xy(0, 72, top, (0, 120), (40, 86))[1]
    band_high = map_xy(0, 78, top, (0, 120), (40, 86))[1]
    c.saveState()
    c.setFillColor(PALE)
    c.setFillAlpha(0.85)
    c.rect(x0, band_low, x1 - x0, band_high - band_low, stroke=0, fill=1)
    c.restoreState()
    target_y = map_xy(0, target, top, (0, 120), (40, 86))[1]
    polyline(c, [(x0, target_y), (x1, target_y)], MUTED, width=1.0, alpha=0.65, dash=(4, 3))

    palette = [ACCENT, TEAL, NAVY, ROSE, AMBER]
    for idx, trace in enumerate(bp):
        px, py = map_xy(time, trace, top, (0, 120), (40, 86))
        polyline(c, list(zip(px, py)), palette[idx % len(palette)], width=0.65, alpha=0.18)
    median_bp = np.median(bp, axis=0)
    px, py = map_xy(time, median_bp, top, (0, 120), (40, 86))
    polyline(c, list(zip(px, py)), NAVY, width=2.4, alpha=0.95)
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(536, target_y + 9, "target range")

    bottom = (104, 104, 708, 318)
    axes(c, bottom, "time [min]", "fluid infusion rate")
    for idx, trace in enumerate(infusion):
        px, py = map_xy(time, trace, bottom, (0, 120), (0, 7.5))
        polyline(c, list(zip(px, py)), palette[idx % len(palette)], width=0.65, alpha=0.17)
    median_u = np.median(infusion, axis=0)
    px, py = map_xy(time, median_u, bottom, (0, 120), (0, 7.5))
    polyline(c, list(zip(px, py)), TEAL, width=2.4, alpha=0.96)

    # Measurement-to-treatment feedback cue, kept plot-like and unobtrusive.
    c.saveState()
    c.setStrokeColor(ACCENT)
    c.setStrokeAlpha(0.55)
    c.setLineWidth(1.6)
    p = c.beginPath()
    p.moveTo(702, 430)
    p.curveTo(754, 378, 752, 250, 704, 204)
    c.drawPath(p, stroke=1, fill=0)
    c.restoreState()
    arrow_head(c, (704, 204), (-0.55, -0.75), ACCENT, size=9, alpha=0.65)

    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(124, 667, "virtual-patient cohort")
    c.drawString(124, 293, "patient-specific control actions")
    finish(c, "blood-pressure feedback  ->  patient-specific infusion")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nonlinear_figure()
    predictive_figure()
    biomedical_figure()


if __name__ == "__main__":
    main()
