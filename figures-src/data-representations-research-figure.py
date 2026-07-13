"""Create an original research-style thumbnail for data-based control.

The signals and behavioral projection are deterministic simulations created
specifically for the website; they do not reproduce a published figure.
"""

from pathlib import Path

import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "tmp" / "figures-preview" / "data-representations-research-v1.pdf"

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


def simulate_data():
    rng = np.random.default_rng(17)
    n_steps = 54
    k = np.arange(n_steps)

    a = np.array([
        [0.78, 0.16, 0.00, 0.05],
        [-0.10, 0.74, 0.18, 0.00],
        [0.04, 0.11, 0.72, 0.17],
        [0.08, 0.00, -0.09, 0.70],
    ])
    b = np.array([
        [0.34, 0.05],
        [0.10, 0.28],
        [0.18, -0.08],
        [0.04, 0.24],
    ])
    c = np.array([
        [1.0, 0.0, 0.25, 0.0],
        [0.0, 0.65, 0.0, 0.75],
    ])

    u = np.vstack([
        0.72 * np.sin(0.31 * k) + 0.27 * np.sin(0.87 * k + 0.4),
        0.55 * np.cos(0.23 * k + 0.5) + 0.20 * np.sin(0.69 * k),
    ])
    u += 0.055 * rng.normal(size=u.shape)

    x = np.zeros((4, n_steps + 1))
    x[:, 0] = np.array([0.3, -0.2, 0.16, 0.08])
    for idx in range(n_steps):
        x[:, idx + 1] = a @ x[:, idx] + b @ u[:, idx]
    y = c @ x[:, :n_steps] + 0.035 * rng.normal(size=(2, n_steps))

    w = np.vstack([u, y])
    depth = 6
    hankel = np.column_stack([
        w[:, start : start + depth].reshape(-1, order="F")
        for start in range(n_steps - depth + 1)
    ])
    hankel -= hankel.mean(axis=1, keepdims=True)
    _, _, vh = np.linalg.svd(hankel, full_matrices=False)
    scores = vh[:2].T
    scores /= np.max(np.abs(scores), axis=0, keepdims=True)

    missing = np.zeros(n_steps, dtype=bool)
    missing[[7, 8, 17, 25, 26, 39, 47]] = True
    return k, u, y, scores, missing


def set_alpha(c, fill=None, stroke=None):
    if fill is not None:
        c.setFillAlpha(fill)
    if stroke is not None:
        c.setStrokeAlpha(stroke)


def draw_polyline(c, points, color, width=1.5, alpha=1.0, dash=None):
    c.saveState()
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setStrokeAlpha(alpha)
    if dash:
        c.setDash(*dash)
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path, stroke=1, fill=0)
    c.restoreState()


def arrow_head(c, tip, direction, color, size=11):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    base = np.asarray(tip) - size * direction
    left = base + 0.48 * size * normal
    right = base - 0.48 * size * normal
    p = c.beginPath()
    p.moveTo(*tip)
    p.lineTo(*left)
    p.lineTo(*right)
    p.close()
    c.setFillColor(color)
    c.drawPath(p, stroke=0, fill=1)


def draw_signal_strip(c, x0, x1, y0, values, observed, color, label):
    vals = values / max(np.max(np.abs(values)), 1e-9)
    xs = np.linspace(x0, x1, len(vals))
    ys = y0 + 35 * vals

    c.setStrokeColor(GRID)
    c.setLineWidth(0.8)
    c.line(x0, y0, x1, y0)
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 12)
    c.drawRightString(x0 - 13, y0 - 4, label)

    draw_polyline(c, list(zip(xs, ys)), color, width=1.2, alpha=0.24)
    for idx, (xv, yv) in enumerate(zip(xs, ys)):
        if observed[idx]:
            c.setFillColor(color)
            c.setStrokeColor(BG)
            c.setLineWidth(0.7)
            c.circle(xv, yv, 3.4, stroke=1, fill=1)
        elif idx in {7, 8, 17, 25, 26, 39, 47}:
            c.setStrokeColor(MUTED)
            c.setLineWidth(0.8)
            c.line(xv - 3, yv - 3, xv + 3, yv + 3)
            c.line(xv - 3, yv + 3, xv + 3, yv - 3)


def draw_behavior_plane(c, scores):
    center = np.array([500.0, 280.0])
    basis_1 = np.array([205.0, 58.0])
    basis_2 = np.array([-68.0, 138.0])

    # Soft data-derived plane.
    theta = np.linspace(0, 2 * np.pi, 120)
    boundary = [
        center + 0.96 * np.cos(t) * basis_1 + 0.92 * np.sin(t) * basis_2
        for t in theta
    ]
    c.saveState()
    c.setFillColor(PALE)
    c.setStrokeColor(GRID)
    c.setLineWidth(1.0)
    c.setFillAlpha(0.72)
    plane = c.beginPath()
    plane.moveTo(*boundary[0])
    for point in boundary[1:]:
        plane.lineTo(*point)
    plane.close()
    c.drawPath(plane, stroke=1, fill=1)
    c.restoreState()

    # Thin coordinate family on the plane.
    for level in np.linspace(-0.75, 0.75, 5):
        pts = [center + t * basis_1 + level * basis_2 for t in np.linspace(-0.83, 0.83, 36)]
        draw_polyline(c, pts, GRID, width=0.65, alpha=0.9)
    for level in np.linspace(-0.7, 0.7, 5):
        pts = [center + level * basis_1 + t * basis_2 for t in np.linspace(-0.78, 0.78, 36)]
        draw_polyline(c, pts, GRID, width=0.65, alpha=0.9)

    points = [center + 0.78 * s[0] * basis_1 + 0.73 * s[1] * basis_2 for s in scores]
    draw_polyline(c, points, NAVY, width=2.0, alpha=0.78)
    for idx, point in enumerate(points):
        frac = idx / max(len(points) - 1, 1)
        color = ACCENT if frac < 0.62 else TEAL
        radius = 2.4 if idx % 3 else 3.5
        c.setFillColor(color)
        c.setStrokeColor(BG)
        c.setLineWidth(0.6)
        c.circle(point[0], point[1], radius, stroke=1, fill=1)

    # Closed-loop path converging to an operating point.
    t = np.linspace(0, 4.7 * np.pi, 88)
    decay = np.exp(-0.15 * t)
    loop = [
        center + (0.72 * decay[i] * np.cos(t[i])) * basis_1
        + (0.72 * decay[i] * np.sin(t[i])) * basis_2
        for i in range(len(t))
    ]
    draw_polyline(c, loop, TEAL, width=2.6, alpha=0.92)
    c.setFillColor(NAVY)
    c.setStrokeColor(BG)
    c.setLineWidth(1.4)
    c.circle(center[0], center[1], 6.2, stroke=1, fill=1)

    c.setFillColor(MUTED)
    c.setFont("Helvetica-Oblique", 13)
    c.drawString(640, 178, "finite-length behavior")


def draw_window_stack(c):
    # Translucent stacked windows connect the time records to the projection.
    for idx in range(6):
        x = 250 + idx * 18
        y = 418 - idx * 13
        c.saveState()
        c.setFillColor(PALE)
        c.setStrokeColor(ACCENT)
        c.setFillAlpha(0.19 + idx * 0.035)
        c.setStrokeAlpha(0.28 + idx * 0.05)
        c.setLineWidth(0.9)
        p = c.beginPath()
        p.moveTo(x, y)
        p.lineTo(x + 115, y + 23)
        p.lineTo(x + 115, y - 72)
        p.lineTo(x, y - 94)
        p.close()
        c.drawPath(p, stroke=1, fill=1)
        c.restoreState()

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 11)
    c.drawCentredString(352, 292, "past input-output windows")


def draw_feedback_arc(c):
    c.saveState()
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2.0)
    c.setStrokeAlpha(0.62)
    path = c.beginPath()
    path.moveTo(682, 232)
    path.curveTo(754, 348, 753, 585, 665, 646)
    c.drawPath(path, stroke=1, fill=0)
    c.restoreState()
    arrow_head(c, (665, 646), (-0.65, 0.76), ACCENT, size=12)


def main():
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    k, u, y, scores, missing = simulate_data()
    observed = ~missing

    c = canvas.Canvas(str(PDF_PATH), pagesize=(W, H))
    c.setFillColor(BG)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    # Research-figure heading is intentionally minimal and plot-like.
    c.setFillColor(TEXT)
    c.setFont("Helvetica", 12)
    c.drawString(66, 744, "short and irregular input-output record")
    c.setStrokeColor(GRID)
    c.setLineWidth(0.9)
    c.line(66, 730, 736, 730)

    draw_signal_strip(c, 86, 650, 670, u[0], np.ones_like(observed), NAVY, "u")
    draw_signal_strip(c, 86, 650, 585, y[0], observed, ACCENT, "y1")
    draw_signal_strip(c, 86, 650, 500, y[1], observed, AMBER, "y2")

    # Show the selected short record used below.
    c.saveState()
    c.setFillColor(PALE)
    c.setFillAlpha(0.48)
    c.rect(278, 456, 94, 255, stroke=0, fill=1)
    c.setStrokeColor(ACCENT)
    c.setStrokeAlpha(0.4)
    c.setDash(3, 3)
    c.rect(278, 456, 94, 255, stroke=1, fill=0)
    c.restoreState()

    draw_window_stack(c)
    draw_behavior_plane(c, scores)
    draw_feedback_arc(c)

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 10)
    c.drawRightString(736, 38, "data  ->  representation  ->  output feedback")

    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
