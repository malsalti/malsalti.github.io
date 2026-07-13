"""Generate four compact MATLAB-style research plots for the website.

The data are deterministic illustrative simulations created for these
thumbnails; they do not reproduce results from published figures.
"""

from pathlib import Path

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "figures-preview"
SIZE = 800

WHITE = HexColor("#FFFFFF")
BLACK = HexColor("#262626")
GRID = HexColor("#D9D9D9")
LIGHT = HexColor("#F2F2F2")
BLUE = HexColor("#0072BD")
ORANGE = HexColor("#D95319")
YELLOW = HexColor("#EDB120")
PURPLE = HexColor("#7E2F8E")
GREEN = HexColor("#77AC30")
CYAN = HexColor("#4DBEEE")
RED = HexColor("#A2142F")
GRAY = HexColor("#8A8A8A")


def new_canvas(filename):
    OUT.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT / filename), pagesize=(SIZE, SIZE))
    c.setFillColor(WHITE)
    c.rect(0, 0, SIZE, SIZE, stroke=0, fill=1)
    return c


def finish(c):
    c.showPage()
    c.save()


def map_values(values, source, target):
    values = np.asarray(values, dtype=float)
    return target[0] + (values - source[0]) / (source[1] - source[0]) * (target[1] - target[0])


def polyline(c, xs, ys, color, width=2.0, alpha=1.0, dash=None):
    c.saveState()
    c.setStrokeColor(color)
    c.setStrokeAlpha(alpha)
    c.setLineWidth(width)
    if dash:
        c.setDash(*dash)
    path = c.beginPath()
    path.moveTo(float(xs[0]), float(ys[0]))
    for x, y in zip(xs[1:], ys[1:]):
        path.lineTo(float(x), float(y))
    c.drawPath(path, stroke=1, fill=0)
    c.restoreState()


def plot_axes(c, box, xlim, ylim, xlabel, ylabel, xticks, yticks):
    x0, y0, x1, y1 = box
    c.saveState()
    c.setStrokeColor(BLACK)
    c.setLineWidth(1.3)
    c.rect(x0, y0, x1 - x0, y1 - y0, stroke=1, fill=0)

    c.setFont("Helvetica", 24)
    c.setFillColor(BLACK)
    c.setStrokeColor(GRID)
    c.setLineWidth(0.8)
    for value, label in xticks:
        px = map_values(value, xlim, (x0, x1))
        c.line(px, y0, px, y1)
        c.setFillColor(BLACK)
        c.drawCentredString(px, y0 - 34, label)
        c.setFillColor(BLACK)
    for value, label in yticks:
        py = map_values(value, ylim, (y0, y1))
        c.setStrokeColor(GRID)
        c.line(x0, py, x1, py)
        c.setFillColor(BLACK)
        c.drawRightString(x0 - 14, py - 8, label)

    c.setFont("Helvetica", 34)
    c.drawCentredString((x0 + x1) / 2, y0 - 82, xlabel)
    c.saveState()
    c.translate(x0 - 82, (y0 + y1) / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, ylabel)
    c.restoreState()
    c.restoreState()


def data_representations():
    c = new_canvas("data-representations-matlab-v3.pdf")
    xlim = (0, 80)
    input_box = (125, 475, 735, 720)
    output_box = (125, 130, 735, 405)
    plot_axes(c, input_box, xlim, (-1.15, 1.15), "", "u(k)", [], [(-1, "-1"), (0, "0"), (1, "1")])
    plot_axes(c, output_box, xlim, (-1.15, 1.15), "sample k", "y(k)", [(0, "0"), (40, "40"), (80, "80")], [(-1, "-1"), (0, "0"), (1, "1")])

    # A short impulse experiment, inspired by the persistency-of-excitation
    # result, followed by irregular output measurements and data completion.
    k = np.arange(81)
    u = np.zeros_like(k, dtype=float)
    impulse_index = np.array([4, 15, 26, 37, 48, 59, 70])
    u[impulse_index] = np.array([0.95, -0.78, 0.88, -0.94, 0.72, -0.86, 0.80])

    input_zero = map_values(0, (-1.15, 1.15), (input_box[1], input_box[3]))
    c.saveState()
    c.setStrokeColor(BLUE)
    c.setFillColor(WHITE)
    c.setLineWidth(3.4)
    for idx in impulse_index:
        px = map_values(idx, xlim, (input_box[0], input_box[2]))
        py = map_values(u[idx], (-1.15, 1.15), (input_box[1], input_box[3]))
        c.line(px, input_zero, px, py)
        c.circle(px, py, 6.2, stroke=1, fill=1)
    c.restoreState()

    y = np.zeros_like(k, dtype=float)
    for idx in range(2, k.size):
        y[idx] = 1.46 * y[idx - 1] - 0.64 * y[idx - 2] + 0.48 * u[idx - 1] - 0.20 * u[idx - 2]
    y *= 0.92 / np.max(np.abs(y))

    output_px = map_values(k, xlim, (output_box[0], output_box[2]))
    output_py = map_values(y, (-1.15, 1.15), (output_box[1], output_box[3]))
    polyline(c, output_px, output_py, ORANGE, width=4.4)

    rng = np.random.default_rng(13)
    measured = np.array([0, 2, 5, 9, 13, 18, 19, 25, 31, 32, 39, 45, 51, 58, 59, 66, 72, 77, 80])
    noisy = y[measured] + rng.normal(scale=0.035, size=measured.size)
    c.saveState()
    c.setStrokeColor(BLUE)
    c.setFillColor(WHITE)
    c.setLineWidth(2.7)
    for idx, value in zip(measured, noisy):
        px = map_values(idx, xlim, (output_box[0], output_box[2]))
        py = map_values(value, (-1.15, 1.15), (output_box[1], output_box[3]))
        c.circle(px, py, 6.3, stroke=1, fill=1)
    c.restoreState()
    finish(c)


def nonlinear_control():
    c = new_canvas("nonlinear-control-matlab-v3.pdf")
    box = (125, 125, 735, 725)
    xlim = (-2.1, 2.1)
    ylim = (-2.0, 2.0)
    plot_axes(c, box, xlim, ylim, "x1", "x2", [(-2, "-2"), (0, "0"), (2, "2")], [(-2, "-2"), (0, "0"), (2, "2")])

    # A data-certified region-of-attraction estimate.  The non-elliptic
    # outer contour suggests a state-dependent nonlinear controller, while
    # the dashed ellipse provides a compact baseline comparison.
    theta = np.linspace(0, 2 * np.pi, 361)
    modulation = 1 + 0.075 * np.cos(3 * theta) - 0.045 * np.sin(2 * theta)
    outer_x = 1.72 * modulation * np.cos(theta)
    outer_y = 1.36 * modulation * np.sin(theta)
    outer_px = map_values(outer_x, xlim, (box[0], box[2]))
    outer_py = map_values(outer_y, ylim, (box[1], box[3]))

    path = c.beginPath()
    path.moveTo(float(outer_px[0]), float(outer_py[0]))
    for px, py in zip(outer_px[1:], outer_py[1:]):
        path.lineTo(float(px), float(py))
    path.close()
    c.saveState()
    c.setFillColor(HexColor("#DCEEF8"))
    c.setFillAlpha(0.88)
    c.setStrokeColor(BLUE)
    c.setLineWidth(4.0)
    c.drawPath(path, stroke=1, fill=1)
    c.restoreState()

    # Nested data-dependent Lyapunov sublevel sets.
    for scale, color, width, alpha in ((0.72, CYAN, 2.6, 0.90), (0.43, BLUE, 2.4, 0.75)):
        px = map_values(scale * outer_x, xlim, (box[0], box[2]))
        py = map_values(scale * outer_y, ylim, (box[1], box[3]))
        polyline(c, px, py, color, width=width, alpha=alpha)

    # Sparse numerical initial-condition tests keep the plot recognizably
    # MATLAB-like without overwhelming the small web thumbnail.
    rng = np.random.default_rng(23)
    for _ in range(180):
        angle = rng.uniform(0, 2 * np.pi)
        radial = np.sqrt(rng.uniform(0.03, 0.94))
        mod = 1 + 0.075 * np.cos(3 * angle) - 0.045 * np.sin(2 * angle)
        x = 1.72 * radial * mod * np.cos(angle)
        y = 1.36 * radial * mod * np.sin(angle)
        px = map_values(x, xlim, (box[0], box[2]))
        py = map_values(y, ylim, (box[1], box[3]))
        c.setFillColor(BLUE)
        c.setFillAlpha(0.28)
        c.circle(px, py, 1.8, stroke=0, fill=1)

    baseline_x = 1.22 * np.cos(theta)
    baseline_y = 0.93 * np.sin(theta)
    px = map_values(baseline_x, xlim, (box[0], box[2]))
    py = map_values(baseline_y, ylim, (box[1], box[3]))
    polyline(c, px, py, ORANGE, width=3.0, alpha=0.88, dash=(10, 7))
    ox = map_values(0, xlim, (box[0], box[2]))
    oy = map_values(0, ylim, (box[1], box[3]))
    c.setFillColor(BLACK)
    c.circle(ox, oy, 8.0, stroke=0, fill=1)
    finish(c)


def predictive_learning():
    c = new_canvas("predictive-learning-matlab-v2.pdf")
    box = (125, 125, 735, 725)
    xlim = (0, 20)
    ylim = (-1.35, 1.35)
    plot_axes(c, box, xlim, ylim, "prediction step", "output", [(0, "0"), (10, "10"), (20, "20")], [(-1, "-1"), (0, "0"), (1, "1")])

    # Input/output constraint bounds.
    for bound in (-1.1, 1.1):
        py = map_values(bound, ylim, (box[1], box[3]))
        polyline(c, [box[0], box[2]], [py, py], RED, width=2.3, alpha=0.72, dash=(9, 7))

    k = np.arange(21)
    start = 1.0
    for idx, decay in enumerate(np.linspace(0.065, 0.24, 15)):
        candidate = start * np.exp(-decay * k) + 0.16 * np.sin((0.28 + 0.01 * idx) * k) * np.exp(-0.08 * k)
        px = map_values(k, xlim, (box[0], box[2]))
        py = map_values(candidate, ylim, (box[1], box[3]))
        polyline(c, px, py, GRAY, width=1.7, alpha=0.30)

    selected = start * np.exp(-0.19 * k) + 0.06 * np.sin(0.42 * k) * np.exp(-0.13 * k)
    px = map_values(k, xlim, (box[0], box[2]))
    py = map_values(selected, ylim, (box[1], box[3]))
    polyline(c, px, py, BLUE, width=5.0, alpha=1.0)
    for idx in range(0, len(k), 2):
        c.setFillColor(WHITE)
        c.setStrokeColor(BLUE)
        c.setLineWidth(2.2)
        c.circle(px[idx], py[idx], 6.0, stroke=1, fill=1)

    zero_y = map_values(0, ylim, (box[1], box[3]))
    polyline(c, [box[0], box[2]], [zero_y, zero_y], BLACK, width=1.8, alpha=0.55, dash=(5, 5))
    finish(c)


def biomedical_control():
    c = new_canvas("biomedical-control-matlab-v2.pdf")
    box = (125, 125, 735, 725)
    xlim = (0, 120)
    ylim = (40, 86)
    plot_axes(c, box, xlim, ylim, "time [min]", "MAP [mmHg]", [(0, "0"), (60, "60"), (120, "120")], [(40, "40"), (60, "60"), (80, "80")])

    target = 75.0
    band_bottom = map_values(72, ylim, (box[1], box[3]))
    band_top = map_values(78, ylim, (box[1], box[3]))
    c.saveState()
    c.setFillColor(LIGHT)
    c.rect(box[0], band_bottom, box[2] - box[0], band_top - band_bottom, stroke=0, fill=1)
    c.restoreState()

    rng = np.random.default_rng(41)
    time = np.linspace(0, 120, 121)
    cohort = []
    colors = [BLUE, ORANGE, YELLOW, PURPLE, GREEN, CYAN, RED]
    for idx in range(26):
        initial = rng.uniform(43, 63)
        tau = rng.uniform(20, 43)
        oscillation = rng.uniform(0.8, 3.0) * np.exp(-time / rng.uniform(45, 78)) * np.sin(time / rng.uniform(8, 14) + rng.uniform(0, 1.8))
        response = target - (target - initial) * np.exp(-time / tau) + oscillation
        response += rng.normal(scale=0.18, size=time.size)
        cohort.append(response)
        px = map_values(time, xlim, (box[0], box[2]))
        py = map_values(response, ylim, (box[1], box[3]))
        polyline(c, px, py, colors[idx % len(colors)], width=1.6, alpha=0.22)

    median = np.median(np.asarray(cohort), axis=0)
    px = map_values(time, xlim, (box[0], box[2]))
    py = map_values(median, ylim, (box[1], box[3]))
    polyline(c, px, py, BLUE, width=5.0, alpha=1.0)

    target_y = map_values(target, ylim, (box[1], box[3]))
    polyline(c, [box[0], box[2]], [target_y, target_y], BLACK, width=2.2, alpha=0.65, dash=(8, 6))
    c.setFillColor(BLACK)
    c.setFont("Helvetica", 28)
    c.drawRightString(720, target_y + 15, "target")
    finish(c)


def main():
    data_representations()
    nonlinear_control()
    predictive_learning()
    biomedical_control()


if __name__ == "__main__":
    main()
