"""Add ShapCCS (B5) to the UCI HAR budget figures of the paper.

Figure 1 (fig_uci_{10,20,30,40,50}mb.png) was made outside this repo and the per-round B1/Hybrid UCI
histories are not here, so the ShapCCS seed-mean curve is drawn on top of each existing PNG.
The axes are calibrated from the image itself: spines give the 0-100 y range and x = 0, and the
light-grey gridlines (piecewise-linear) give the x and y scales. In the 30 MB panel, FedMobile was labelled
"B5"; that digit is replaced with "4" so it matches the paper (B4 = FedMobile, B5 = ShapCCS).

Figure 5 (fig_uci_budget_summary.png) and Figure 3 (fig_uci_budget_bars.png) are rebuilt from the Table 4
values plus ShapCCS.

Writes new files next to the originals (the originals are untouched):
  figures/fig_uci_{b}mb_shapccs.png, figures/fig_uci_budget_summary_shapccs.png,
  figures/fig_uci_budget_bars_shapccs.{png,pdf}
Run: venv/bin/python plot_uci_shapccs_figures.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "Multimodal_Federated_Learning", "figures")
RES = os.path.join(ROOT, "ShapCCS", "torch_port", "results")
SEEDS = [43, 44, 45, 46, 48]
BUDGETS = {10: (32, 1), 20: (72, 2), 30: (112, 5), 40: (152, 5), 50: (194, 5)}  # budget: (rounds, x tick step)
SHAP_COLOR = "black"


def shapccs_curve(rounds):
    accs, ups = [], None
    for s in SEEDS:
        d = np.load(os.path.join(RES, f"ShapCCS_uci_har_mm_K10_C0.2_pf0.3_rem7_R{rounds}_seed{s}.npz"))
        accs.append(np.asarray(d["test_acc"], float))
        u = np.cumsum(np.asarray(d["upload_bytes"], float)) / 1e6
        if ups is not None:
            assert np.allclose(ups, u), "upload differs across seeds"
        ups = u
    acc = np.mean(accs, axis=0)
    if acc.max() <= 1.0:
        acc = acc * 100
    return ups, acc


def _groups(idx):
    out = []
    for x in idx:
        if out and x - out[-1][-1] <= 2:
            out[-1].append(x)
        else:
            out.append([x])
    return [float(np.mean(g)) for g in out]


def calibrate(a, step):
    """Return functions data->pixel (x, y) from spines and gridlines."""
    H, W, _ = a.shape
    dark = a.sum(2) < 150
    cols = np.where(dark.sum(0) > 0.5 * H)[0]
    rows = np.where(dark.sum(1) > 0.5 * W)[0]
    l = cols[cols < W // 4].mean()
    t = rows[rows < H // 4].mean()
    b = rows[rows > H // 2].mean()
    L, R, T, B = int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())
    sub = a[T + 3:B - 3, L + 3:R - 3]
    grey = (sub.min(2) >= 200) & (sub.max(2) <= 245) & ((sub.max(2) - sub.min(2)) < 8)
    vx = [g + L + 3 for g in _groups(np.where(grey.mean(0) > 0.4)[0])]
    hy = [g + T + 3 for g in _groups(np.where(grey.mean(1) > 0.4)[0])]
    # x: knots at 0 (left spine) and k*step (gridlines)
    xk_data = [0.0] + [step * (i + 1) for i in range(len(vx))]
    xk_pix = [l] + vx
    # y: 0 (bottom spine) .. 100 (top spine), gridlines every 5; drop gridline outliers
    lin = lambda v: b + (t - b) * v / 100.0
    yk_data, yk_pix = [0.0], [b]
    for v in range(5, 100, 5):
        cand = [p for p in hy if abs(p - lin(v)) < 0.01 * (b - t)]
        if cand:
            yk_data.append(v)
            yk_pix.append(min(cand, key=lambda p: abs(p - lin(v))))
    yk_data.append(100.0)
    yk_pix.append(t)
    slope = (xk_pix[-1] - xk_pix[0]) / (xk_data[-1] - xk_data[0])

    def fx(x):
        x = np.asarray(x, float)
        inside = np.interp(x, xk_data, xk_pix)
        return np.where(x > xk_data[-1], xk_pix[-1] + (x - xk_data[-1]) * slope, inside)

    fy = lambda y: np.interp(np.asarray(y, float), yk_data, yk_pix)
    info = dict(left=l, top=t, bottom=b, right=R, x_ticks=len(vx), y_ticks=len(yk_pix) - 2, px_per_unit=slope)
    return fx, fy, info


def dejavu(px_digit_height):
    path = font_manager.findfont("DejaVu Sans")
    size = int(round(px_digit_height / 0.729))
    return ImageFont.truetype(path, size)


def replace_digit(im, a, box, digit):
    """Replace a black digit glyph inside box=(x0,y0,x1,y1) by `digit`, restoring the background by blending
    the columns just left and right of the box (keeps lines that pass behind the text)."""
    x0, y0, x1, y1 = box
    left = a[y0:y1 + 1, x0 - 2].astype(float)
    right = a[y0:y1 + 1, x1 + 2].astype(float)
    w = x1 - x0 + 1
    patch = np.zeros((y1 - y0 + 1, w, 3))
    for i in range(w):
        f = (i + 0.5) / w
        patch[:, i] = (1 - f) * left + f * right
    im.paste(Image.fromarray(patch.astype(np.uint8)), (x0, y0))
    d = ImageDraw.Draw(im)
    font = dejavu(y1 - y0 + 1)
    bb = d.textbbox((0, 0), digit, font=font)
    gx = x0 + (w - (bb[2] - bb[0])) / 2 - bb[0]
    gy = y0 - bb[1]
    d.text((gx, gy), digit, fill=(0, 0, 0), font=font)


def overlay_panel(budget):
    rounds, step = BUDGETS[budget]
    src = os.path.join(FIG, f"fig_uci_{budget}mb.png")
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).astype(int)
    fx, fy, info = calibrate(a, step)
    if budget == 30:  # "B5 - FedMobile" / "B5 @ 32.08 MiB" -> B4 (glyph boxes measured from the image)
        replace_digit(im, a, (2258, 549, 2272, 573), "4")
        replace_digit(im, a, (2485, 1557, 2499, 1581), "4")
    H, W = a.shape[:2]
    ups, acc = shapccs_curve(rounds)
    print(f"{budget:>2} MB  calib {info}  ShapCCS final {ups[-1]:.2f} MB {acc[-1]:.2f}%")

    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.asarray(im), interpolation="none")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    s = H / 1000.0  # scale relative to a 1000-px-high image
    px, py = fx(ups), fy(acc)
    ax.plot(px, py, color=SHAP_COLOR, lw=2.6 * s, zorder=5)
    ax.plot(px[-1], py[-1], "o", color=SHAP_COLOR, ms=8 * s, zorder=6)
    fs = 13.5 * s
    # small legend box for the added curve, lower centre of the axes
    lx = fx(0) + 0.42 * (info["right"] - fx(0))
    h = ax.plot([], [], color=SHAP_COLOR, lw=2.6 * s, marker="o", ms=6 * s, label=f"B5 — ShapCCS ({acc[-1]:.2f}% @ {ups[-1]:.2f} MB)")[0]
    ax.legend(handles=[h], loc="lower left", fontsize=fs,
              bbox_to_anchor=(lx / W, 1 - fy(3.5) / H), framealpha=0.9, borderpad=0.4)
    out = os.path.join(FIG, f"fig_uci_{budget}mb_shapccs.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# Figure 5 values: Table 4 (tab:uci_summary); FedMobile and ShapCCS at their actual cumulative uploads.
NOMINAL = [10, 20, 30, 40, 50]
SUMMARY = [
    ("B1 (IC)", NOMINAL, [47.55, 50.13, 69.23, 72.39, 72.83], "#1f77b4", "o"),
    ("B2 (MFedMC)", NOMINAL, [31.09, 34.76, 34.74, 34.87, 54.59], "#ff7f0e", "s"),
    ("B3 (PFMH)", NOMINAL, [36.60, 47.98, 53.26, 54.64, 54.96], "#2ca02c", "^"),
    ("B4 (FedMobile)", [9.62, 19.25, 32.08, 41.70, 51.33], [48.30, 66.24, 66.14, 66.34, 67.88], "#9467bd", "P"),
    ("B5 (ShapCCS)", None, None, SHAP_COLOR, "X"),
    ("Hybrid", NOMINAL, [72.64, 76.34, 77.63, 79.45, 80.88], "#d62728", "D"),
]


def summary_figure():
    ux, uy = [], []
    for b in NOMINAL:
        u, a = shapccs_curve(BUDGETS[b][0])
        ux.append(u[-1])
        uy.append(a[-1])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=250)
    for name, xs, ys, c, m in SUMMARY:
        if name.startswith("B5"):
            xs, ys = ux, uy
        ax.plot(xs, ys, color=c, marker=m, lw=3 if name == "Hybrid" else 2.5, ms=9, label=name,
                zorder=4 if name == "Hybrid" else 3, clip_on=False)
    ax.set_xlim(10, 50)
    ax.set_ylim(0, 100)
    ax.set_xticks(NOMINAL)
    ax.set_yticks(range(0, 101, 5))
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlabel("Cumulative Upload (MB)", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=11, frameon=True)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_uci_budget_summary_shapccs.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("ShapCCS finals:", " / ".join(f"{u:.2f} MB {a:.2f}%" for u, a in zip(ux, uy)))
    return out


def bars_figure():
    """Figure 3: grouped bars in the style of the original fig_uci_budget_bars.png, plus B5 (ShapCCS)."""
    finals = [shapccs_curve(BUDGETS[b][0])[1][-1] for b in NOMINAL]
    vals = {name: ys for name, _, ys, _, _ in SUMMARY if ys is not None}
    vals["B5 (ShapCCS)"] = finals
    series = [  # name, colour, hatch
        ("B1 (IC)", "#1f77b4", "//"), ("B2 (MFedMC)", "#ff7f0e", "--"), ("B3 (PFMH)", "#9467bd", ".."),
        ("B4 (FedMobile)", "#8c564b", "\\\\"), ("B5 (ShapCCS)", "#7f7f7f", "++"), ("Hybrid", "#2ca02c", "xx"),
    ]
    with plt.rc_context({"font.family": "serif", "font.serif": ["DejaVu Serif"], "hatch.linewidth": 0.8}):
        fig, ax = plt.subplots(figsize=(5.2, 4.3), dpi=200)
        n, w = len(series), 0.8 / len(series)
        x = np.arange(len(NOMINAL))
        for i, (name, c, h) in enumerate(series):
            ax.bar(x + (i - (n - 1) / 2) * w, vals[name], w, color=c, edgecolor="black", hatch=h, lw=0.8,
                   label="HybridSketchMFedMC" if name == "Hybrid" else name, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(NOMINAL)
        ax.set_ylim(0, 90)
        ax.set_yticks(range(0, 81, 20))
        ax.yaxis.grid(True, color="#dddddd", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Cumulative upload budget (MB)", fontsize=15)
        ax.set_ylabel("Test accuracy (%)", fontsize=15)
        ax.tick_params(labelsize=14)
        ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=11, frameon=False,
                  handlelength=2.2, columnspacing=1.2)
        fig.tight_layout()
        out = os.path.join(FIG, "fig_uci_budget_bars_shapccs.png")
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out[:-4] + ".pdf", bbox_inches="tight")
        plt.close(fig)
    return out


if __name__ == "__main__":
    for b in BUDGETS:
        print(overlay_panel(b))
    print(summary_figure())
    print(bars_figure())
