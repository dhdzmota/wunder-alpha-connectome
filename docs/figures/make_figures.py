"""Figures of the README / write-up -> docs/img/<name>-{light,dark}.svg.

Diagrams are drawn as hand-laid SVG from the same theme tokens as the charts. Charts read the real artifacts
($WNN_ARTIFACTS) and the numbers logged in docs/EXPERIMENTS.md. Usage:
    python docs/figures/make_figures.py              # everything
    python docs/figures/make_figures.py --diagrams   # diagrams only (no data needed)
"""
import argparse
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "img"
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------------------------------------ theme tokens
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  border="#d8d7d0", blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", violet="#6250d6",
                  tint=0.10, gray_fill="#f0efec"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 border="#3a3a37", blue="#3987e5", orange="#d95926", aqua="#199e70", violet="#9085e9",
                 tint=0.20, gray_fill="#262624"),
}
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def mix(c, surface, a):
    """c blended onto surface with opacity a (box tints)."""
    f = lambda h: [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x * a + s * (1 - a)):02x}" for x, s in zip(f(c), f(surface)))


# ------------------------------------------------------------------------------------------------ SVG diagrams
class Svg:
    def __init__(self, w, h, t):
        self.w, self.h, self.t, self.parts = w, h, t, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=14, color="ink", weight=400, anchor="middle", italic=False):
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{self.t[color]}" '
                 f'text-anchor="{anchor}"{st}>{escape(s)}</text>')

    def box(self, x, y, w, h, title, lines=(), hue=None, dashed=False, title_size=15):
        t = self.t
        fill = mix(t[hue], t["surface"], t["tint"]) if hue else t["gray_fill"]
        stroke = t[hue] if hue else t["border"]
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="1.5"{dash}/>')
        n = len(lines)
        y0 = y + h / 2 - (n * 17) / 2 + (5 if n else 5)
        self.text(x + w / 2, y0, title, size=title_size, weight=600)
        for i, ln in enumerate(lines):
            self.text(x + w / 2, y0 + 19 + i * 17, ln, size=12.5, color="ink2")

    def arrow(self, x1, y1, x2, y2, label=None, color="muted", dx=0, dy=-8, anchor="middle", curve=None):
        c = self.t[color]
        if curve:
            d = f"M{x1},{y1} {curve} {x2},{y2}"
        else:
            d = f"M{x1},{y1} L{x2},{y2}"
        self.add(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.8" marker-end="url(#ah-{color})"/>')
        if label:
            self.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, label, size=12, color="ink2", anchor=anchor)

    def render(self):
        t = self.t
        marks = "".join(f'<marker id="ah-{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t[c]}"/></marker>'
                        for c in ("muted", "blue", "orange", "aqua", "violet", "ink2"))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" font-family="{FONT}"><defs>{marks}</defs>'
                f'<rect width="{self.w}" height="{self.h}" rx="12" fill="{t["surface"]}"/>' + "".join(self.parts) + "</svg>")


def diagram_problem(t):
    """One sequence: warm-up, required rows, the hidden scored subset; one row in, one prediction out."""
    s = Svg(960, 400, t)
    s.text(40, 42, "One sequence = 20,000 rows of a live order book, seen one row at a time", 18, weight=600, anchor="start")
    x0, x1, y = 40, 920, 78
    wx = x0 + (x1 - x0) * 99 / 20000 * 12  # warm-up drawn wider than scale so it is visible
    s.add(f'<rect x="{x0}" y="{y}" width="{wx - x0}" height="34" rx="6" fill="{t["gray_fill"]}" stroke="{t["border"]}"/>')
    s.add(f'<rect x="{wx + 3}" y="{y}" width="{x1 - wx - 3}" height="34" rx="6" '
          f'fill="{mix(t["blue"], t["surface"], t["tint"])}" stroke="{t["blue"]}"/>')
    import random
    rnd = random.Random(7)
    for _ in range(150):  # hidden scored rows
        xx = rnd.uniform(wx + 8, x1 - 6)
        s.add(f'<line x1="{xx:.1f}" y1="{y + 6}" x2="{xx:.1f}" y2="{y + 28}" stroke="{t["orange"]}" stroke-width="1.6"/>')
    s.text(x0, y + 56, "rows 0–98: warm-up", 12.5, "ink2", anchor="start")
    s.text(x0, y + 73, "(no prediction)", 12.5, "muted", anchor="start")
    s.text((wx + x1) / 2 + 40, y + 56, "rows 99–19,999: a prediction is required on every row", 12.5, "ink2")
    s.add(f'<line x1="{x1 - 250}" y1="{y + 70}" x2="{x1 - 236}" y2="{y + 70}" stroke="{t["orange"]}" stroke-width="2.5"/>')
    s.text(x1 - 230, y + 74, "hidden subset that is actually scored", 12.5, "ink2", anchor="start")
    # one step
    yb = 210
    s.box(40, yb, 250, 120, "Row t (112 numbers)", ["prices & volumes of the book", "for instruments i0, i1", "+ 8 anonymous extras"])
    s.box(375, yb, 230, 120, "Model with memory", ["reads row t, updates", "its internal state,", "never sees the future"], hue="blue")
    s.box(690, yb, 230, 120, "Prediction for row t", ["two numbers: t0, t1", "(future price moves", "of instrument i0)"], hue="aqua")
    s.arrow(290, yb + 60, 373, yb + 60)
    s.arrow(605, yb + 60, 688, yb + 60)
    s.add(f'<path d="M455,{yb + 120} C455,{yb + 150} 525,{yb + 150} 525,{yb + 122}" fill="none" stroke="{t["blue"]}" '
          f'stroke-width="1.8" marker-end="url(#ah-blue)"/>')
    s.text(490, yb + 160, "state carried to row t+1", 12, "ink2")
    s.text(805, yb + 152, "Budget: 1 CPU core, 60 minutes", 12.5, "ink2")
    s.text(805, yb + 168, "for the whole test set", 12.5, "ink2")
    return s.render()


def diagram_inference(t):
    """Submission at inference time: regime detection, the route's fused blend, optional row stack."""
    s = Svg(960, 540, t)
    s.text(40, 40, "What runs on the scoring server, for every row", 18, weight=600, anchor="start")
    s.box(40, 70, 220, 105, "Row t", ["112 features", "+ change since the previous row"])
    s.box(330, 70, 300, 105, "Regime detector", ["reads warm-up rows 0–98 and guesses", "the tick size of this sequence's", "target (4 possible grids)"], hue="violet")
    s.box(700, 70, 220, 105, "Budget", ["1 CPU core, 60 minutes", "for the whole test set;", "final run: 51 minutes"])
    s.arrow(260, 122, 328, 122)
    # the route
    s.add(f'<rect x="40" y="225" width="880" height="150" rx="12" fill="none" stroke="{t["border"]}" stroke-width="1.5" stroke-dasharray="6 4"/>')
    s.text(60, 250, "Route r: one fused numba call per row", 13.5, weight=600, anchor="start")
    s.text(900, 250, "blend p = weighted average of the three", 12.5, "ink2", anchor="end")
    for i, (name, lines) in enumerate((("minGRU specialist", ["recurrent, trained on regime r only", "weight 0.50"]),
                                       ("SSM", ["state-space model, shared", "weight ≈ 0.25"]),
                                       ("TCN", ["convolution over the last 1,024 rows", "weight ≈ 0.25"]))):
        s.box(60 + i * 290, 268, 260, 90, name, lines, hue="blue", title_size=14)
    s.arrow(150, 175, 150, 223)
    s.arrow(480, 175, 480, 223, "picks route r (once per sequence)", color="violet", dx=12, dy=4, anchor="start")
    # output
    s.box(40, 420, 470, 95, "Row stack (routes 0 and 1 only)", ["LightGBM on (x, Δx, p):  ŷ = 0.65 · p + 0.35 · trees", "routes 2 and 3:  ŷ = p"], hue="orange")
    s.box(590, 420, 330, 95, "Prediction for row t", ["t0, t1; the state is", "carried to the next row"], hue="aqua")
    s.arrow(275, 375, 275, 418, "p", dx=10, dy=4, anchor="start")
    s.arrow(510, 467, 588, 467)
    return s.render()


def diagram_training(t):
    """Training pipeline: teacher with future inputs -> soft labels -> students -> valid fine-tune -> stack -> package."""
    s = Svg(960, 520, t)
    s.text(40, 40, "How the models were trained", 18, weight=600, anchor="start")
    s.box(40, 80, 210, 110, "Train set", ["10,607 sequences", "(older period,", "no scoring mask)"])
    s.box(330, 70, 270, 130, "Teacher (training only)", ["a GRU that may look ahead:", "it gets the average of the", "next 2–50 rows as inputs"], hue="orange",
          dashed=True)
    s.box(680, 80, 240, 110, "Soft labels", ["0.2 · real target", "+ 0.8 · teacher prediction", "(denoised target)"], hue="orange")
    s.arrow(250, 135, 328, 135)
    s.arrow(600, 135, 678, 135)
    s.box(40, 250, 560, 110, "Students: the models that ship (see only the past)", [
        "4 regime specialists (minGRU + tick-class head)  ·  SSM  ·  TCN",
        "loss = weighted MSE + 3 · (1 − weighted correlation)"], hue="blue")
    s.arrow(800, 190, 600, 290, "trained on", dx=40, dy=-4, anchor="start")
    s.arrow(145, 190, 145, 248)
    s.box(680, 250, 240, 110, "Validation set", ["1,873 sequences, newer,", "with the real scoring mask"])
    s.box(40, 405, 360, 95, "Fine-tune on validation", ["3 epochs · scored rows weighted ×5", "two halves (out-of-fold) + full fit"], hue="blue")
    s.box(470, 405, 210, 95, "Row stack", ["LightGBM on out-", "of-fold predictions"], hue="orange")
    s.box(745, 405, 175, 95, "Package", ["numba · int16", "weights · 12 MB"], hue="aqua")
    s.arrow(160, 360, 160, 403)
    s.add(f'<path d="M800,360 L800,382 L330,382 L330,403" fill="none" stroke="{t["muted"]}" stroke-width="1.8" '
          f'marker-end="url(#ah-muted)"/>')
    s.text(565, 376, "same validation data", 12, "ink2")
    s.arrow(400, 452, 468, 452)
    s.arrow(680, 452, 743, 452)
    return s.render()


def diagram_crossfit(t):
    """Cross-fit protocol on validation: two halves, out-of-fold predictions, the two-halves gate."""
    s = Svg(960, 400, t)
    s.text(40, 40, "Deciding what ships: out-of-fold scores on two halves of validation", 18, weight=600, anchor="start")
    hb, ho = t["blue"], t["orange"]
    rows = [("Half A", "fit", hb), ("Half B", "fit", ho)]
    y = 80
    for i, (name, _, c) in enumerate(rows):
        yy = y + i * 70
        s.text(40, yy + 30, f"Model {i + 1}", 14, weight=600, anchor="start")
        fill_a = mix(hb, t["surface"], t["tint"] * 2.2) if i == 0 else t["gray_fill"]
        fill_b = mix(ho, t["surface"], t["tint"] * 2.2) if i == 1 else t["gray_fill"]
        s.add(f'<rect x="140" y="{yy + 8}" width="300" height="34" rx="6" fill="{fill_a}" stroke="{hb}"/>')
        s.add(f'<rect x="446" y="{yy + 8}" width="300" height="34" rx="6" fill="{fill_b}" stroke="{ho}"/>')
        s.text(290, yy + 30, "fine-tune here" if i == 0 else "predict here", 13, "ink")
        s.text(596, yy + 30, "predict here" if i == 0 else "fine-tune here", 13, "ink")
    s.text(290, y - 6, "validation half A", 12.5, "ink2")
    s.text(596, y - 6, "validation half B", 12.5, "ink2")
    s.box(140, 240, 606, 60, "Every validation row is predicted by a model that never trained on it",
          [], hue="aqua", title_size=14)
    s.arrow(443, 222, 443, 238)
    s.box(780, 80, 140, 220, "Gate", ["a change ships", "only if it", "improves the", "score on", "both halves"], hue="violet")
    s.text(443, 340, "The shipped model is fitted on all of validation; the two half-models only measure.", 13, "ink2")
    s.text(443, 362, "Public leaderboard gains were 35–60 % of these out-of-fold gains.", 13, "ink2")
    return s.render()


# ------------------------------------------------------------------------------------------------ icon explainer
def _tint(t, hue, a=None):
    return mix(t[hue], t["surface"], t["tint"] if a is None else a)


def ic_book(s, cx, cy):
    """Order book: sell orders (orange) above the price, buy orders (blue) below."""
    t = s.t
    for i, w in enumerate((26, 40, 58, 34)):
        s.add(f'<rect x="{cx - 46}" y="{cy - 50 + i * 11}" width="{w}" height="8" rx="2" fill="{t["orange"]}"/>')
    s.add(f'<line x1="{cx - 52}" y1="{cy - 3}" x2="{cx + 30}" y2="{cy - 3}" stroke="{t["ink2"]}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    for i, w in enumerate((48, 62, 30, 22)):
        s.add(f'<rect x="{cx - 46}" y="{cy + 5 + i * 11}" width="{w}" height="8" rx="2" fill="{t["blue"]}"/>')
    s.text(cx + 36, cy - 26, "sell", 11.5, "ink2", anchor="start")
    s.text(cx + 36, cy + 2, "price", 11.5, "ink2", anchor="start")
    s.text(cx + 36, cy + 30, "buy", 11.5, "ink2", anchor="start")


def ic_detect(s, cx, cy):
    """A ruler of tick marks under a magnifier, and 4 options with one chosen."""
    t = s.t
    s.add(f'<line x1="{cx - 60}" y1="{cy + 8}" x2="{cx + 30}" y2="{cy + 8}" stroke="{t["ink2"]}" stroke-width="2"/>')
    for k in range(7):
        x = cx - 60 + k * 15
        s.add(f'<line x1="{x}" y1="{cy + 8}" x2="{x}" y2="{cy - 6}" stroke="{t["ink2"]}" stroke-width="2"/>')
    s.add(f'<circle cx="{cx + 8}" cy="{cy - 4}" r="24" fill="{_tint(t, "violet", 0.25)}" fill-opacity="0.6" stroke="{t["violet"]}" stroke-width="4"/>')
    s.add(f'<line x1="{cx + 25}" y1="{cy + 13}" x2="{cx + 44}" y2="{cy + 32}" stroke="{t["violet"]}" stroke-width="7" stroke-linecap="round"/>')
    for k in range(4):
        x = cx - 42 + k * 22
        fill = t["violet"] if k == 2 else "none"
        s.add(f'<circle cx="{x}" cy="{cy + 50}" r="7" fill="{fill}" stroke="{t["violet"]}" stroke-width="2"/>')


def ic_memory(s, cx, cy):
    """Three small networks: a loop (recurrent), a wave (state-space), a window (convolution)."""
    t = s.t
    c = t["blue"]
    for i, lab in enumerate(("minGRU", "SSM", "TCN")):
        x = cx - 87 + i * 60
        s.add(f'<rect x="{x}" y="{cy - 40}" width="54" height="54" rx="10" fill="{_tint(t, "blue")}" stroke="{c}" stroke-width="1.8"/>')
        mx, my = x + 27, cy - 13
        if i == 0:
            s.add(f'<path d="M{mx + 13},{my - 5} A14,14 0 1,0 {mx + 11},{my + 9}" fill="none" stroke="{c}" stroke-width="3"/>')
            s.add(f'<path d="M{mx + 7},{my - 9} L{mx + 17},{my - 11} L{mx + 15},{my - 1} Z" fill="{c}"/>')
        elif i == 1:
            s.add(f'<path d="M{mx - 18},{my} C{mx - 12},{my - 18} {mx - 4},{my - 18} {mx},{my} S{mx + 12},{my + 18} {mx + 18},{my}" '
                  f'fill="none" stroke="{c}" stroke-width="3"/>')
        else:
            for k in range(5):
                h = (8, 14, 10, 18, 12)[k]
                op = 1 if k >= 2 else 0.35
                s.add(f'<rect x="{mx - 19 + k * 8}" y="{my + 10 - h}" width="6" height="{h}" fill="{c}" opacity="{op}"/>')
            s.add(f'<rect x="{mx - 4}" y="{my - 14}" width="26" height="28" rx="3" fill="none" stroke="{t["ink2"]}" stroke-width="1.5" stroke-dasharray="3 2"/>')
        s.text(mx, cy + 30, lab, 11.5, "ink2")
    s.text(cx, cy + 52, "→ average of the three", 12, "ink2")


def ic_tree(s, cx, cy):
    """A small decision tree."""
    t = s.t
    c = t["orange"]
    root, mid, leaf = (cx, cy - 38), [(cx - 32, cy), (cx + 32, cy)], [(cx - 48, cy + 38), (cx - 16, cy + 38), (cx + 16, cy + 38), (cx + 48, cy + 38)]
    for a, b in [(root, mid[0]), (root, mid[1]), (mid[0], leaf[0]), (mid[0], leaf[1]), (mid[1], leaf[2]), (mid[1], leaf[3])]:
        s.add(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{c}" stroke-width="2.5"/>')
    for (x, y), r in [(root, 11)] + [(m, 9) for m in mid] + [(l, 7) for l in leaf]:
        s.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{_tint(t, "orange", 0.35)}" stroke="{c}" stroke-width="2.5"/>')


def ic_target(s, cx, cy):
    """A target with an up and a down arrow: direction and size of the move."""
    t = s.t
    c = t["aqua"]
    for r, w in ((40, 3), (26, 3), (12, 0)):
        fill = c if r == 12 else _tint(t, "aqua", 0.12)
        s.add(f'<circle cx="{cx - 14}" cy="{cy}" r="{r}" fill="{fill}" stroke="{c}" stroke-width="{w}"/>')
    s.add(f'<line x1="{cx + 46}" y1="{cy - 4}" x2="{cx + 46}" y2="{cy - 38}" stroke="{c}" stroke-width="4" marker-end="url(#ah-aqua)"/>')
    s.add(f'<line x1="{cx + 46}" y1="{cy + 4}" x2="{cx + 46}" y2="{cy + 38}" stroke="{t["muted"]}" stroke-width="4" marker-end="url(#ah-muted)"/>')


def ic_person(s, x, y, hue, scale=1.0):
    t = s.t
    r = 10 * scale
    s.add(f'<circle cx="{x}" cy="{y - 16 * scale}" r="{r}" fill="{_tint(t, hue, 0.35)}" stroke="{t[hue]}" stroke-width="2.2"/>')
    s.add(f'<path d="M{x - 18 * scale},{y + 18 * scale} C{x - 18 * scale},{y - 2 * scale} {x + 18 * scale},{y - 2 * scale} '
          f'{x + 18 * scale},{y + 18 * scale} Z" fill="{_tint(t, hue, 0.35)}" stroke="{t[hue]}" stroke-width="2.2"/>')


def ic_teacher(s, cx, cy):
    """A person looking at future rows (dashed = not available at prediction time)."""
    t = s.t
    ic_person(s, cx - 34, cy, "orange", 1.2)
    s.add(f'<path d="M{cx - 14},{cy - 22} L{cx + 2},{cy - 30} L{cx + 2},{cy - 12} Z" fill="{t["orange"]}"/>')
    for k in range(4):
        s.add(f'<rect x="{cx + 8 + k * 11}" y="{cy - 36}" width="8" height="34" rx="2" fill="none" stroke="{t["orange"]}" '
              f'stroke-width="1.6" stroke-dasharray="3 2"/>')
    s.text(cx + 29, cy + 14, "future", 11, "ink2")


def ic_clean(s, cx, cy):
    """Noisy steps (raw target) vs a clean curve (teacher's answer)."""
    t = s.t
    s.add(f'<rect x="{cx - 50}" y="{cy - 38}" width="100" height="68" rx="8" fill="{t["surface"]}" stroke="{t["border"]}" stroke-width="1.5"/>')
    s.add(f'<path d="M{cx - 42},{cy + 8} h14 v-26 h10 v34 h12 v-20 h14 v-12 h8 v28 h16" fill="none" stroke="{t["muted"]}" stroke-width="2"/>')
    s.add(f'<path d="M{cx - 42},{cy + 4} C{cx - 24},{cy - 22} {cx - 6},{cy + 14} {cx + 10},{cy - 4} S{cx + 34},{cy - 14} {cx + 42},{cy + 4}" '
          f'fill="none" stroke="{t["orange"]}" stroke-width="3"/>')


def ic_students(s, cx, cy):
    for dx in (-30, 0, 30):
        ic_person(s, cx + dx, cy + (0 if dx == 0 else 6), "blue", 0.9 if dx else 1.05)


def ic_calendar(s, cx, cy):
    t = s.t
    c = t["aqua"]
    s.add(f'<rect x="{cx - 34}" y="{cy - 30}" width="68" height="60" rx="8" fill="{_tint(t, "aqua", 0.15)}" stroke="{c}" stroke-width="2.2"/>')
    s.add(f'<rect x="{cx - 34}" y="{cy - 30}" width="68" height="16" rx="6" fill="{c}"/>')
    for k in range(8):
        x, y = cx - 22 + (k % 4) * 15, cy - 2 + (k // 4) * 16
        fill = c if k == 7 else t["muted"]
        s.add(f'<circle cx="{x}" cy="{y}" r="{5 if k == 7 else 3}" fill="{fill}"/>')


def ic_stopwatch(s, cx, cy):
    t = s.t
    c = t["violet"]
    s.add(f'<rect x="{cx - 7}" y="{cy - 46}" width="14" height="9" rx="2" fill="{c}"/>')
    s.add(f'<circle cx="{cx}" cy="{cy}" r="34" fill="{_tint(t, "violet", 0.15)}" stroke="{c}" stroke-width="4"/>')
    s.add(f'<path d="M{cx},{cy} L{cx},{cy - 34} A34,34 0 1,1 {cx - 31},{cy - 14} Z" fill="{_tint(t, "violet", 0.45)}"/>')
    s.add(f'<line x1="{cx}" y1="{cy}" x2="{cx - 24}" y2="{cy - 10}" stroke="{t["ink"]}" stroke-width="3" stroke-linecap="round"/>')
    s.add(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{t["ink"]}"/>')


def ic_chip(s, cx, cy):
    t = s.t
    for k in range(4):
        o = -18 + k * 12
        for x1, y1, x2, y2 in ((cx + o, cy - 34, cx + o, cy - 26), (cx + o, cy + 26, cx + o, cy + 34),
                               (cx - 34, cy + o, cx - 26, cy + o), (cx + 26, cy + o, cx + 34, cy + o)):
            s.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t["ink2"]}" stroke-width="2.5"/>')
    s.add(f'<rect x="{cx - 26}" y="{cy - 26}" width="52" height="52" rx="6" fill="{t["gray_fill"]}" stroke="{t["ink2"]}" stroke-width="2.2"/>')
    s.text(cx, cy + 6, "1 CPU", 12, "ink", weight=600)


def diagram_howitworks(t):
    """Icon explainer: the five steps of one prediction, how the models learned, why it is fast."""
    s = Svg(1200, 760, t)
    s.text(36, 44, "How the model works, in five steps", 22, weight=700, anchor="start")
    s.text(36, 70, "Every row of market data goes through the same five steps, one row at a time, 20,000 rows per sequence.",
           14, "ink2", anchor="start")
    cards = [
        ("border", "1  New data arrives", ic_book, ["The market sends one row:", "prices and volumes of buy", "and sell orders for two", "instruments (112 numbers)."]),
        ("violet", "2  Which market?", ic_detect, ["From the first 99 rows it", "finds the step size of the", "price (one of 4 kinds) and", "picks the matching expert."]),
        ("blue", "3  Use its memory", ic_memory, ["Three neural networks read", "the row and update their", "memory of everything seen", "so far in this sequence."]),
        ("orange", "4  Final correction", ic_tree, ["Small decision trees fine-", "tune the average with the", "raw row (only in the 2", "fine-step kinds of market)."]),
        ("aqua", "5  Prediction", ic_target, ["Two numbers: in which", "direction and how far the", "price will move. Then the", "next row arrives."]),
    ]
    w, gap, x0, y0, h = 210, 21, 36, 96, 330
    for i, (hue, title, icon, lines) in enumerate(cards):
        x = x0 + i * (w + gap)
        stroke = t[hue] if hue != "border" else t["border"]
        fill = _tint(t, hue) if hue != "border" else t["gray_fill"]
        s.add(f'<rect x="{x}" y="{y0}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        s.add(f'<rect x="{x + 14}" y="{y0 + 16}" width="{w - 28}" height="140" rx="10" fill="{t["surface"]}"/>')
        icon(s, x + w / 2, y0 + 86)
        num, name = title.split("  ")
        badge = t[hue] if hue != "border" else t["ink2"]
        s.add(f'<circle cx="{x + 30}" cy="{y0 + 186}" r="13" fill="{badge}"/>')
        s.add(f'<text x="{x + 30}" y="{y0 + 191}" font-size="14" font-weight="700" fill="{t["surface"]}" text-anchor="middle">{num}</text>')
        s.text(x + 50, y0 + 191, name, 15.5, weight=700, anchor="start")
        for k, ln in enumerate(lines):
            s.text(x + 16, y0 + 226 + k * 21, ln, 13, "ink2", anchor="start")
        if i < 4:
            ax = x + w + 3
            s.add(f'<path d="M{ax},{y0 + 76} l12,10 l-12,10" fill="none" stroke="{t["muted"]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
    # loop back: 5 -> 1
    c1, c5, yb = x0 + w / 2, x0 + 4 * (w + gap) + w / 2, y0 + h
    s.add(f'<path d="M{c5},{yb + 2} L{c5},{yb + 20} Q{c5},{yb + 30} {c5 - 10},{yb + 30} L{c1 + 10},{yb + 30} Q{c1},{yb + 30} {c1},{yb + 20} '
          f'L{c1},{yb + 6}" fill="none" stroke="{t["muted"]}" stroke-width="2.2" stroke-dasharray="7 5" marker-end="url(#ah-muted)"/>')
    s.add(f'<rect x="{(c1 + c5) / 2 - 205}" y="{yb + 19}" width="410" height="22" fill="{t["surface"]}"/>')
    s.text((c1 + c5) / 2, yb + 35, "repeat for the next row: the networks keep their memory", 13.5, "ink2", italic=True)
    # bottom panels
    py, ph = 490, 245
    s.add(f'<rect x="36" y="{py}" width="760" height="{ph}" rx="14" fill="none" stroke="{t["border"]}" stroke-width="1.5"/>')
    s.text(58, py + 32, "How the networks learned", 17, weight=700, anchor="start")
    steps = [(ic_teacher, ["A teacher model may peek", "at the next 50 rows", "(training only)."]),
             (ic_clean, ["Its answers are cleaner", "than the raw, noisy", "target (gray)."]),
             (ic_students, ["The three networks learn", "from those cleaner", "answers."]),
             (ic_calendar, ["Last lesson on the newest", "data, where we know which", "rows are scored."])]
    for i, (icon, lines) in enumerate(steps):
        cx = 142 + i * 182
        icon(s, cx, py + 100)
        for k, ln in enumerate(lines):
            s.text(cx, py + 168 + k * 19, ln, 13, "ink2")
        if i < 3:
            s.add(f'<path d="M{cx + 82},{py + 90} l10,9 l-10,9" fill="none" stroke="{t["muted"]}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>')
    s.add(f'<rect x="826" y="{py}" width="338" height="{ph}" rx="14" fill="none" stroke="{t["border"]}" stroke-width="1.5"/>')
    s.text(848, py + 32, "Fast enough for the rules", 17, weight=700, anchor="start")
    ic_stopwatch(s, 925, py + 105)
    ic_chip(s, 1065, py + 105)
    for k, ln in enumerate(["About 0.1 milliseconds per row:", "the whole test ran in 51 of the", "60 allowed minutes, on a single", "processor core."]):
        s.text(995, py + 168 + k * 19, ln, 13, "ink2")
    return s.render()


DIAGRAMS = {"howitworks": diagram_howitworks, "problem": diagram_problem, "inference": diagram_inference, "training": diagram_training,
            "crossfit": diagram_crossfit}


# ------------------------------------------------------------------------------------------------ charts
def mpl_theme(name):
    import matplotlib as mpl
    t = THEMES[name]
    mpl.rcParams.update({
        "svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 10.5,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11.5,
        "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    })
    return t


def _save(fig, name, theme):
    fig.savefig(OUT / f"{name}-{theme}.svg", bbox_inches="tight", pad_inches=0.25)
    import matplotlib.pyplot as plt
    plt.close(fig)


def chart_ticks(theme, data):
    """Target values of one validation sequence per regime: spikes on a grid whose spacing differs by regime."""
    import matplotlib.pyplot as plt
    import numpy as np
    t = mpl_theme(theme)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    for r, ax in enumerate(axes):
        y, unit = data["ticks"][r]
        v, c = np.unique(np.round(y[np.abs(y) <= 2.05], 6), return_counts=True)
        ax.vlines(v, 0.8, c, color=t["blue"], linewidth=2)
        ax.set_yscale("log")
        ax.set_xlim(-2.2, 2.2)
        ax.set_title(f"Regime {r} · tick {unit:.3f}", fontsize=11)
        ax.set_xlabel("target value")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("rows (log scale)")
    fig.suptitle("Every sequence's target sits on its own grid: y = k × tick, with one of four tick sizes",
                 x=0.07, ha="left", fontsize=12.5, fontweight="bold", color=t["ink"])
    fig.tight_layout()
    _save(fig, "ticks", theme)


def chart_teacher(theme, data):
    """300 rows of one validation sequence: target, look-ahead teacher, causal student."""
    import matplotlib.pyplot as plt
    import numpy as np
    t = mpl_theme(theme)
    y, te, st, a = data["teacher"]
    x = np.arange(a, a + len(y))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.step(x, y, where="mid", color=t["muted"], linewidth=1.6, label="real target (on the tick grid)")
    ax.plot(x, te, color=t["orange"], linewidth=2, label="teacher (also sees the next 50 rows)")
    ax.plot(x, st, color=t["blue"], linewidth=2, label="student (sees only the past)")
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("row of the sequence")
    ax.set_ylabel("target t0")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.13), ncol=3, fontsize=10)
    ax.set_title("A teacher that peeks at the future tracks the target far better: weighted correlation "
                 f"{data['teacher_wp'][0]:.2f} vs {data['teacher_wp'][1]:.2f} on validation", pad=30)
    fig.tight_layout()
    _save(fig, "teacher", theme)


def chart_regimes(theme, data):
    """Share of sequences per regime in train vs validation, and how hard each regime is."""
    import matplotlib.pyplot as plt
    import numpy as np
    t = mpl_theme(theme)
    tr, va = (np.asarray(v, float) / sum(v) * 100 for v in (data["mix_train"], data["mix_valid"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"wspace": 0.28})
    xx = np.arange(4)
    w = 0.38
    for off, vals, c, lab in ((-w / 2 - 0.01, tr, t["blue"], "train"), (w / 2 + 0.01, va, t["orange"], "validation")):
        bars = a1.bar(xx + off, vals, w, color=c, label=lab)
        for b, v in zip(bars, vals):
            a1.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9.5, color=t["ink2"])
    a1.set_xticks(xx, [f"regime {r}\ntick {u:.3f}" for r, u in enumerate(data["units"])])
    a1.set_ylabel("% of sequences")
    a1.set_ylim(0, 52)
    a1.legend(loc="upper left", ncol=2)
    a1.set_title("Validation has more fine-tick sequences")
    a1.grid(axis="x", visible=False)
    wp, share = data["wp"], data["weight_share"]
    bars = a2.bar(xx, wp, 0.55, color=t["blue"])
    for b, v in zip(bars, wp):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9.5, color=t["ink2"])
    a2.set_xticks(xx, [f"regime {r}\n{s}%" for r, s in enumerate(share)])
    a2.set_xlabel("regime and its share of the metric weight")
    a2.set_ylabel("weighted correlation")
    a2.set_ylim(0, 0.95)
    a2.set_title("Regime 3: half the metric weight, hardest to predict")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "regimes", theme)


def chart_progress(theme, data):
    """Public leaderboard score per submission: the early climb, then the zoomed late phase."""
    import matplotlib.pyplot as plt
    t = mpl_theme(theme)
    early, late = data["lb_early"], data["lb_late"]
    PLACE = {"sub01": "dr", "sub26a": "ul"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.22})
    for ax, rows, fmt in ((a1, early, "{:.3f}"), (a2, late, "{:.4f}")):
        xs = range(len(rows))
        ys = [r[1] for r in rows]
        ax.plot(xs, ys, color=t["blue"], marker="o", markersize=6, markeredgecolor=t["surface"], markeredgewidth=1.5)
        for i, (name, v, note) in enumerate(rows):
            if note:  # labels sit off the rising line: above-left ("ul"), below-right ("dr") or below-left ("dl")
                pos = PLACE.get(name, "ul" if i % 2 else "dr")
                off = {"ul": (-7, 7), "dr": (7, -13), "dl": (-7, -14)}[pos]
                ax.annotate(note, (i, v), xytext=off, textcoords="offset points",
                            ha="left" if pos == "dr" else "right", fontsize=8.8, color=t["ink2"])
        ax.set_xticks(list(xs), [r[0] for r in rows], rotation=45, ha="right", fontsize=9)
        ax.grid(axis="x", visible=False)
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.18
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt.format(v)))
    a1.set_ylabel("public leaderboard score")
    a1.set_title("October: from the baseline to a 3-model blend")
    a2.set_title("Then: regimes, soft labels, stack (zoomed in)")
    fig.tight_layout()
    _save(fig, "progress", theme)


def chart_latency(theme, data):
    """Per-member cost per row and the server time of the main submissions vs the 60-minute limit."""
    import matplotlib.pyplot as plt
    import numpy as np
    t = mpl_theme(theme)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"wspace": 0.45})
    names, vals = zip(*data["member_us"])
    colors = [t["muted"] if "exact" in n else t["blue"] for n in names]
    yy = np.arange(len(names))[::-1]
    a1.barh(yy, vals, 0.6, color=colors)
    for y_, v in zip(yy, vals):
        a1.text(v + 1.2, y_, f"{v:.0f} µs", va="center", fontsize=9.5, color=t["ink2"])
    a1.set_yticks(yy, names)
    a1.set_xlabel("µs per row (local machine)")
    a1.set_xlim(0, 82)
    a1.grid(axis="y", visible=False)
    a1.set_title("Cost of each piece, per row")
    subs, mins = zip(*data["server_min"])
    a2.plot(range(len(subs)), mins, color=t["blue"], marker="o", markersize=6, markeredgecolor=t["surface"],
            markeredgewidth=1.5)
    for i, m in enumerate(mins):
        a2.annotate(f"{m:.0f}", (i, m), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=t["ink2"])
    a2.axhline(60, color=t["orange"], linewidth=1.5, linestyle="--")
    a2.text(len(subs) - 1, 61.2, "limit: 60 min", ha="right", fontsize=9.5, color=t["ink2"])
    a2.set_xticks(range(len(subs)), subs, rotation=45, ha="right", fontsize=9)
    a2.set_ylim(0, 68)
    a2.set_ylabel("minutes on the scoring server")
    a2.grid(axis="x", visible=False)
    a2.set_title("Server time of each submission")
    fig.tight_layout()
    _save(fig, "latency", theme)


def load_data():
    import numpy as np
    from wnn.data import ARTIFACTS, SequenceReader
    from wnn.nets import REGIME_EDGES, REGIME_UNITS
    rd = SequenceReader("valid")
    uv = np.load(ARTIFACTS / "preds" / "units_valid.npy")[:, 0]
    ut = np.load(ARTIFACTS / "preds" / "units_train.npy")[:, 0]
    rv, rt = np.searchsorted(REGIME_EDGES, uv), np.searchsorted(REGIME_EDGES, ut)
    ticks = []
    for r in range(4):
        g = int(np.flatnonzero(rv == r)[0])
        ticks.append((rd.read(g).Y[99:, 0].astype(np.float64), REGIME_UNITS[r]))
    g, a, n = 0, 2200, 120  # sequence 0 is regime 0
    sl = slice(g * 20000 + a, g * 20000 + a + n)
    te = np.load(ARTIFACTS / "nn/teacher_la50_gru/pred_valid.npy", mmap_mode="r")[sl, 0]
    st = np.load(ARTIFACTS / "nn/tcn64_ema_c3_st16/pred_valid.npy", mmap_mode="r")[sl, 0]
    y = rd.read(g).Y[a:a + n, 0]
    return {
        "ticks": ticks,
        "teacher": (np.asarray(y), np.asarray(te), np.asarray(st), a),
        "units": REGIME_UNITS,
        # target t0, |y|-weighted Pearson over every 8th validation sequence (rows >= 99): teacher_la50_gru vs the
        # train-only TCN student (computed 2026-11-21; t1: 0.708 vs 0.540)
        "teacher_wp": (0.725, 0.549),
        "mix_train": np.bincount(rt, minlength=4), "mix_valid": np.bincount(rv, minlength=4),
        # EXPERIMENTS.md, 2026-10-20: out-of-fold WP of the sub11 blend per regime, and each regime's share of the metric weight
        "wp": [0.800, 0.773, 0.736, 0.434], "weight_share": [28, 16, 8, 48],
        # EXPERIMENTS.md §2 and LB results (public score; name, score, note)
        "lb_early": [("baseline", 0.5935, "organizers' GRU"), ("sub01", 0.6178, "LightGBM"), ("sub02", 0.6449, "own GRU"),
                     ("sub04", 0.6488, "metric loss"), ("sub05", 0.6548, "+ minGRU"), ("sub07", 0.6585, "valid fine-tune"),
                     ("sub09", 0.6604, "+ SSM, EMA"), ("sub11", 0.6611, "+ TCN")],
        "lb_late": [("sub11", 0.66110, ""), ("sub14b", 0.66134, "FT mix16"), ("sub17b", 0.66155, "r3 specialist"),
                    ("sub18", 0.66226, "4 regime routes"), ("sub22", 0.66277, "soft-label FT"),
                    ("sub23b", 0.66364, "soft TCN"), ("sub23d", 0.66418, "soft r3"), ("sub23g", 0.66453, "soft SSM"),
                    ("sub24", 0.66487, "row stack"), ("sub25", 0.66506, "SSM under stack"), ("sub26a", 0.66502, "final")],
        # EXPERIMENTS.md latency audit (local, loaded machine) and §44
        "member_us": [("overhead", 9), ("minGRU", 33), ("TCN", 35), ("SSM, exact exp", 68), ("SSM, fast exp", 40)],
        "server_min": [("sub09", 57.0), ("sub09b", 46.4), ("sub11", 53.2), ("sub18", 55.3), ("sub24", 48.8),
                       ("sub24f", 42.4), ("sub25", 51.0), ("sub26a", 51.4)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagrams", action="store_true", help="only the diagrams (no data needed)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for theme, t in THEMES.items():
        for name, fn in DIAGRAMS.items():
            (OUT / f"{name}-{theme}.svg").write_text(fn(t), encoding="utf-8")
    if a.diagrams:
        return
    data = load_data()
    for theme in THEMES:
        for fn in (chart_ticks, chart_teacher, chart_regimes, chart_progress, chart_latency):
            fn(theme, data)
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
