"""
DPVO Motion Encoder -- Architecture Animation for Paper
=======================================================
Render commands
---------------
    # Animated video (high quality, for supplemental / presentation):
    manim -pqh dpvo_motion_encoder.py DPVOMotionEncoderDiagram

    # Ultra-high quality 4K video (for paper supplemental):
    manim -pqk dpvo_motion_encoder.py DPVOMotionEncoderDiagram

    # Static PNG for direct LaTeX inclusion (fastest):
    manim -sqh dpvo_motion_encoder.py DPVOMotionEncoderDiagram
    # -> media/images/dpvo_motion_encoder/DPVOMotionEncoderDiagram_ManimCE_vX.png

Dependencies
------------
    pip install manim

Fix note (v0.19+)
-----------------
    interpolate_color() requires ManimColor objects, not raw hex strings.
    ALL palette constants are wrapped with ManimColor() below.
    A thin ic() wrapper guarantees this everywhere in the file.
"""

from manim import *
from manim.utils.color import ManimColor
import numpy as np

# ---------------------------------------------------------------------------
# Palette  --  all wrapped as ManimColor so interpolate_color works in v0.19+
# ---------------------------------------------------------------------------
BG         = ManimColor("#0D1117")
GRID_COL   = ManimColor("#1C2333")
FROZEN_COL = ManimColor("#4A90D9")   # blue-steel  -> frozen / parameter-free
TRAIN_COL  = ManimColor("#F5A623")   # amber       -> trainable
ACCENT_COL = ManimColor("#00E5CC")   # teal        -> tensors / data flow
ARROW_COL  = ManimColor("#C9D1D9")   # light gray  -> arrows
DIM_COL    = ManimColor("#8B949E")   # muted       -> secondary labels
WHITE      = ManimColor("#E6EDF3")
RED_SOFT   = ManimColor("#FF7B72")
GREEN_SOFT = ManimColor("#7EE787")
DARK_BLUE  = ManimColor("#0D1B2A")   # shape-badge fill
DEEP_TEAL  = ManimColor("#0A2540")   # feature-map dark end

config.background_color = BG

TITLE_FONT = "Courier New"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ic(c1, c2, alpha):
    """Safe interpolate_color: coerces both args to ManimColor."""
    return interpolate_color(ManimColor(c1), ManimColor(c2), float(alpha))


def mono(text, size=18, color=None, weight=NORMAL):
    if color is None:
        color = WHITE
    return Text(text, font=TITLE_FONT, font_size=size,
                color=ManimColor(color), weight=weight)


def tiny_label(text, color=None):
    if color is None:
        color = DIM_COL
    return mono(text, size=11, color=ManimColor(color))


def shape_badge(text):
    """Teal tensor-shape annotation badge."""
    t = mono(text, size=10, color=ACCENT_COL)
    bg = SurroundingRectangle(
        t, buff=0.08, corner_radius=0.05,
        fill_color=DARK_BLUE, fill_opacity=0.9,
        stroke_color=ACCENT_COL, stroke_width=0.8)
    return VGroup(bg, t)


def phase_badge(text, col):
    t = mono(text, size=9, color=ManimColor(col))
    bg = SurroundingRectangle(
        t, buff=0.07, corner_radius=0.05,
        fill_color=ic(BG, col, 0.15), fill_opacity=1,
        stroke_color=ManimColor(col), stroke_width=0.7)
    return VGroup(bg, t)


# ---------------------------------------------------------------------------
# Block factories
# ---------------------------------------------------------------------------

def frozen_box(title, subtitle="", width=2.2, height=0.85):
    col = FROZEN_COL
    rect = RoundedRectangle(
        corner_radius=0.12, width=width, height=height,
        fill_color=ic(BG, col, 0.12), fill_opacity=1,
        stroke_color=col, stroke_width=1.6)
    title_t = mono(title, size=13, color=col, weight=BOLD).move_to(rect)
    group = VGroup(rect, title_t)
    if subtitle:
        sub = tiny_label(subtitle, color=ic(DIM_COL, col, 0.45))
        sub.next_to(title_t, DOWN, buff=0.08)
        group.add(sub)
    return group


def trainable_box(title, subtitle="", width=2.2, height=0.85, col=None):
    if col is None:
        col = TRAIN_COL
    col = ManimColor(col)
    rect = RoundedRectangle(
        corner_radius=0.12, width=width, height=height,
        fill_color=ic(BG, col, 0.10), fill_opacity=1,
        stroke_color=col, stroke_width=1.6)
    stripe = RoundedRectangle(
        corner_radius=0.12, width=width - 0.04, height=0.13,
        fill_color=col, fill_opacity=0.22, stroke_width=0)
    stripe.align_to(rect, UP).shift(DOWN * 0.02)
    title_t = mono(title, size=13, color=col, weight=BOLD).move_to(rect)
    group = VGroup(rect, stripe, title_t)
    if subtitle:
        sub = tiny_label(subtitle, color=ic(DIM_COL, col, 0.45))
        sub.next_to(title_t, DOWN, buff=0.08)
        group.add(sub)
    return group


# ---------------------------------------------------------------------------
# Visual data-structure widgets
# ---------------------------------------------------------------------------

def tiny_feature_map(rows=5, cols=5, cell=0.115, seed=0):
    """Pseudo-heatmap grid representing a (B,128,H/4,W/4) feature tensor."""
    rng = np.random.default_rng(seed)
    vals = rng.random((rows, cols))
    group = VGroup()
    for r in range(rows):
        for c in range(cols):
            col = ic(DEEP_TEAL, FROZEN_COL, float(vals[r, c]))
            sq = Square(side_length=cell,
                        fill_color=col, fill_opacity=0.95,
                        stroke_color=GRID_COL, stroke_width=0.35)
            sq.move_to(RIGHT * c * cell + DOWN * r * cell)
            group.add(sq)
    group.move_to(ORIGIN)
    return group


def frame_thumbnail(col, label="t"):
    """Stylised camera-frame input widget."""
    col = ManimColor(col)
    outer = RoundedRectangle(
        corner_radius=0.08, width=1.1, height=0.82,
        fill_color=ManimColor("#1A2332"), fill_opacity=1,
        stroke_color=col, stroke_width=1.2)
    lines = VGroup(*[
        Line(LEFT * 0.38 + UP * (0.22 - i * 0.13),
             RIGHT * 0.38 + UP * (0.22 - i * 0.13),
             stroke_color=ic(BG, col, 0.22 + i * 0.12),
             stroke_width=0.7)
        for i in range(4)
    ])
    lbl = mono(label, size=9, color=col)
    lbl.next_to(outer, DOWN, buff=0.04)
    return VGroup(outer, lines, lbl)


def patch_dot_grid(rows=4, cols=8):
    """Grid of coloured dots representing 64 sampled patches."""
    total = rows * cols
    group = VGroup()
    for r in range(rows):
        for c in range(cols):
            t = (r * cols + c) / max(total - 1, 1)
            d = Dot(radius=0.045, color=ic(FROZEN_COL, ACCENT_COL, t))
            d.move_to(RIGHT * (c * 0.13 - (cols - 1) * 0.065) +
                      UP    * (r * 0.13 - (rows - 1) * 0.065))
            group.add(d)
    return group


def corr_volume_vis(cols=5, rows=4, depth=3, cell=0.115):
    """Isometric 3-layer block representing the correlation volume."""
    ox, oy = 0.12, -0.10
    group = VGroup()
    for d in range(depth):
        for r in range(rows):
            for c in range(cols):
                v = float(np.sin(r * 1.1 + c * 0.8 + d * 0.5) * 0.5 + 0.5)
                col = ic(FROZEN_COL, ACCENT_COL, v)
                sq = Square(side_length=cell,
                            fill_color=col,
                            fill_opacity=0.30 + 0.25 * d,
                            stroke_color=BG, stroke_width=0.4)
                sq.move_to(RIGHT * (c * cell + d * ox) +
                           UP    * (r * cell + d * oy))
                group.add(sq)
    return group


def gaussian_peak_vis(size=5, cell=0.115):
    """5x5 Gaussian blob illustrating the windowed soft-argmax peak."""
    cx = cy = size // 2
    group = VGroup()
    for r in range(size):
        for c in range(size):
            dist = float(np.sqrt((r - cy) ** 2 + (c - cx) ** 2))
            v = float(np.exp(-dist ** 2 / 1.3))
            col = ic(DEEP_TEAL, ACCENT_COL, v)
            sq = Square(side_length=cell,
                        fill_color=col, fill_opacity=0.95,
                        stroke_color=BG, stroke_width=0.4)
            sq.move_to(RIGHT * (c * cell - cx * cell) +
                       UP    * (r * cell - cy * cell))
            group.add(sq)
    return group


def motion_token_bars(n=32, width=1.8, height=0.30, seed=42):
    """Mini bar chart representing the 128-D latent motion token."""
    rng = np.random.default_rng(seed)
    vals = np.abs(rng.normal(0, 1, n))
    vals = vals / vals.max()
    bar_w = width / n - 0.012
    group = VGroup()
    for i, v in enumerate(vals):
        col = ic(ACCENT_COL, TRAIN_COL, float(v))
        bar = Rectangle(width=bar_w, height=float(v) * height,
                        fill_color=col, fill_opacity=0.9, stroke_width=0)
        bar.align_to(ORIGIN, DOWN)
        bar.shift(RIGHT * (i * (width / n) - width / 2 + bar_w / 2))
        group.add(bar)
    bg = Rectangle(width=width + 0.10, height=height + 0.08,
                   fill_color=GRID_COL, fill_opacity=0.5, stroke_width=0)
    bg.align_to(group, DOWN).shift(DOWN * 0.03)
    return VGroup(bg, group)


# ---------------------------------------------------------------------------
# Arrow helpers
# ---------------------------------------------------------------------------

def flow_arrow(start, end, col=None):
    if col is None:
        col = ARROW_COL
    return Arrow(start, end, buff=0.0, stroke_width=1.5,
                 tip_length=0.16, color=ManimColor(col),
                 max_tip_length_to_length_ratio=0.28)


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def make_legend():
    items = [
        (FROZEN_COL, "Frozen / parameter-free"),
        (TRAIN_COL,  "Trainable (head only)"),
        (ACCENT_COL, "Tensor / feature flow"),
    ]
    rows = VGroup()
    for col, label in items:
        dot = Dot(radius=0.07, color=ManimColor(col))
        txt = mono(label, size=10, color=ManimColor(col))
        txt.next_to(dot, RIGHT, buff=0.12)
        rows.add(VGroup(dot, txt))
    rows.arrange(RIGHT, buff=0.40)
    return rows


# ---------------------------------------------------------------------------
# Main Scene
# ---------------------------------------------------------------------------

class DPVOMotionEncoderDiagram(Scene):

    def construct(self):

        # Background grid
        grid = VGroup(
            *[Line(LEFT * 8 + UP * y, RIGHT * 8 + UP * y,
                   stroke_color=GRID_COL, stroke_width=0.4)
              for y in np.arange(-4.5, 4.6, 0.45)],
            *[Line(UP * 4.5 + RIGHT * x, DOWN * 4.5 + RIGHT * x,
                   stroke_color=GRID_COL, stroke_width=0.4)
              for x in np.arange(-8, 8.1, 0.45)],
        )
        self.add(grid)

        # Title
        title = mono("DPVO  MOTION  ENCODER", size=22, color=WHITE, weight=BOLD)
        subtitle = mono(
            "Frozen DPVO trunk  .  Correlation Tracker  .  Trainable Displacement Head",
            size=11, color=DIM_COL)
        subtitle.next_to(title, DOWN, buff=0.12)
        title_grp = VGroup(title, subtitle).to_edge(UP, buff=0.22)
        self.play(FadeIn(title_grp, shift=DOWN * 0.15), run_time=0.7)

        # Legend
        legend = make_legend().to_corner(DR, buff=0.28).shift(UP * 0.08)
        self.play(FadeIn(legend), run_time=0.4)

        Y = 1.4   # pipeline vertical centre

        # ===================================================================
        # STAGE 0 -- Input frames
        # ===================================================================
        fr_prev = frame_thumbnail(FROZEN_COL, "t-1")
        fr_curr = frame_thumbnail(ACCENT_COL, "t")
        fr_prev.move_to(LEFT * 6.1 + UP * (Y + 0.78))
        fr_curr.move_to(LEFT * 6.1 + UP * (Y - 0.78))

        inp_badge = phase_badge("INPUT  (B, 2, 3, H, W)", ACCENT_COL)
        inp_badge.next_to(fr_prev, UP, buff=0.14)

        self.play(
            FadeIn(fr_prev, shift=RIGHT * 0.15),
            FadeIn(fr_curr, shift=RIGHT * 0.15),
            FadeIn(inp_badge),
            run_time=0.6,
        )

        # ===================================================================
        # STAGE 1 -- Preprocess
        # ===================================================================
        pre = frozen_box("PREPROCESS",
                         "undo ImageNet norm\n2x - 0.5  (DPVO norm)",
                         width=1.95)
        pre.move_to(LEFT * 4.1 + UP * Y)

        a_p1 = flow_arrow(fr_prev.get_right(), pre.get_top(),    col=FROZEN_COL)
        a_p2 = flow_arrow(fr_curr.get_right(), pre.get_bottom(), col=ACCENT_COL)

        self.play(Create(a_p1), Create(a_p2),
                  FadeIn(pre, shift=RIGHT * 0.12), run_time=0.6)

        # ===================================================================
        # STAGE 2 -- Frozen fnet trunk
        # ===================================================================
        trunk = frozen_box("DPVO  fnet",
                           "BasicEncoder4  stride 4\n128-D feature maps",
                           width=2.5, height=1.05)
        trunk.move_to(LEFT * 1.7 + UP * Y)

        frz_badge = phase_badge("FROZEN  |  grad = 0", FROZEN_COL)
        frz_badge.next_to(trunk, UP, buff=0.10)

        a_pre_trunk = flow_arrow(pre.get_right(), trunk.get_left(), col=FROZEN_COL)

        fmap_p = tiny_feature_map(rows=5, cols=5, cell=0.115, seed=7)
        fmap_c = tiny_feature_map(rows=5, cols=5, cell=0.115, seed=13)
        fmap_p.next_to(trunk, RIGHT, buff=0.55).shift(UP * 0.50)
        fmap_c.next_to(trunk, RIGHT, buff=0.55).shift(DOWN * 0.50)

        sh_fp = shape_badge("(B,128,H/4,W/4)")
        sh_fc = shape_badge("(B,128,H/4,W/4)")
        sh_fp.next_to(fmap_p, RIGHT, buff=0.10)
        sh_fc.next_to(fmap_c, RIGHT, buff=0.10)

        a_t_fp = flow_arrow(trunk.get_right(), fmap_p.get_left(), col=FROZEN_COL)
        a_t_fc = flow_arrow(trunk.get_right(), fmap_c.get_left(), col=FROZEN_COL)

        self.play(Create(a_pre_trunk),
                  FadeIn(trunk, shift=RIGHT * 0.12), FadeIn(frz_badge),
                  run_time=0.6)
        self.play(Create(a_t_fp), FadeIn(fmap_p), FadeIn(sh_fp),
                  Create(a_t_fc), FadeIn(fmap_c), FadeIn(sh_fc),
                  run_time=0.65)

        # ===================================================================
        # STAGE 3 -- Patch sampling from fmap_p
        # ===================================================================
        patch_box = frozen_box("PATCH SAMPLE",
                               "64 patches on 8x8 grid\nfrom frame t-1",
                               width=2.2)
        patch_box.move_to(RIGHT * 0.9 + UP * (Y + 1.05))

        pdots = patch_dot_grid()
        pdots.next_to(patch_box, RIGHT, buff=0.20)

        sh_patch = shape_badge("(B, 64, 128)")
        sh_patch.next_to(pdots, RIGHT, buff=0.10)

        a_fp_patch = flow_arrow(fmap_p.get_top(), patch_box.get_left(),
                                col=FROZEN_COL)

        self.play(Create(a_fp_patch),
                  FadeIn(patch_box, shift=DOWN * 0.10), run_time=0.5)
        self.play(FadeIn(pdots, lag_ratio=0.04), FadeIn(sh_patch), run_time=0.65)

        # ===================================================================
        # STAGE 4 -- Cosine correlation + local search window
        # ===================================================================
        corr_box = frozen_box("COSINE CORR\n+ LOCAL SEARCH",
                              "L2-norm  |  cosine in [-1,1]\nwindow +/-32 cells",
                              width=2.35, height=1.05)
        corr_box.move_to(RIGHT * 0.9 + UP * (Y - 0.68))

        corr_vis = corr_volume_vis()
        corr_vis.next_to(corr_box, RIGHT, buff=0.22)

        sh_corr = shape_badge("(B, 64, H, W)")
        sh_corr.next_to(corr_vis, RIGHT, buff=0.10)

        a_patch_corr = flow_arrow(patch_box.get_bottom(), corr_box.get_top(),
                                  col=FROZEN_COL)
        a_fc_corr    = flow_arrow(fmap_c.get_top(), corr_box.get_bottom(),
                                  col=ACCENT_COL)

        self.play(Create(a_patch_corr), Create(a_fc_corr),
                  FadeIn(corr_box, shift=RIGHT * 0.10), run_time=0.6)
        self.play(FadeIn(corr_vis, lag_ratio=0.01), FadeIn(sh_corr), run_time=0.65)

        # ===================================================================
        # STAGE 5 -- Windowed soft-argmax
        # ===================================================================
        argmax_box = frozen_box("WINDOWED\nSOFT-ARGMAX",
                                "hard peak + 7x7 window\ntemp=20  sub-pixel",
                                width=2.35, height=1.05)
        argmax_box.move_to(RIGHT * 3.7 + UP * (Y - 0.68))

        peak_vis = gaussian_peak_vis()
        peak_vis.next_to(argmax_box, RIGHT, buff=0.22)

        sh_flow = shape_badge("flow (dx,dy)\n+ sharp score")
        sh_flow.next_to(peak_vis, RIGHT, buff=0.10)

        a_corr_argmax = flow_arrow(corr_box.get_right(), argmax_box.get_left(),
                                   col=FROZEN_COL)

        self.play(Create(a_corr_argmax),
                  FadeIn(argmax_box, shift=RIGHT * 0.10), run_time=0.5)
        self.play(FadeIn(peak_vis, lag_ratio=0.015), FadeIn(sh_flow), run_time=0.6)

        # ===================================================================
        # STAGE 6 -- Per-patch token assembly  132-D
        # ===================================================================
        assemble_box = frozen_box("TOKEN ASSEMBLY",
                                  "feat(128)|dx|dy|norm|sharp\n=> 132-D per patch",
                                  width=2.5, height=1.05)
        assemble_box.move_to(RIGHT * 3.7 + UP * (Y + 1.05))

        sh_tok = shape_badge("(B, 64, 132)")
        sh_tok.next_to(assemble_box, RIGHT, buff=0.12)

        a_peak_asm = flow_arrow(peak_vis.get_top(), assemble_box.get_bottom(),
                                col=FROZEN_COL)
        a_fp_asm = Arrow(fmap_p.get_right(), assemble_box.get_left(),
                         buff=0.05, stroke_width=1.2, tip_length=0.13,
                         color=FROZEN_COL,
                         max_tip_length_to_length_ratio=0.28)

        cache_lbl = mono("  CACHEABLE -- zero trainable params above  ",
                         size=9, color=FROZEN_COL)
        cache_bg = SurroundingRectangle(
            cache_lbl, buff=0.07, corner_radius=0.05,
            fill_color=ic(BG, FROZEN_COL, 0.08), fill_opacity=1,
            stroke_color=FROZEN_COL, stroke_width=0.6)
        cache_grp = VGroup(cache_bg, cache_lbl)
        cache_grp.next_to(assemble_box, UP, buff=0.14)

        self.play(Create(a_peak_asm), Create(a_fp_asm),
                  FadeIn(assemble_box, shift=DOWN * 0.10), FadeIn(sh_tok),
                  run_time=0.6)
        self.play(FadeIn(cache_grp), run_time=0.4)

        # ===================================================================
        # Trainable boundary divider
        # ===================================================================
        div_x = 5.55
        divider = DashedLine(
            UP * 3.4 + RIGHT * div_x,
            DOWN * 3.1 + RIGHT * div_x,
            color=TRAIN_COL, stroke_width=0.9,
            dash_length=0.12, dashed_ratio=0.45)
        div_lbl = mono("TRAINABLE  REGION", size=9, color=TRAIN_COL)
        div_lbl.next_to(divider, RIGHT, buff=0.12).shift(UP * 2.7)

        self.play(Create(divider), FadeIn(div_lbl), run_time=0.45)

        # ===================================================================
        # STAGE 7a -- Linear 132 -> 128
        # ===================================================================
        linear_box = trainable_box("Linear  132 -> 128",
                                   "per-patch projection",
                                   width=2.35)
        linear_box.move_to(RIGHT * 6.0 + UP * (Y + 1.05))

        a_asm_lin = flow_arrow(assemble_box.get_right(), linear_box.get_left(),
                               col=TRAIN_COL)

        self.play(Create(a_asm_lin),
                  FadeIn(linear_box, shift=RIGHT * 0.15), run_time=0.5)

        # ===================================================================
        # STAGE 7b -- Attentive pool (1-query MHA)
        # ===================================================================
        pool_box = trainable_box("ATTENTIVE  POOL",
                                 "1-query MHA\nacross 64 patches",
                                 width=2.35, height=1.05)
        pool_box.move_to(RIGHT * 6.0 + UP * (Y - 0.05))

        a_lin_pool = flow_arrow(linear_box.get_bottom(), pool_box.get_top(),
                                col=TRAIN_COL)

        self.play(Create(a_lin_pool),
                  FadeIn(pool_box, shift=DOWN * 0.10), run_time=0.5)

        # ===================================================================
        # STAGE 7c -- MLP  128 -> 256 -> 128
        # ===================================================================
        mlp_box = trainable_box("MLP  128 -> 256 -> 128",
                                "motion embedding",
                                width=2.35)
        mlp_box.move_to(RIGHT * 6.0 + UP * (Y - 1.20))

        a_pool_mlp = flow_arrow(pool_box.get_bottom(), mlp_box.get_top(),
                                col=TRAIN_COL)

        self.play(Create(a_pool_mlp),
                  FadeIn(mlp_box, shift=DOWN * 0.10), run_time=0.5)

        # ===================================================================
        # Output token (128-D bar chart)
        # ===================================================================
        token_vis = motion_token_bars(n=32, width=1.65, height=0.28, seed=99)
        token_vis.next_to(mlp_box, RIGHT, buff=0.24)

        sh_tok_out = shape_badge("128-D motion token")
        sh_tok_out.next_to(token_vis, UP, buff=0.09)

        a_mlp_tok = flow_arrow(mlp_box.get_right(), token_vis.get_left(),
                               col=TRAIN_COL)

        self.play(Create(a_mlp_tok),
                  FadeIn(token_vis), FadeIn(sh_tok_out), run_time=0.6)

        # ===================================================================
        # STAGE 8 -- Delta regression head
        # ===================================================================
        delta_box = trainable_box("DELTA  HEAD",
                                  "Linear  128 -> 2",
                                  width=2.0, col=RED_SOFT)
        delta_box.next_to(token_vis, DOWN, buff=0.30)

        a_tok_delta = flow_arrow(token_vis.get_bottom(), delta_box.get_top(),
                                 col=TRAIN_COL)

        loss_note = mono("Huber loss  delta=0.1", size=9, color=RED_SOFT)
        loss_note.next_to(delta_box, DOWN, buff=0.09)

        self.play(Create(a_tok_delta), FadeIn(delta_box), run_time=0.5)
        self.play(FadeIn(loss_note), run_time=0.3)

        # ===================================================================
        # Final output arrow
        # ===================================================================
        out_arr = Arrow(
            delta_box.get_right(),
            delta_box.get_right() + RIGHT * 0.75,
            buff=0.0, stroke_width=2.2, tip_length=0.20,
            color=GREEN_SOFT,
            max_tip_length_to_length_ratio=0.35)
        out_lbl = mono("(dx, dy)  metres", size=13,
                       color=GREEN_SOFT, weight=BOLD)
        out_lbl.next_to(out_arr, RIGHT, buff=0.12)

        self.play(Create(out_arr), FadeIn(out_lbl), run_time=0.5)

        # ===================================================================
        # Bottom step-number strip
        # ===================================================================
        y_strip = -3.25
        steps = [
            (-6.1, "1", FROZEN_COL, "Input"),
            (-4.1, "2", FROZEN_COL, "Preproc"),
            (-1.7, "3", FROZEN_COL, "fnet"),
            ( 0.9, "4", FROZEN_COL, "Corr"),
            ( 3.7, "5", FROZEN_COL, "Argmax+Asm"),
            ( 6.0, "6", TRAIN_COL,  "Head"),
        ]
        for x, num, col, lbl_t in steps:
            col = ManimColor(col)
            circ = Circle(radius=0.18,
                          fill_color=ic(BG, col, 0.20), fill_opacity=1,
                          stroke_color=col, stroke_width=1.2)
            circ.move_to(RIGHT * x + UP * y_strip)
            num_t = mono(num, size=10, color=col, weight=BOLD).move_to(circ)
            lbl_obj = tiny_label(lbl_t, color=ic(DIM_COL, col, 0.5))
            lbl_obj.next_to(circ, DOWN, buff=0.08)
            self.play(FadeIn(VGroup(circ, num_t, lbl_obj)), run_time=0.16)

        # ===================================================================
        # Validation metrics panel (bottom-left)
        # ===================================================================
        m_title = mono("VALIDATION  METRICS", size=10, color=DIM_COL, weight=BOLD)
        m1 = mono("Delta-MAE  0.055 m   |   bias  [+0.005, -0.008]",
                  size=10, color=WHITE)
        m2 = mono("Direction error  7.9 deg   |   scale  0.997",
                  size=10, color=WHITE)
        m3 = mono("Dead-reckoning drift  0.8% - 7.4%  path length",
                  size=10, color=ACCENT_COL)
        m_grp = VGroup(m_title, m1, m2, m3)
        m_grp.arrange(DOWN, buff=0.10, aligned_edge=LEFT)
        m_bg = SurroundingRectangle(
            m_grp, buff=0.14, corner_radius=0.08,
            fill_color=GRID_COL, fill_opacity=0.65,
            stroke_color=DIM_COL, stroke_width=0.7)
        metrics = VGroup(m_bg, m_grp).to_corner(DL, buff=0.22)

        self.play(FadeIn(metrics, shift=UP * 0.10), run_time=0.5)

        # Final output pulse
        self.play(out_lbl.animate.scale(1.10).set_color(WHITE),  run_time=0.35)
        self.play(out_lbl.animate.scale(1/1.10).set_color(GREEN_SOFT), run_time=0.30)

        self.wait(2.5)