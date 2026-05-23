"""
DPVO Motion Encoder — Static Architecture Figure  (v3 — final)
===============================================================
Render:
    manim -sqh dpvo_static.py DPVOMotionEncoderStatic
    manim -sqk dpvo_static.py DPVOMotionEncoderStatic
"""

from manim import *
from manim.utils.color import ManimColor
import numpy as np

BG          = ManimColor("#FFFFFF")
INK         = ManimColor("#0D0D0D")
INK_MED     = ManimColor("#3A3A3A")
INK_LIGHT   = ManimColor("#666666")
RULE        = ManimColor("#CCCCCC")
FROZEN_FILL = ManimColor("#EBF4FB")
FROZEN_STR  = ManimColor("#1565C0")
TRAIN_FILL  = ManimColor("#FFF8E7")
TRAIN_STR   = ManimColor("#B45309")
CORR_FILL   = ManimColor("#F1F8F1")
CORR_STR    = ManimColor("#2E7D32")
OUT_FILL    = ManimColor("#FFF0F0")
OUT_STR     = ManimColor("#C62828")
BAND_COL    = ManimColor("#F2F2F2")
WIRE_P      = ManimColor("#1565C0")
WIRE_C      = ManimColor("#2E7D32")

config.background_color = BG
RF  = "Latin Modern Roman"
MF  = "Latin Modern Mono"
CAP = "Latin Modern Roman Caps"


def ic(c1, c2, a):
    return interpolate_color(ManimColor(c1), ManimColor(c2), float(np.clip(a,0,1)))

def T(s, sz=13, col=INK, font=RF, w=NORMAL):
    return Text(s, font=font, font_size=sz, color=ManimColor(col), weight=w)
def Ts(s, sz=10, col=INK_LIGHT): return Text(s, font=RF, font_size=sz, color=ManimColor(col))
def Tm(s, sz=9,  col=INK_LIGHT): return Text(s, font=MF, font_size=sz, color=ManimColor(col))
def Tc(s, sz=9,  col=INK_LIGHT): return Text(s, font=CAP, font_size=sz, color=ManimColor(col), weight=BOLD)

def fblock(title, sub="", w=2.6, h=0.62):
    r = RoundedRectangle(corner_radius=0.07, width=w, height=h,
                         fill_color=FROZEN_FILL, fill_opacity=1,
                         stroke_color=FROZEN_STR, stroke_width=1.3)
    bar = Rectangle(width=0.08, height=h, fill_color=FROZEN_STR,
                    fill_opacity=1, stroke_width=0).align_to(r,LEFT).align_to(r,UP)
    ti = T(title, sz=12, col=INK, w=BOLD).move_to(r).shift(RIGHT*0.04)
    g = VGroup(r, bar, ti)
    if sub:
        g.add(Ts(sub, sz=9).next_to(ti, DOWN, buff=0.045))
    return g

def tblock(title, sub="", w=2.6, h=0.62, fc=None, sc=None):
    fc = ManimColor(fc or TRAIN_FILL); sc = ManimColor(sc or TRAIN_STR)
    r = RoundedRectangle(corner_radius=0.07, width=w, height=h,
                         fill_color=fc, fill_opacity=1,
                         stroke_color=sc, stroke_width=1.3)
    bar = Rectangle(width=0.08, height=h, fill_color=sc,
                    fill_opacity=1, stroke_width=0).align_to(r,LEFT).align_to(r,UP)
    ti = T(title, sz=12, col=INK, w=BOLD).move_to(r).shift(RIGHT*0.04)
    g = VGroup(r, bar, ti)
    if sub:
        g.add(Ts(sub, sz=9).next_to(ti, DOWN, buff=0.045))
    return g

# ── widgets ──────────────────────────────────────────────────────────────────
def hmap(rows, cols, cell, seed, lo, hi):
    rng = np.random.default_rng(seed)
    v = rng.random((rows,cols))
    v = (v-v.min())/(v.max()-v.min()+1e-9)
    g = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(side_length=cell,
                        fill_color=ic(lo,hi,float(v[r,c])),
                        fill_opacity=1, stroke_color=BG, stroke_width=0.2)
            sq.move_to(RIGHT*c*cell+DOWN*r*cell)
            g.add(sq)
    g.add(Rectangle(width=cols*cell, height=rows*cell,
                    fill_opacity=0, stroke_color=ic(lo,hi,0.55),
                    stroke_width=0.7).move_to(g))
    return g.move_to(ORIGIN)

def corrvol(rows=4, cols=4, depth=3, cell=0.088):
    ox,oy = 0.085,-0.072
    rng = np.random.default_rng(7)
    base = rng.random((rows,cols))
    base = (base-base.min())/(base.max()-base.min()+1e-9)
    g = VGroup()
    for d in range(depth-1,-1,-1):
        for r in range(rows-1,-1,-1):
            for c in range(cols):
                v = float(base[r,c])*(0.35+0.65*d/max(depth-1,1))
                sq = Square(side_length=cell,
                            fill_color=ic(CORR_FILL,CORR_STR,v),
                            fill_opacity=0.92, stroke_color=BG, stroke_width=0.28)
                sq.move_to(RIGHT*(c*cell+d*ox)+UP*(r*cell+d*oy))
                g.add(sq)
    return g.move_to(ORIGIN)

def argmaxvis(cell=0.082):
    rows,cols = 6,8
    rng = np.random.default_rng(5)
    base = rng.random((rows,cols))*0.2
    pr,pc = 2,5
    for r in range(rows):
        for c in range(cols):
            base[r,c] += np.exp(-((r-pr)**2+(c-pc)**2)/1.5)*0.9
    base = np.clip(base,0,1)
    g = VGroup()
    wr0,wr1,wc0,wc1 = 1,3,4,6
    for r in range(rows):
        for c in range(cols):
            inw = wr0<=r<=wr1 and wc0<=c<=wc1
            v = float(base[r,c])
            col = ic(ManimColor("#E8F5E9"),CORR_STR,v) if inw else ic(ManimColor("#F5F5F5"),ManimColor("#AAAAAA"),v*0.4)
            sq = Square(side_length=cell,
                        fill_color=col, fill_opacity=1.0 if inw else 0.8,
                        stroke_color=BG, stroke_width=0.2)
            sq.move_to(RIGHT*c*cell+DOWN*r*cell)
            g.add(sq)
    wb = Rectangle(width=(wc1-wc0+1)*cell, height=(wr1-wr0+1)*cell,
                   fill_opacity=0, stroke_color=CORR_STR, stroke_width=1.0)
    wb.move_to(RIGHT*((wc0+(wc1-wc0)/2)*cell)+DOWN*((wr0+(wr1-wr0)/2)*cell))
    g.add(wb)
    g.add(Rectangle(width=cols*cell, height=rows*cell,
                    fill_opacity=0, stroke_color=RULE, stroke_width=0.55).move_to(g))
    return g.move_to(ORIGIN)

def patchvis(rows=3, cols=8, cell=0.080):
    rng = np.random.default_rng(3)
    v = rng.random((rows,cols))
    g = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(side_length=cell,
                        fill_color=ic(FROZEN_FILL,FROZEN_STR,float(v[r,c])),
                        fill_opacity=1, stroke_color=BG, stroke_width=0.2)
            sq.move_to(RIGHT*c*cell+DOWN*r*cell)
            g.add(sq)
    g.add(Rectangle(width=cols*cell, height=rows*cell,
                    fill_opacity=0, stroke_color=FROZEN_STR, stroke_width=0.7).move_to(g))
    return g.move_to(ORIGIN)

def tokbars(n=24, w=1.4, h=0.24, seed=9):
    rng = np.random.default_rng(seed)
    vals = np.abs(rng.normal(0,1,n)); vals /= vals.max()
    bw = w/n - 0.007
    g = VGroup()
    for i,v in enumerate(vals):
        bar = Rectangle(width=bw, height=float(v)*h,
                        fill_color=ic(TRAIN_FILL,TRAIN_STR,float(v)),
                        fill_opacity=1, stroke_width=0)
        bar.align_to(ORIGIN,DOWN).shift(RIGHT*(i*(w/n)-w/2+bw/2))
        g.add(bar)
    g.add(Line(LEFT*w/2,RIGHT*w/2,stroke_color=RULE,stroke_width=0.5).align_to(g,DOWN))
    g.add(Rectangle(width=w+0.04,height=h+0.05,fill_opacity=0,
                    stroke_color=ic(TRAIN_FILL,TRAIN_STR,0.4),
                    stroke_width=0.55).align_to(g,DOWN).shift(DOWN*0.01))
    return g

def stag(txt, col):
    t = Tm(txt, sz=8, col=col)
    b = SurroundingRectangle(t, buff=0.04, corner_radius=0.03,
                             fill_color=BG, fill_opacity=1,
                             stroke_color=ManimColor(col), stroke_width=0.5)
    return VGroup(b,t)

def ar(s,e,col,sw=1.15,tl=0.12):
    return Arrow(s,e,buff=0,stroke_width=sw,tip_length=tl,
                 color=ManimColor(col),max_tip_length_to_length_ratio=0.3)

def elbHV(s,e,col,sw=1.15,tl=0.12):
    m = np.array([e[0],s[1],0])
    return VGroup(Line(s,m,stroke_width=sw,color=ManimColor(col)),
                  Arrow(m,e,buff=0,stroke_width=sw,tip_length=tl,
                        color=ManimColor(col),max_tip_length_to_length_ratio=0.3))

def elbVH(s,e,col,sw=1.15,tl=0.12):
    m = np.array([s[0],e[1],0])
    return VGroup(Line(s,m,stroke_width=sw,color=ManimColor(col)),
                  Arrow(m,e,buff=0,stroke_width=sw,tip_length=tl,
                        color=ManimColor(col),max_tip_length_to_length_ratio=0.3))

# =============================================================================
class DPVOMotionEncoderStatic(Scene):
    def construct(self):
        self.add(Rectangle(width=config.frame_width+2,height=config.frame_height+2,
                           fill_color=BG,fill_opacity=1,stroke_width=0))

        # ── Y layout ─────────────────────────────────────────────────────────
        Y_PREV  =  2.90   # prev stream centre
        Y_CURR  =  1.82   # curr stream centre
        # corr zone: centre at Y_CZ, height CZ_H
        Y_CZ    =  0.85   # corr zone centre
        CZ_H    =  1.55   # corr zone height  (contains corr + argmax only)
        Y_ASM   = -0.28   # token assembly    (outside corr zone, below)
        Y_BAND  = -1.05   # boundary band
        Y_HEAD  = -1.85   # trainable head row
        Y_OUT   = -2.90   # output row
        Y_MET   = -3.52   # metrics
        Y_CAP   = -3.88   # caption

        # ── TITLE ─────────────────────────────────────────────────────────────
        ti = T("DPVO Motion Encoder", sz=20, col=INK, w=BOLD).to_edge(UP,buff=0.20)
        su = Ts("Appearance-assisted 2-D displacement regression  ·  frozen trunk + trainable head",sz=10)
        su.next_to(ti,DOWN,buff=0.08)
        rl = Line(LEFT*6.5,RIGHT*6.5,stroke_color=INK,stroke_width=0.7).next_to(su,DOWN,buff=0.12)
        self.add(ti,su,rl)

        # ── INPUT FRAMES ──────────────────────────────────────────────────────
        def frame_w(col,lbl,y):
            o = RoundedRectangle(corner_radius=0.07,width=0.86,height=0.66,
                                 fill_color=ic(BG,col,0.07),fill_opacity=1,
                                 stroke_color=ManimColor(col),stroke_width=1.1)
            for i in range(3):
                o.add(Line(LEFT*0.31+UP*(0.15-i*0.11),RIGHT*0.31+UP*(0.15-i*0.11),
                           stroke_color=ic(BG,col,0.14+i*0.09),stroke_width=0.5))
            l = T(lbl,sz=9,col=col).next_to(o,DOWN,buff=0.05)
            return VGroup(o,l).move_to(RIGHT*-6.0+UP*y)

        fp = frame_w(WIRE_P,"frame  t−1",Y_PREV)
        fc = frame_w(WIRE_C,"frame  t", Y_CURR)
        cap_p = Tc("PREV STREAM",col=FROZEN_STR); cap_p.next_to(fp,UP,buff=0.06)
        cap_c = Tc("CURR STREAM",col=CORR_STR);   cap_c.next_to(fc,UP,buff=0.06)
        self.add(fp,fc,cap_p,cap_c)

        # ── PREPROCESS ────────────────────────────────────────────────────────
        pre = fblock("Preprocess","undo ImageNet  ·  ×2 − 0.5",w=2.0,h=0.58)
        pre.move_to(RIGHT*-4.15+UP*((Y_PREV+Y_CURR)/2))
        self.add(pre)
        self.add(ar(fp.get_right(),pre.get_top(),   col=WIRE_P))
        self.add(ar(fc.get_right(),pre.get_bottom(),col=WIRE_C))

        # ── FROZEN fnet ───────────────────────────────────────────────────────
        fnet = fblock("DPVO  fnet","BasicEncoder4  ·  stride 4  ·  128-D",w=2.45,h=0.72)
        fnet.move_to(RIGHT*-1.85+UP*((Y_PREV+Y_CURR)/2))
        cap_f = Tc("FROZEN  ·  ∇ = 0",col=FROZEN_STR); cap_f.next_to(fnet,UP,buff=0.07)
        self.add(fnet,cap_f)
        self.add(ar(pre.get_right(),fnet.get_left(),col=FROZEN_STR))

        # feature map thumbnails  — placed right of fnet, vertically centred on each lane
        fmp = hmap(5,6,0.086,seed=4, lo=FROZEN_FILL,hi=FROZEN_STR)
        fmc = hmap(5,6,0.086,seed=17,lo=ManimColor("#E8F5E9"),hi=CORR_STR)
        fmp.next_to(fnet,RIGHT,buff=0.48).move_to(RIGHT*fmp.get_center()[0]+UP*Y_PREV)
        fmc.next_to(fnet,RIGHT,buff=0.48).move_to(RIGHT*fmc.get_center()[0]+UP*Y_CURR)

        # ensure same x as fmp
        fmc.move_to(RIGHT*fmp.get_center()[0]+UP*Y_CURR)

        st_fp = stag("(B,128,H/4,W/4)",FROZEN_STR); st_fp.next_to(fmp,RIGHT,buff=0.07)
        st_fc = stag("(B,128,H/4,W/4)",CORR_STR);   st_fc.next_to(fmc,RIGHT,buff=0.07)
        self.add(fmp,fmc,st_fp,st_fc)
        self.add(ar(fnet.get_right(),fmp.get_left(),col=FROZEN_STR,tl=0.10))
        self.add(ar(fnet.get_right(),fmc.get_left(),col=CORR_STR,  tl=0.10))

        # ── PATCH SAMPLING ────────────────────────────────────────────────────
        psamp = fblock("Patch sample","8×8 grid  ·  64 patches from t−1",w=2.35,h=0.58)
        psamp.move_to(RIGHT*3.35+UP*Y_PREV)
        pv = patchvis(); pv.next_to(psamp,RIGHT,buff=0.16)
        st_ps = stag("(B,64,128)",FROZEN_STR); st_ps.next_to(pv,RIGHT,buff=0.07)
        self.add(psamp,pv,st_ps)
        self.add(ar(st_fp.get_right(),psamp.get_left(),col=FROZEN_STR,tl=0.10))
        self.add(ar(psamp.get_right(),pv.get_left(),   col=FROZEN_STR,tl=0.09))

        # ── CORRELATION ZONE  (contains ONLY corr block + argmax block) ───────
        cz_top = Y_CZ + CZ_H/2
        cz_bot = Y_CZ - CZ_H/2
        cz = RoundedRectangle(corner_radius=0.13,width=7.6,height=CZ_H,
                              fill_color=CORR_FILL,fill_opacity=1,
                              stroke_color=CORR_STR,stroke_width=0.85)
        cz.move_to(RIGHT*-0.5+UP*Y_CZ)
        cz_cap = Tc("CORRELATION ZONE  ·  PARAMETER-FREE",col=CORR_STR)
        cz_cap.next_to(cz,UP,buff=0.07)
        self.add(cz,cz_cap)

        # cosine correlation
        cb = fblock("Cosine correlation",
                    "L2-norm  ·  cosine ∈ [−1,1]  ·  local window ±32 cells",
                    w=3.1,h=0.62)
        cb[0].set_fill(BG,opacity=1)
        cb.move_to(RIGHT*-1.4+UP*(Y_CZ+0.37))
        cv = corrvol(); cv.next_to(cb,RIGHT,buff=0.20)
        st_cv = stag("(B,64,H,W)",CORR_STR); st_cv.next_to(cv,RIGHT,buff=0.07)
        self.add(cb,cv,st_cv)

        # argmax
        ab = fblock("Windowed soft-argmax",
                    "hard peak  →  7×7 window  ·  temperature 20",
                    w=3.1,h=0.62)
        ab[0].set_fill(BG,opacity=1)
        ab.move_to(RIGHT*-1.4+UP*(Y_CZ-0.37))
        av = argmaxvis(); av.next_to(ab,RIGHT,buff=0.20)
        st_av = stag("(Δx,Δy) + sharp",CORR_STR); st_av.next_to(av,RIGHT,buff=0.07)
        self.add(ab,av,st_av)
        self.add(ar(cb.get_bottom(),ab.get_top(),col=CORR_STR))

        # wires INTO corr zone from the two streams
        # prev patches → top of corr block (elbow: down then left)
        pts = pv.get_bottom()+DOWN*0.0
        self.add(elbHV(pts, cb.get_top(), col=FROZEN_STR))
        # curr fmap → bottom of corr block (elbow: down then left)
        self.add(elbHV(st_fc.get_bottom()+DOWN*0.0, cb.get_bottom(), col=WIRE_C))

        # ── TOKEN ASSEMBLY  (below corr zone, outside it) ─────────────────────
        asm = fblock("Token assembly",
                     "feat(128) ‖ Δx ‖ Δy ‖ ‖flow‖ ‖ sharp   →   132-D per patch",
                     w=4.5,h=0.62)
        asm.move_to(RIGHT*-0.5+UP*Y_ASM)
        st_asm = stag("(B,64,132)",FROZEN_STR); st_asm.next_to(asm,RIGHT,buff=0.10)
        self.add(asm,st_asm)
        # wire from argmax down to assembly
        self.add(ar(ab.get_bottom(),asm.get_top(),col=FROZEN_STR))

        # feat(128) bypass: shown as a small annotation on the left side of asm
        # (a full wire would cross the corr zone — the note is cleaner)
        fl = Tm("+ fnet feat(128) via descriptors",sz=8,col=FROZEN_STR)
        fl.next_to(asm, LEFT, buff=0.10)
        self.add(fl)

        # cacheable note
        cn = Ts("tokens cacheable — zero gradients above this line",sz=8,col=FROZEN_STR)
        cn.next_to(asm,DOWN,buff=0.06)
        self.add(cn)

        # ── BOUNDARY BAND ─────────────────────────────────────────────────────
        band = Rectangle(width=config.frame_width+2,height=0.36,
                         fill_color=BAND_COL,fill_opacity=1,stroke_width=0)
        band.move_to(UP*Y_BAND)
        rt = Line(LEFT*7,RIGHT*7,stroke_color=INK_MED,stroke_width=0.5)
        rb = Line(LEFT*7,RIGHT*7,stroke_color=INK_MED,stroke_width=0.5)
        rt.move_to(UP*(Y_BAND+0.18)); rb.move_to(UP*(Y_BAND-0.18))
        bl = Tc("FROZEN  /  PARAMETER-FREE",col=FROZEN_STR); bl.move_to(LEFT*3.0+UP*Y_BAND)
        br = Tc("TRAINABLE  ·  HEAD  ONLY", col=TRAIN_STR);  br.move_to(RIGHT*3.0+UP*Y_BAND)
        ba = Arrow(LEFT*0.38,RIGHT*0.38,stroke_width=0.9,tip_length=0.10,
                   color=INK_LIGHT,max_tip_length_to_length_ratio=0.3)
        ba.move_to(UP*Y_BAND)
        self.add(band,rt,rb,bl,br,ba)
        # crossing wire
        self.add(Line(asm.get_bottom(),asm.get_bottom()+DOWN*0.50,
                      stroke_width=1.15,color=FROZEN_STR))

        # ── TRAINABLE HEAD ────────────────────────────────────────────────────
        lin  = tblock("Linear  132→128","per-patch projection",w=2.3,h=0.62)
        pool = tblock("Attentive pool","1-query MHA  ·  64 patches",w=2.3,h=0.62)
        mlp  = tblock("MLP  128→256→128","motion embedding",w=2.45,h=0.62)
        lin .move_to(RIGHT*-3.55+UP*Y_HEAD)
        pool.move_to(RIGHT*0.0 +UP*Y_HEAD)
        mlp .move_to(RIGHT*3.5 +UP*Y_HEAD)
        self.add(lin,pool,mlp)

        # entry
        ep = asm.get_bottom()+DOWN*0.50
        self.add(ar(ep,lin.get_top(),col=TRAIN_STR))

        # lin→pool
        self.add(ar(lin.get_right(),pool.get_left(),col=TRAIN_STR))
        sl = stag("(B,64,128)",TRAIN_STR)
        sl.move_to((lin.get_right()+pool.get_left())/2+UP*0.17)
        self.add(sl)

        # pool→mlp
        self.add(ar(pool.get_right(),mlp.get_left(),col=TRAIN_STR))
        sm = stag("(B,128)",TRAIN_STR)
        sm.move_to((pool.get_right()+mlp.get_left())/2+UP*0.17)
        self.add(sm)

        # ── OUTPUT ROW ────────────────────────────────────────────────────────
        # token vis under MLP
        tok = tokbars(); tok.move_to(RIGHT*3.5+UP*Y_OUT)
        tl_ = Ts("128-D token",sz=8,col=TRAIN_STR); tl_.next_to(tok,RIGHT,buff=0.10)
        st_t = stag("(B,128)",TRAIN_STR); st_t.next_to(tok,UP,buff=0.06)
        self.add(tok,tl_,st_t)
        self.add(ar(mlp.get_bottom(),tok.get_top(),col=TRAIN_STR))

        # delta head
        dh = tblock("Delta head  ·  Linear 128→2",
                    "Huber loss  δ=0.1  ·  standardised targets",
                    w=2.75,h=0.62,fc=OUT_FILL,sc=OUT_STR)
        dh.move_to(RIGHT*-0.2+UP*Y_OUT)
        self.add(dh)
        self.add(ar(tok.get_left(),dh.get_right(),col=OUT_STR))

        # output
        ob = RoundedRectangle(corner_radius=0.07,width=1.50,height=0.52,
                              fill_color=ic(BG,OUT_STR,0.07),fill_opacity=1,
                              stroke_color=OUT_STR,stroke_width=1.5)
        ob.move_to(RIGHT*-3.8+UP*Y_OUT)
        ol = T("(Δx, Δy)  [m]",sz=13,col=OUT_STR,w=BOLD); ol.move_to(ob)
        self.add(ob,ol)
        self.add(ar(dh.get_left(),ob.get_right(),col=OUT_STR,sw=1.8))

        # ── METRICS ───────────────────────────────────────────────────────────
        rm = Line(LEFT*6.5,RIGHT*6.5,stroke_color=INK_MED,stroke_width=0.45)
        rm.move_to(UP*(Y_MET+0.30))
        self.add(rm)
        items = [("Δ-MAE","0.055 m"),("direction","7.9°"),
                 ("scale","0.997"),("bias","[+0.005, −0.008] m"),("DR drift","0.8–7.4 %")]
        step = 12.0/len(items)
        for i,(k,v) in enumerate(items):
            x = -6.0+step*(i+0.5)
            self.add(Ts(k,sz=8,col=INK_LIGHT).move_to(RIGHT*x+UP*(Y_MET+0.12)))
            self.add(T(v,sz=10,col=INK,w=BOLD).move_to(RIGHT*x+UP*(Y_MET-0.10)))
            if i<len(items)-1:
                sp = Line(UP*(Y_MET+0.26),UP*(Y_MET-0.20),
                          stroke_color=RULE,stroke_width=0.4)
                sp.move_to(RIGHT*(x+step/2)+UP*(Y_MET+0.03))
                self.add(sp)

        # ── CAPTION ───────────────────────────────────────────────────────────
        cap = Ts("Fig.  DPVO Motion Encoder.  Blue: frozen, ∇=0, tokens cached.  "
                 "Amber: trainable head.  Green: correlation zone.  "
                 "Metrics on Webots validation paths, stride=5 (~1 s).",
                 sz=8,col=INK_LIGHT).to_edge(DOWN,buff=0.18)
        rb2 = Line(LEFT*6.5,RIGHT*6.5,stroke_color=INK,stroke_width=0.7)
        rb2.next_to(cap,UP,buff=0.08)
        self.add(rb2,cap)

        # ── LEGEND ────────────────────────────────────────────────────────────
        litems = [(FROZEN_FILL,FROZEN_STR,"Frozen / parameter-free"),
                  (TRAIN_FILL, TRAIN_STR, "Trainable (head only)"),
                  (CORR_FILL,  CORR_STR,  "Correlation zone"),
                  (OUT_FILL,   OUT_STR,   "Output / delta head")]
        rows = VGroup()
        for f,s,l in litems:
            sq = Square(side_length=0.17,fill_color=ManimColor(f),fill_opacity=1,
                        stroke_color=ManimColor(s),stroke_width=0.9)
            lt = Ts(l,sz=8,col=INK_MED); lt.next_to(sq,RIGHT,buff=0.09)
            rows.add(VGroup(sq,lt))
        rows.arrange(DOWN,buff=0.12,aligned_edge=LEFT)
        rows.to_corner(UR,buff=0.22).shift(DOWN*0.55)
        lbg = SurroundingRectangle(rows,buff=0.12,corner_radius=0.06,
                                   fill_color=BG,fill_opacity=1,
                                   stroke_color=RULE,stroke_width=0.65)
        self.add(lbg,rows)

