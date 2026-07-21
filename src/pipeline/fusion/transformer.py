"""FusionTransformer — Stage B+C collapsed into a single set-transformer.

One token type, one transformer. The three attention mechanisms are not
three modules — they are three things this one transformer does as the
token set it is fed grows:

* **self-attention (cross-modal)** — the modality tokens of one instant
  attend to each other → fusion of WiFi / IMU / Odom / Vision.
* **self-attention (temporal)** — once the token set spans K instants the
  *same* encoder layers also attend across time. "Temporal attention" is
  not new code; it is self-attention the moment the set has >1 instant.
* **cross-attention (readout)** — a learned ``PositionQuery`` (optionally
  parameterised by a query time τ) attends to the contextualised token
  set. Query ∉ set, so this one is genuinely cross-attention.

The universal token::

    token = encoder_embedding        (what was sensed)
          + modality_embedding       (which sensor)
          + time_encoding(Δt)        (when, relative to the query time)

A boolean padding mask marks every absent token slot (missing modality,
dropped modality, or missing instant). Robustness to missing modalities
is therefore a property of *training* (modality dropout + the mask), not
of the architecture.

Input convention (carries unchanged from Iteration 1 → 2):
    inputs[mod] : (B, K, *window)   raw per-modality window, K instants
    avail[mod]  : (B, K) bool       True = observation present
    dt[mod]     : (B, K) float      seconds, Δt = t_obs − t_query (≤ 0 = past)

K = 1 recovers single-instant fusion (Iteration 1).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ContinuousTimeEncoding(nn.Module):
    """Maps a token's time signature to a ``dim``-vector.

    Default mode (``"learned_continuous"``) is where mTAN's continuous-time
    idea lives — folded into a token feature instead of a separate alignment
    stage. A bank of sinusoids at geometrically-spaced periods makes nearby
    times produce similar codes while keeping distant times distinguishable;
    a linear layer mixes them into the embedding space.

    Modes (M1 ablation, reviewer-mandated):
      - ``"learned_continuous"`` — current default, real-valued Δt.
      - ``"none"``               — no time encoding (returns zeros).
      - ``"binned"``             — bucketize ``|Δt|`` log-uniformly into
                                    ``n_bins`` buckets; learned embedding per
                                    bucket (the AFT-VO style this paper criticises).
      - ``"posindex"``           — sinusoidal positional encoding by within-
                                    modality arrival rank (0..K-1) — discards
                                    the real Δt magnitude and keeps only ordering.
                                    Caller must pass integer-valued ``dt`` (rank).
    """

    def __init__(self, dim: int, n_freqs: int = 32, min_period: float = 0.05,
                 max_period: float = 120.0, mode: str = "learned_continuous",
                 n_bins: int = 16):
        super().__init__()
        if mode not in {"learned_continuous", "none", "binned", "posindex"}:
            raise ValueError(f"unknown time_enc mode {mode!r}")
        self.mode = mode
        self.dim = dim
        if mode in ("learned_continuous", "posindex"):
            periods = torch.logspace(
                math.log10(min_period), math.log10(max_period), n_freqs,
            )
            self.register_buffer("omega", 2.0 * math.pi / periods)  # (n_freqs,)
            self.proj = nn.Linear(2 * n_freqs, dim)
        elif mode == "binned":
            edges = torch.logspace(
                math.log10(min_period), math.log10(max_period), n_bins - 1,
            )
            self.register_buffer("edges", edges)
            self.bin_emb = nn.Embedding(n_bins, dim)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """dt: (...) → (..., dim).

        In ``posindex`` mode, ``dt`` is interpreted as integer rank within
        the K-instant window (caller's responsibility — see ``encode_tokens``).
        """
        if self.mode == "none":
            return torch.zeros(*dt.shape, self.dim,
                               device=dt.device, dtype=torch.float32)
        if self.mode == "binned":
            bin_idx = torch.bucketize(dt.abs(), self.edges)
            return self.bin_emb(bin_idx)
        ang = dt.unsqueeze(-1) * self.omega          # (..., n_freqs)
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.proj(feat)


class FusionTransformer(nn.Module):
    """Set-transformer over (modality × time) tokens → (x, y).

    Parameters
    ----------
    encoders : dict[str, nn.Module]
        Per-modality encoder. Each maps ``(B, *window) → (B, embed_dim)``.
        Modality order is fixed by dict insertion order.
    embed_dim : int
        Shared token dimension (default 128).
    depth : int
        Number of self-attention encoder layers.
    n_heads : int
        Attention heads.
    ff_mult : int
        Feed-forward width multiplier.
    dropout : float
        Dropout inside attention / FFN / head.
    use_time : bool
        Add continuous-time encoding to every token (Iteration 2+).
    readout : {"cls", "query"}
        ``"cls"`` — a learned CLS token pooled by self-attention
        (Iterations 1–2). ``"query"`` — a separate ``PositionQuery`` that
        cross-attends to the token set (Iteration 3).
    """

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        embed_dim: int = 128,
        depth: int = 4,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_time: bool = True,
        readout: str = "cls",
        absolute_modalities: set[str] | list[str] | None = None,
        time_enc_mode: str = "learned_continuous",
        time_min_period: float = 0.05,
        time_max_period: float = 120.0,
    ):
        super().__init__()
        if readout not in {"cls", "query", "decomposed"}:
            raise ValueError(
                f"readout must be 'cls', 'query' or 'decomposed', got {readout!r}")
        self.modalities = list(encoders.keys())
        self.encoders = nn.ModuleDict(encoders)
        self.embed_dim = embed_dim
        self.use_time = use_time
        self.time_enc_mode = time_enc_mode
        self.readout = readout
        # Which modalities carry an ABSOLUTE position signal (place
        # recognition) vs RELATIVE motion. Used only by the decomposed
        # readout: absolute tokens feed the anchor head, relative tokens
        # feed the motion-correction head. Default: WiFi is the only
        # absolute modality (place-recognition vision was removed).
        default_abs = {"wifi"}
        self.absolute_modalities = (set(absolute_modalities)
                                    if absolute_modalities is not None
                                    else default_abs)

        self.modality_emb = nn.Parameter(
            torch.randn(len(self.modalities), embed_dim) * 0.02
        )
        self.time_enc = (
            ContinuousTimeEncoding(
                embed_dim, mode=time_enc_mode,
                min_period=time_min_period, max_period=time_max_period,
            )
            if use_time else None
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * ff_mult, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Readout — CLS (self-attention pooling) or cross-attention query.
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        if readout == "query":
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, n_heads, dropout=dropout, batch_first=True,
            )
            self.query_norm = nn.LayerNorm(embed_dim)

        if readout == "decomposed":
            # Two queries over two token pools: an anchor (absolute) and a
            # motion correction (relative). pred = p_abs + g * delta.
            self.anchor_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.motion_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.anchor_attn = nn.MultiheadAttention(
                embed_dim, n_heads, dropout=dropout, batch_first=True)
            self.motion_attn = nn.MultiheadAttention(
                embed_dim, n_heads, dropout=dropout, batch_first=True)
            self.anchor_norm = nn.LayerNorm(embed_dim)
            self.motion_norm = nn.LayerNorm(embed_dim)
            self.anchor_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(embed_dim, 2))
            self.motion_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(embed_dim, 2))
            # Adaptive correction gate g in [0, 1] from the motion feature.
            self.gate = nn.Linear(embed_dim, 1)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
        )

    # ------------------------------------------------------------------
    def _token_type_masks(self, M: int, K: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Boolean masks over the [CLS, M*K] key axis.

        Returns ``(is_abs, is_rel)`` each shape ``(1 + M*K,)`` where True
        marks a key the corresponding query is ALLOWED to attend to. CLS
        (index 0) is True in both so neither cross-attention can ever see an
        all-masked row (NaN safety). Tokens are grouped by modality, K per
        modality, in ``self.modalities`` order — matching ``encode_tokens``.
        """
        is_abs = torch.zeros(1 + M * K, dtype=torch.bool, device=device)
        is_rel = torch.zeros(1 + M * K, dtype=torch.bool, device=device)
        is_abs[0] = True
        is_rel[0] = True
        for mi, mod in enumerate(self.modalities):
            sl = slice(1 + mi * K, 1 + (mi + 1) * K)
            if mod in self.absolute_modalities:
                is_abs[sl] = True
            else:
                is_rel[sl] = True
        return is_abs, is_rel

    # ------------------------------------------------------------------
    def encode_tokens(
        self,
        inputs: dict[str, torch.Tensor],
        avail: dict[str, torch.Tensor],
        dt: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run per-modality encoders and assemble the universal token set.

        Returns
        -------
        tokens : (B, K*M, embed_dim)
        pad    : (B, K*M) bool — True = ignore (absent / dropped)
        """
        toks: list[torch.Tensor] = []
        pads: list[torch.Tensor] = []
        for mi, mod in enumerate(self.modalities):
            x = inputs[mod]                       # (B, K, *window)
            B, K = x.shape[0], x.shape[1]
            z = self.encoders[mod](x.flatten(0, 1))   # (B*K, embed_dim)
            z = z.view(B, K, self.embed_dim)
            z = z + self.modality_emb[mi]
            if self.use_time and dt is not None:
                if self.time_enc_mode == "posindex":
                    rank = torch.arange(K, device=z.device, dtype=z.dtype)
                    rank = rank.unsqueeze(0).expand(B, K)
                    z = z + self.time_enc(rank)
                else:
                    z = z + self.time_enc(dt[mod])
            toks.append(z)
            pads.append(~avail[mod])              # (B, K)
        tokens = torch.cat(toks, dim=1)           # (B, K*M, embed_dim)
        pad = torch.cat(pads, dim=1)              # (B, K*M)
        return tokens, pad

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        avail: dict[str, torch.Tensor],
        dt: dict[str, torch.Tensor] | None = None,
        query_dt: torch.Tensor | None = None,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Predict position.

        Parameters
        ----------
        inputs, avail, dt
            Per-modality token inputs / availability / Δt (see module doc).
        query_dt : (B,) or None
            Only used with the query / decomposed readouts: the time τ at
            which to predict position, as Δt from the anchor instant.
            ``None`` / zeros → predict at the anchor instant.
        return_parts : bool
            With ``readout="decomposed"``, also return the intermediate
            ``{p_abs, delta, gate}`` so the trainer can apply the auxiliary
            anchor loss. Ignored for other readouts.

        Returns
        -------
        (B, 2) predicted (x, y); or ``(pred, parts)`` if ``return_parts``.
        """
        tokens, pad = self.encode_tokens(inputs, avail, dt)
        B = tokens.shape[0]

        # Prepend CLS — it is never padded, so every row has ≥1 live token
        # and attention softmax can never see an all-(-inf) row → no NaN.
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)               # (B, 1+K*M, D)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_pad = torch.cat([cls_pad, pad], dim=1)       # (B, 1+K*M)

        x = self.encoder(x, src_key_padding_mask=full_pad)

        if self.readout == "decomposed":
            pred, parts = self._readout_decomposed(x, full_pad, query_dt, tokens)
            return (pred, parts) if return_parts else pred

        if self.readout == "cls":
            pooled = x[:, 0]                              # CLS
        else:
            # Cross-attention readout: PositionQuery attends to the whole
            # contextualised set, CLS included. Keeping CLS in the keys
            # guarantees every row has ≥1 unpadded key — so a sample whose
            # only live modality happens to have an empty window can never
            # produce an all-(-inf) softmax row (→ NaN).
            q = self.query.expand(B, -1, -1).clone()
            if (self.use_time and query_dt is not None
                    and self.time_enc_mode not in {"posindex", "none"}):
                q = q + self.time_enc(query_dt).unsqueeze(1)
            attn_out, _ = self.cross_attn(
                q, x, x, key_padding_mask=full_pad,
            )
            pooled = self.query_norm(attn_out.squeeze(1) + q.squeeze(1))

        return self.head(self.norm(pooled))

    # ------------------------------------------------------------------
    def _readout_decomposed(self, x, full_pad, query_dt, tokens):
        """pred = p_abs + g * delta.

        ``p_abs`` is read from the absolute (WiFi) tokens, ``delta`` from the
        relative (motion) tokens, and ``g`` is an adaptive correction gate.
        Each query keeps CLS in its key set (NaN safety). Returns
        ``(pred, {p_abs, delta, gate})``.
        """
        B = x.shape[0]
        M = len(self.modalities)
        K = (x.shape[1] - 1) // M
        is_abs, is_rel = self._token_type_masks(M, K, x.device)
        # key_padding_mask: True = ignore. Ignore padded tokens AND tokens of
        # the wrong type for this query.
        abs_pad = full_pad | (~is_abs).unsqueeze(0)
        rel_pad = full_pad | (~is_rel).unsqueeze(0)

        # Anchor (absolute) path — no time conditioning; it estimates the
        # current absolute position from place-recognition tokens.
        qa = self.anchor_query.expand(B, -1, -1)
        a_out, _ = self.anchor_attn(qa, x, x, key_padding_mask=abs_pad)
        a_feat = self.anchor_norm(a_out.squeeze(1) + qa.squeeze(1))
        p_abs = self.anchor_head(a_feat)

        # Motion (relative) path — τ-conditioned; estimates the correction.
        qm = self.motion_query.expand(B, -1, -1).clone()
        if (self.use_time and query_dt is not None
                and self.time_enc_mode not in {"posindex", "none"}):
            qm = qm + self.time_enc(query_dt).unsqueeze(1)
        m_out, _ = self.motion_attn(qm, x, x, key_padding_mask=rel_pad)
        m_feat = self.motion_norm(m_out.squeeze(1) + qm.squeeze(1))
        delta = self.motion_head(m_feat)

        gate = torch.sigmoid(self.gate(m_feat))           # (B, 1)
        pred = p_abs + gate * delta
        return pred, {"p_abs": p_abs, "delta": delta, "gate": gate}

    @torch.no_grad()
    def forward_attribution(
        self,
        inputs: dict[str, torch.Tensor],
        avail: dict[str, torch.Tensor],
        dt: dict[str, torch.Tensor] | None = None,
        query_dt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Predict, and report how much each modality drove the prediction.

        Only meaningful with ``readout="query"`` — the cross-attention
        readout has explicit query→key weights we can read. The ``"cls"``
        readout pools through self-attention with no single attributable
        layer, so this raises for it.

        The PositionQuery attends over ``[CLS, tokens]`` where ``tokens``
        are grouped by modality, K consecutive per modality (the layout
        built in :meth:`encode_tokens`). We sum the softmax attention mass
        over each modality's K tokens. Padded (absent / stale) tokens get
        ~zero weight automatically, so the percentages reflect what the
        model *actually used* for this specific prediction.

        Returns
        -------
        pred : (B, 2)
        attribution : dict with
            ``modalities`` : list[str]            — column order
            ``attn``       : (B, M) float          — attention mass per modality
            ``cls``        : (B,)  float           — attention mass on CLS
            ``avail_frac`` : (B, M) float          — fraction of K instants
                                                     present per modality
        (``attn`` columns + ``cls`` sum to 1 per row.)
        """
        if self.readout not in {"query", "decomposed"}:
            raise ValueError(
                "forward_attribution requires readout='query' or "
                f"'decomposed'; this model uses readout={self.readout!r}. "
                "The 'cls' readout pools through self-attention with no "
                "single attributable layer.")

        tokens, pad = self.encode_tokens(inputs, avail, dt)
        B = tokens.shape[0]
        M = len(self.modalities)
        K = tokens.shape[1] // M

        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_pad = torch.cat([cls_pad, pad], dim=1)
        x = self.encoder(x, src_key_padding_mask=full_pad)

        avail_frac = torch.stack(
            [avail[m].float().mean(dim=1) for m in self.modalities], dim=1
        )                                           # (B, M)

        if self.readout == "decomposed":
            return self._attribution_decomposed(x, full_pad, query_dt,
                                                 M, K, avail_frac)

        q = self.query.expand(B, -1, -1).clone()
        if self.use_time and query_dt is not None:
            q = q + self.time_enc(query_dt).unsqueeze(1)
        attn_out, attn_w = self.cross_attn(
            q, x, x, key_padding_mask=full_pad,
            need_weights=True, average_attn_weights=True,
        )
        # attn_w: (B, 1, 1 + M*K) — query→key weights averaged over heads.
        w = attn_w.squeeze(1)                       # (B, 1 + M*K)
        cls_mass = w[:, 0]                          # (B,)
        per_mod = torch.empty(B, M, device=w.device)
        for mi in range(M):
            start = 1 + mi * K
            per_mod[:, mi] = w[:, start:start + K].sum(dim=1)

        pooled = self.query_norm(attn_out.squeeze(1) + q.squeeze(1))
        pred = self.head(self.norm(pooled))

        return pred, {
            "modalities": list(self.modalities),
            "attn": per_mod,
            "cls": cls_mass,
            "avail_frac": avail_frac,
        }

    @torch.no_grad()
    def _attribution_decomposed(self, x, full_pad, query_dt, M, K, avail_frac):
        """Attribution for the decomposed readout.

        Reports per-modality attention from whichever query owns it (anchor
        query for absolute modalities, motion query for relative ones), the
        adaptive gate ``g``, and ``motion_frac`` = how much the gated motion
        correction contributed to the final prediction magnitude. The last
        is the key 'is motion actually being used?' diagnostic.
        """
        B = x.shape[0]
        is_abs, is_rel = self._token_type_masks(M, K, x.device)
        abs_pad = full_pad | (~is_abs).unsqueeze(0)
        rel_pad = full_pad | (~is_rel).unsqueeze(0)

        qa = self.anchor_query.expand(B, -1, -1)
        a_out, wa = self.anchor_attn(qa, x, x, key_padding_mask=abs_pad,
                                     need_weights=True, average_attn_weights=True)
        a_feat = self.anchor_norm(a_out.squeeze(1) + qa.squeeze(1))
        p_abs = self.anchor_head(a_feat)

        qm = self.motion_query.expand(B, -1, -1).clone()
        if (self.use_time and query_dt is not None
                and self.time_enc_mode not in {"posindex", "none"}):
            qm = qm + self.time_enc(query_dt).unsqueeze(1)
        m_out, wm = self.motion_attn(qm, x, x, key_padding_mask=rel_pad,
                                     need_weights=True, average_attn_weights=True)
        m_feat = self.motion_norm(m_out.squeeze(1) + qm.squeeze(1))
        delta = self.motion_head(m_feat)
        gate = torch.sigmoid(self.gate(m_feat))            # (B, 1)
        pred = p_abs + gate * delta

        wa = wa.squeeze(1)                                  # (B, 1+M*K)
        wm = wm.squeeze(1)
        per_mod = torch.zeros(B, M, device=x.device)
        for mi, mod in enumerate(self.modalities):
            start = 1 + mi * K
            src = wa if mod in self.absolute_modalities else wm
            per_mod[:, mi] = src[:, start:start + K].sum(dim=1)
        cls_mass = 0.5 * (wa[:, 0] + wm[:, 0])              # mean CLS use

        gm = (gate * delta).norm(dim=1)                     # (B,)
        am = p_abs.norm(dim=1) + 1e-6
        motion_frac = gm / (am + gm)                        # (B,)

        return pred, {
            "modalities": list(self.modalities),
            "attn": per_mod,
            "cls": cls_mass,
            "avail_frac": avail_frac,
            "gate": gate.squeeze(1),
            "motion_frac": motion_frac,
        }
