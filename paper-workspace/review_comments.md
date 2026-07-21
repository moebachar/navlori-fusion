# 47 comment(s)


## Page 1

**[1] Highlight**
> WiFi (about 1 Hz) and inertial (about 30 Hz)

**Comment:** is it good to mention frequency values here ?

**[2] Highlight**
> either can go missing

**Comment:** non pro

**[3] Highlight**
> Section 4 reports the experiments, and Section 5 concludes.

**Comment:** update to the new sections structuration


## Page 2

**[4] Highlight**
> (∼1 Hz),

**Comment:** remove this value as it dont generelize

**[5] Highlight**
> baked

**Comment:** non pro

**[6] Highlight**
> and even the

**Comment:** non pro

**[7] Highlight**
> that the nearest competitors do not pair with the first two (Zhou et al., 2024; Horn et al., 2020).

**Comment:** it is so agressive saying competitor and position like this while we only want to show prometing work

**[8] Highlight**
> and the “video dropout” of Perceiver (Jaegle et al., 2021);

**Comment:** if not mentionned explecitly in upcoming work, just remove it


## Page 3

**[9] Highlight**
> 1 Hz

**Comment:** generelize this hard coded value

**[10] Highlight**
> asks for a

**Comment:** non pro

**[11] Highlight**
> The remainder of this section formalizes the localization problem (Section 3.2), develops the universal token and the two encoders (Section 3.3), and defines the fusion block and the position readout (Section 3.4).

**Comment:** you can remove this

**[12] Highlight**
> not on any

**Comment:** non pro

**[13] Highlight**
> which motivates a set encoder in the Deep Sets / Set Transformer family (Zaheer et al., 2017; Lee et al., 2019):

**Comment:** i dont like mentionning an external method in here

**[14] Highlight**
> that answers three questions: what was sensed, which sensor produced it, and when it was measured relative to the query,

**Comment:** non pro

**[15] Highlight**
> 1 Hz and 30 Hz

**Comment:** no hard coded frequencies in methodology

**[16] Highlight**
> a bank of n sinusoids

**Comment:** non pro


## Page 4

**[17] Highlight**
> The Qmotion branch shown for reference is a deferred extension.

**Comment:** what q motion you are talking about, we agreed to present only the query mode in this paper not the the decomposed

**[18] Highlight**
> (4)

**Comment:** this (4) is under the equation not next to it

**[19] Highlight**
> a slot index,

**Comment:** unclear term

**[20] Highlight**
> learned codebook

**Comment:** non pro

**[21] Highlight**
> (anchors),

**Comment:** replace this anchors term by something else in all the paper

**[22] Highlight**
> horizon

**Comment:** horizon ?!


## Page 5

**[23] Highlight**
> The encoder therefore reads a fixed window X ∈RT×F, with Xt,f the f-th inertial channel at step t, and processes it so that the final summary is invariant to the temporal location of a motion pattern. The first operation expresses this hypothesis directly: a learned bank of C1 temporal filters {k(c)}, each of support κ over the F input channels, is crosscorrelated with the window. Filter c at time t responds to the local stretch of motion around t, U(1) c,t = σ b(c) + κ−1 ∑ τ=0 F ∑ f=1 k(c) τ,f Xt+τ−⌊κ/2⌋, f , (7) where σ is a GELU nonlinearity (preceded by batch normalization) and the same filter weights are reused at every t, so the response is shift-equivariant along the window.

**Comment:** this is a so different explanation of 1d convolution, we wound the explanation to look like fondumentl reaserch but still keep it talking about convolution (edit the equation to)

**[24] Highlight**
> —

**Comment:** here and every were in the paper, using "--" is banned

**[25] Highlight**
> a temporal mean of the final feature map averages each channel’s response across all positions, ¯u = 1 T ′ T ′ ∑ t=1 U:,t ∈RC3. (8) Averaging over t is what makes the summary invariant to where in the window a motion pattern occurs:

**Comment:** make sure this is true from the source code

**[26] Highlight**
> stacks L pre-norm transformer

**Comment:** make sure this is real from the code, again we only do "query mode"


## Page 6

**[27] Highlight**
> rD

**Comment:** saying rD without mentionning what is r is ambegious

**[28] Highlight**
> a softmax over an all-(−∞) row would return not-a-number when every sensor token of a sample is absent, and the always-live CLS token removes that case.

**Comment:** no need to mention this

**[29] Highlight**
> the Qabsolute

**Comment:** there is one query so why do we annotate it as absolute ?

**[30] Highlight**
> MSILN

**Comment:** do we give citation for this ?

**[31] Highlight**
> not-a-number

**Comment:** non pro

**[32] Highlight**
> IMUWiFine

**Comment:** do we give citation for this ?

**[33] Highlight**
> a decomposed variant that estimates an absolute anchor and a gated motion correction separately (Qmotion in Figure 1) is left as a deferred extension.

**Comment:** no need for this


## Page 7

**[34] Highlight**
> external method

**Comment:** you dont call them external methods, that is a term for the code source not for paper

**[35] Highlight**
> Per-Modality

**Comment:** find better term than Per-... in all the paper

**[36] Highlight**
> is the harder case.

**Comment:** not the good expression

**[37] Highlight**
> by the harder remainder rather than a uniform deficit across the split.

**Comment:** ambiguous sentence

**[38] Highlight**
> Umeyama-aligned ATE,

**Comment:** we need to reference this

**[39] Highlight**
> baselines.

**Comment:** can we name them something else here rather than "basline"

**[40] Highlight**
> The architecture is described in Section 3.

**Comment:** why to start the section with this sentence ?


## Page 8

**[41] Highlight**
> Table 2: Per-modality encoder evaluation, each encoder scored on its field’s standard benchmark (MAE on UJIIndoorLoc, raw ATE on RoNIN canonical). Bold marks the lowest error in each row.

**Comment:** that delta is not a good sign for a metric

**[42] Highlight**
> simulated WiFi field is synthesized rather than measured, this sub-metre figure is optimistic; it is reported only as a controlled-lab check that fusion behaves sensibly when the WiFi anchor is clean, and is not comparable to the real-world results below.

**Comment:** its imoportant to say this but not phrase it in this way showing that we suck

**[43] Highlight**

**Comment:** this  ai "--" is forbiden

**[44] Highlight**
> while the IMUWiFine LSTM plateaus near 65 m and never fits the cross-session mapping.

**Comment:** no need to say this


## Page 9

**[45] Highlight**
> same model that plateaued near 52–65 m on MSILN tracks these paths to roughly one metre—and

**Comment:** no need for this explicit detail


## Page 10

**[46] Highlight**
> Aging

**Comment:** non pro

**[47] Highlight**
> The inertial encoder trails the reference on its own benchmark. On RoNIN canonical IMUCNN reports a raw ATE of 9.72 m against ResNet1D’s 5.14 m (+89.2%); removing a rigid-body offset (Umeyama alignment) narrows the gap to 7.62 m (+48.2%). Even with alignment the gap remains, and it reflects a 95×-smaller encoder paying a cross-subject generalization cost; closing it at this parameter budget is open work. The MSILN test split is not uniform. One path contributes 786 samples, roughly 28% of the test mass, and is WiFi-dense, so its WiFi-favourable composition shapes the aggregate test figure. The crosssession margin reported above holds against all three named baselines, but the dominance of this single path is the reason the test figure should be read alongside the validation figure, not in isolation.

**Comment:** there is a lot of redundunt bla bla here next to what has been said before

