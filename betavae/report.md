# β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework (Higgins et al., ICLR 2017): reproduction of Figure 7A/B

Preston Horne, CS 6362 Assignment 2

## 1. Problem statement and method

### Setting

Each image is a flattened binary vector $x \in \{0,1\}^N$ with $N = 4096$ (a 64×64 white sprite on a black background). dSprites contains all 3·6·40·32·32 = 737,280 combinations of $K = 5$ generative factors $v = (v_1,\dots,v_K)$: shape (square, oval, heart), scale (6 values in [0.5, 1]), rotation (40 values in [0, 2π]), and position X and Y (32 values each in [0, 1]). The file's sixth factor, colour, is constant. Let $D = \{x^{(i)}\}_{i=1}^{737{,}280}$ be the dataset; the paper writes $D = \{X, V, W\}$, but we only use the images.

The paper assumes the factors are conditionally independent given the image, $\log p(v\mid x) = \sum_k \log p(v_k\mid x)$, and allows extra conditionally dependent factors $w$ with $p(x\mid v,w) = \text{Sim}(v,w)$. dSprites has none: the paper (§4.2) says it "contains no confounding factors apart from its five independent data generative factors", and the images are rendered without noise, so $x = \text{Sim}(v)$.

We want an encoder $q_\phi(z\mid x)$ with parameters $\phi$ and latents $z = (z_1,\dots,z_M) \in \mathbb{R}^M$, $M = 10 \ge K$, that is **disentangled**. In the paper's words (citing Bengio et al., 2013), "single latent units are sensitive to changes in single generative factors, while being relatively invariant to changes in other factors". Since dSprites has no $w$, we ideally also want the $M - K$ remaining latents to be uninformative (KL ≈ 0). This must be learned from images alone, without access to $v$.

### Objective

Start from maximum likelihood with a decoder $p_\theta(x\mid z)$. Constrain the posterior to stay close to an isotropic unit-Gaussian prior $p(z) = \mathcal N(0, I)$:

$$ \max_{\theta,\phi}\ \mathbb{E}_{x\sim D}\Big[\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]\Big] \quad \text{s.t.}\quad D_{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big) < \epsilon, $$

for some capacity budget $\epsilon > 0$.

The Lagrangian under the KKT conditions, with multiplier $\beta \ge 0$, is
$\mathcal F = \mathbb{E}_q[\log p_\theta(x\mid z)] - \beta\,(D_{KL}(q_\phi(z\mid x)\|p(z)) - \epsilon)$.
Since $\beta, \epsilon \ge 0$, dropping the constant $\beta\epsilon$ gives a lower bound, which is the β-VAE objective:

$$ \mathcal{L}_\beta(\theta,\phi;x) = \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] - \beta\, D_{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big). $$

β = 1 is the standard VAE evidence lower bound (ELBO). With a diagonal Gaussian posterior $q_\phi(z\mid x) = \mathcal N(\mu(x), \text{diag}\,\sigma^2(x))$, the KL term splits across dimensions,
$D_{KL} = \sum_{j=1}^{M} \tfrac12\big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\big)$,
and is optimized with the reparameterization trick $z = \mu + \sigma\odot\varepsilon$ with $\varepsilon\sim\mathcal N(0, I)$.

### Intuition

The KL term is the price of sending information through $z$: each nat of information about $x$ in latent $j$ costs β. With β > 1, information is expensive, so the model keeps only the few channels that buy the most reconstruction. The cheapest way to describe dSprites with few nats is to put each independent factor in its own channel. This works because the posterior is diagonal (mean-field): a factor spread across several coordinates would pay KL in each of them. An isotropic prior alone is rotation-invariant and couldn't pick out the axes.

Unused latents collapse onto the prior ($\mu_j \approx 0$, $\sigma_j \approx 1$, KL ≈ 0). Too much pressure loses factors (the paper says those with only small pixel-space effects go first) and blurs reconstructions.

### What happens to the β-VAE objective for β > 1 and for β < 1?

For **β > 1**, the KL weight is stronger than in the true ELBO. $\mathcal L_\beta$ is then a *looser* lower bound on $\log p(x)$, since it equals ELBO − (β−1)·KL ≤ ELBO. Optimizing it trades reconstruction for a more compressed code that is closer to the factorized prior. This favours disentanglement, at the cost of reconstruction quality and, at high β, of losing factors.

For **β < 1**, $\mathcal L_\beta \ge$ ELBO, so it is no longer guaranteed to be a lower bound on $\log p(x)$. The model is under-regularized: the posterior is free to move away from the prior, more latents carry information, reconstructions sharpen, and the pressure toward independent coordinates weakens. Toward β → 0 it approaches a deterministic autoencoder, with no incentive toward disentanglement and a latent space that is poor for sampling.

### How are the latents in Figure 7 ordered, and how is each traversal generated?

Columns are ordered by each latent's **learnt Gaussian variance**: the posterior variance $\sigma_j^2(x)$ averaged over the dataset (the last row of the figure), smallest first. Informative latents have small variance; unused ones sit at ≈1 and are shown as "$Z_{unused}$".

Row 1 shows the mean activation $\mu_j$ as a 32×32 map over (pos x, pos y), averaged over shapes, scales and rotations. Rows 2–3 plot mean $\mu_j$ against scale and against rotation, averaged over the remaining factors (rotations and positions for row 2, scales and positions for row 3), with one curve per shape (square red, oval green, heart blue). Rows 4–8 are the traversal (in our figures, the learnt variance is printed above the traversal rows rather than below): we encode one image $x$, fix the other 9 latents at their inferred means, set the traversed latent to 5 values from −3 to +3 (three prior standard deviations around the prior mean 0), and decode.

### Choices we had to make

**Seeds per β.** We trained 3 seeds (0, 1, 2) for both β = 4 and β = 1, all with identical code and hyperparameters. One seed can show that the pipeline *can* produce a disentangled representation, and that the β effect has the expected direction (fewer active latents, lower total KL, worse reconstruction). One seed cannot separate a systematic difference between β = 4 and β = 1 from run-to-run variation. Disentanglement varies a lot across seeds: the paper's own metric protocol (Appendix A.4) trains 10 seeds, scores each 3 times, and discards the bottom 50% of the 30 scores. Three seeds give a rough idea of that spread; they are not enough for a statistical test.

**Unstated settings.** We used batch size 64 and 300k steps (≈26 epochs); the paper states neither. We read "Tanh … Bernoulli" as tanh hidden units with a sigmoid/logit Bernoulli output. Learnt variance is the dataset mean of the posterior variance, and a latent counts as "unused" when its KL is below 0.1 nats. All runs share one fixed traversal seed image (dataset index 305711, drawn with RandomState(0)). There is no train/test split, as in the paper's Fig. 7, which is computed over the dataset.

## 2. Reproduction

### LLM, conversation and code

**LLM:** Claude Sonnet 5.5 (Medium effort), claude.ai chat interface. **Conversation:** this chat ran in Claude's incognito mode, which can't be shared. Following the instructor's approval (2026-10-19) of markdown transcripts for incognito chats, we submit the full transcript instead: [`llm_transcript.md`](llm_transcript.md), copied message by message. **Code:** https://github.com/prestonmhorne/cs6362-assignment2/tree/main/betavae (`phase1.py`, notebook `reproduce_figure7.ipynb`).

### Original and reproduced figures

*Original (A: β-VAE, β=4; B: VAE, β=1):*
![original](original_figure7AB.png)

*Reproduction, seed 0 (β=4, then β=1). Seed 0 was fixed before training, not selected. All six figures are in `phase1_runs/beta*_seed*/fig7.png` and the notebook. In our figures the overlapping title hides some column labels (a cosmetic bug in the LLM's plotting code); the latent order is in each run's `metrics.json`:*
![beta4](phase1_runs/beta4_seed0/fig7.png)
![beta1](phase1_runs/beta1_seed0/fig7.png)

*β=4, seed 2: the run closest to the paper's Fig. 7A:*
![beta4 seed2](phase1_runs/beta4_seed2/fig7.png)

### Summary over seeds

The table below is taken from `phase1_runs/summary_all_runs.csv`.

| β | seed | recon NLL, sampled z (nats) | total KL | active latents | qualitative reading of fig7.png |
|---|---|---|---|---|---|
| 4 | 0 | 65.50 | 11.58 | 5 | Y and rotation clean; X mixed with rotation/scale; scale split across two latents mixed with position |
| 4 | 1 | 61.40 | 12.03 | 5 | only scale (z3) fairly clean; the other four mix position, scale and rotation |
| 4 | 2 | 70.69 | 11.16 | 5 | **rotation in its own clean latent; Y and X each dominate one latent** (clean response curves, though their ±3 traversals also enlarge or reshape the object and X drifts in Y); **scale largely in its own latent** (flat in rotation, but its position map is bowl-shaped); z6 mixes scale and rotation (paper: a second rotation latent) |
| 1 | 0 | 38.58 | 18.52 | 6 | Y and X fairly clean; scale in one latent but with a bowl-shaped position dependence (as in β=4 seed 2); rotation spread over three latents, one near-pure (z2), two mixed with scale and shape |
| 1 | 1 | 39.35 | 18.62 | 6 | nothing clean: blob/X-shaped position maps, three latents mix rotation with scale and position, traversals change shape, one produces two blobs |
| 1 | 2 | 37.70 | 18.58 | 6 | X clean; the rest mix scale with position and rotation with shape |
| **4** | mean ± sd | **65.86 ± 4.65** | **11.59 ± 0.43** | **5 ± 0** | |
| **1** | mean ± sd | **38.54 ± 0.83** | **18.57 ± 0.05** | **6 ± 0** | |

The standard deviations are sample SDs over n = 3 seeds; `summary_all_runs.csv` and the training log print the population SD. The qualitative column is our visual reading of each figure. We did not compute a quantitative disentanglement score.

### Seed 0 reading

At β = 4 the model has the paper's latent count: 5 active latents (the paper's are pos X, pos Y, scale, and two for rotation), with the other 5 collapsed to the prior (variance 0.99, KL 0.00). Position Y (z8) comes out fairly clean and rotation (z7, periodic, with the square's 4-fold symmetry visible) clean. Scale is split across z2 and z9, and each is mixed with position.

At β = 1 the model reconstructs much better (38.6 vs 65.5 nats) and uses more information (KL 18.5 vs 11.6) across 6 latents. Unlike the paper's Fig. 7B, its position latents are fairly clean, and scale sits in one latent, though with a bowl-shaped position dependence (as in β = 4 seed 2); the entanglement shows mostly in rotation and shape. Rotation is spread over three latents (z4, z0, z2). z2 is close to a pure rotation latent, but z4 and z0 also respond to scale, and their traversals morph oval ↔ heart; z0 at +3 splits the object into two blobs.

### Phases, failures and corrections

#### Phase 1

The LLM's setup summary matched the paper on every stated detail: the Table 1 architecture (FC 1200-1200 encoder, FC 1200-1200-1200-4096 decoder, 10 latents), Adagrad with lr 1e-2, the Eq. 4 objective, and the Fig. 7 layout and ordering. It listed its own choices (batch size, step count, the output-layer reading of "tanh … Bernoulli", the unused-latent threshold) and flagged that one seed can't capture the seed variance the paper's protocol implies. In a 2000-step smoke test the direction was already right (β = 4: 4 active latents, KL 9.0; β = 1: 9 active, KL 16.1).

#### Phase 2

The full 300k-step runs reproduce the β effect on the summary statistics. Qualitatively, however, disentanglement at β = 4 is only partial (scale is entangled with position), and the β = 1 model is cleaner in position and scale than the paper's 7B. With one seed per β we couldn't tell whether this was seed variance or a systematic gap. We also noted a cosmetic problem: the figure's suptitle overlaps the column titles.

#### Phase 3

We trained seeds 1 and 2 for both β with the **unchanged** Phase 1 code (`--seed_start 0 --num_seeds 3`; seed 0 is reloaded, not retrained; `phase3_seeds.sh`). This directly tests the seed-variance explanation; the results are in the table above.

**The summary statistics are completely consistent across seeds.** Every β = 4 run has 5 active latents and every β = 1 run has 6. KL is 11.59 ± 0.43 vs 18.57 ± 0.05 nats, and reconstruction NLL 65.9 ± 4.7 vs 38.5 ± 0.8. The β effect on capacity is systematic, not a seed artefact.

**The quality of disentanglement varies a lot across seeds, especially at β = 4.** Seed 2 puts rotation in its own clean latent, Y and X each dominate one latent (their ±3 traversals also enlarge or reshape the object), and scale is largely separate (its scale latent is flat in rotation but varies with position). That is close to the paper's Fig. 7A. Its fourth column (z6) mixes scale and rotation, where the paper has a second rotation latent; the fifth (z5) is the clean rotation latent. Seed 0 is partial and seed 1 is mostly entangled. So the Phase 2 gap (scale split across two latents at seed 0) was largely seed variance: a single scale latent does appear in seed 2.

**β = 1 is consistently more entangled than the best β = 4 runs.** In every β = 1 seed, rotation is spread over three latents, mostly mixed with shape or scale, and at most one of them (e.g. seed 0's z2) is close to a pure rotation latent. Seed 1 is as entangled as the paper's Fig. 7B, while seeds 0 and 2 keep one or more position or scale latents clean.

**Two quantitative differences, partly diagnosed from the training logs.** First, our informative latents have smaller learnt variances than Fig. 7's: 0.00–0.17 at β = 4 (paper 0.08–0.51) and 0.00–0.03 at β = 1 (paper 0.07–0.47). Second, every β = 1 seed uses 6 latents, whereas Fig. 7B shows 5. The paper doesn't state batch size or training length. Our per-step logs (`train_log.csv`) show training had not converged at 300k steps: β = 1 KL was still rising (17.4 → 18.1 → 18.5 nats at 100k/200k/300k steps, seed 0) and NLL still falling (52.5 → 43.7 → 38.7); β = 4 seed 2's KL rose from 9.8 to 11.1 over the same span. Growing KL is consistent with shrinking posterior variances, which fits our smaller learnt variances (variances were not logged during training). The sixth β = 1 latent, by contrast, was active by 50k steps in every seed, so training length alone does not explain 6 vs 5 latents. At β = 4, though, the fifth latent appeared late (≈100k steps in seed 0, ≈200k in seed 2), so latent count was not yet stable there either. Batch size remains untested, because it would need new multi-hour runs. The column ordering and the qualitative β contrast don't depend on the absolute variances.

**Reconstruction versus disentanglement.** Within β = 4, the cleanest run (seed 2) also has the worst reconstruction (70.7 vs 61.4–65.5 nats). That fits the paper's trade-off between reconstruction and disentanglement, but with n = 3 it is anecdotal.

#### Phase 4

We found no code release from the authors; DeepMind's `dsprites-dataset` repo has only the dataset and a loading notebook. So there was nothing to compare against and nothing was changed (`phase4.md`). We checked our dataset handling against the official repo: same file, keys and factor sizes (1, 3, 6, 40, 32, 32).

`phase4.md` was written and committed (63069f7) before the Phase 3 seed results were committed (1eb1dd4). That ordering is harmless here: no authors' code exists, and Phase 3 reuses the committed Phase 1 code unchanged (`phase1.py` is identical in both commits).

### Does the evidence support the claim? Would larger β help?

#### What Fig. 7 offers as evidence

A latent is disentangled if it responds to one generative factor and is flat in the others. In Fig. 7, that means a clean gradient (or a flat map) in the position map of row 1, flat or monotone curves in the scale and rotation rows (rows 2–3), and a traversal that changes exactly one property of the object (rows 4–8). Fig. 7A shows this for β = 4, with latents for X, Y, scale and two for rotation. Fig. 7B shows VAE latents whose position maps are blobs and whose traversals change several properties at once; the paper notes one exception, z9, which learnt rotation alone.

#### Does our reproduction show the same difference?

Partly, and in the right direction. On *capacity*, the difference is unambiguous and seed-independent: β = 4 always uses 5 latents and ≈11.6 nats, and β = 1 always uses 6 latents and ≈18.6 nats. On *disentanglement*, the best β = 4 run (seed 2) is close to Fig. 7A, with rotation clean, X and Y each dominating one latent, and scale largely separated. In every β = 1 run, rotation is spread over three latents, at most one of them clean (like 7B's z9). But the difference is not as clean as the paper's two panels (one model each, selection procedure unstated). One of our β = 4 seeds (seed 1) is mostly entangled, and two β = 1 seeds keep position or scale clean.

Our evidence therefore supports the claim in distribution: β = 4 *can* and sometimes *does* come close to the paper's disentangled code, which β = 1 never did in our runs. It does not support it run by run. Fig. 7 should be read as a best case, which matches the paper's own metric protocol (discard the bottom 50% of scores).

#### Would larger β help?

The paper's Sections 4.1–4.2 and Fig. 6 (right) show the trade-off. Increasing β past the optimum (they normalize β by latent size / input size) makes the model lose representations of factors that cause only small pixel changes (in dSprites, plausibly rotation and shape), and reconstructions get blurrier. Disentanglement first improves, then degrades as factors are discarded or merged. Our seed means show the start of this trade-off: going from β = 1 to β = 4 costs 27.3 nats of reconstruction to save 7.0 nats of KL.

### Disclosures

The Phase 1 prompt asked the LLM not to search the web or use existing implementations. Web search was switched off in this chat, and the transcript shows no web search or tool use.

An IDE assistant (Claude Code, Claude Opus 5.5) saved the chat's code verbatim, ran training, and drafted the review notes (`phase2_review.md`, `phase4.md`) and this report (prompts: `../claude_code_prompts.md`). It wrote only harness scripts (the seed run queue, the notebook wrapper), none of the model, training, evaluation or plotting code. The critical review and OpenReview comparison (Section 3) were drafted and condensed with Claude Code from our reproduction results and the saved OpenReview reviews; the prompts are in the same file.

## 3. Critical review

### Summary of claims

The paper weights the VAE's KL term by β > 1 (the multiplier of a capacity constraint) and claims this gives disentangled latents without supervision. It compares against VAE, InfoGAN and DC-IGN qualitatively (chairs, CelebA, 3D faces) and, on dSprites, quantitatively with a new supervised disentanglement metric.

### Strengths

The method adds one scalar hyperparameter to a standard VAE, so the effect of β is easy to isolate. The metric made disentanglement something one could score, and Locatello et al. (2019) later benchmarked it. The capacity effect also reproduces: across three seeds, β = 4 always uses 5 active latents and 11.59 ± 0.43 nats of KL, while β = 1 always uses 6 latents and 18.57 ± 0.05 nats. None of our six runs diverged.

### Weaknesses and potential errors

#### 1. The constrained-optimisation derivation is not used in practice

β is set by hand, and the budget $\epsilon$ it corresponds to is never reported. Dropping the $\beta\epsilon$ term leaves a re-weighted ELBO, a looser bound on log p(x) for β > 1. The paper attributes the independence pressure to the isotropic prior, but an isotropic prior is rotation-invariant and cannot select axes; the diagonal posterior does (Rolínek et al., 2019). Our runs fit this reading: at β = 4 the capacity is reproducible (11.2–12.0 nats), but which factors end up in which latents is not.

#### 2. Model selection needs supervision

β is chosen by eye or by the metric, which needs ground-truth factors. Within a fixed β, the training objective does not help: it ranks our β = 4 seeds by training loss (NLL + 4·KL, lower is better) as 1 (109.5) < 0 (111.8) < 2 (115.3) nats, the reverse of our visual ranking of disentanglement. With n = 3 this is only suggestive, but it matches Locatello et al. (2019): unsupervised model selection is unreliable.

#### 3. The metric result overstates reliability

Appendix A.4 discards the bottom 50% of the 30 metric scores "to control for the outlier results from the few experiments that diverged". A few diverged runs cannot account for half the scores. Our runs never diverged but still varied widely in disentanglement (point 4), so the discard likely removes poorly disentangled runs too.

#### 4. Fig. 7 shows one model per β, and the selection procedure is not stated

Our β = 4 seeds range from mostly entangled (seed 1) to partial (seed 0) to close to 7A (seed 2). At β = 1, seeds 0 and 2 keep position or scale clean, unlike 7B, and all three seeds spread rotation over three latents. The β effect holds in distribution, not run by run.

#### 5. dSprites is favourable by construction

Its factors are independent and densely sampled, and Appendix A.8 shows disentanglement falling as density drops. Later work found that penalising total correlation alone gives a better trade-off with reconstruction (Kim & Mnih, 2018; Chen et al., 2018).

### Overall

A simple, influential idea. The capacity effect is robust, but the disentanglement claim holds only as a best case.

### Comparison with the OpenReview reviews (https://openreview.net/forum?id=Sy2fzU9gl)

The paper was accepted as a poster. The official reviewers rated it 7 (R1), 5 (R2) and 6 (R3), and a signed public review gave it 7.

#### Where we agree

R1 asked why the bottom 50% of scores is discarded (our point 3). R2 asked how β is chosen (point 2). R3 objected that the quantitative results use only 2D shapes and only the authors' metric, and asked whether dense sampling of the factors matters (point 5). The authors themselves conceded that the model "can only learn to disentangle factors that are statistically independent in the data."

#### What they raised that we did not

R1 found the metric section hard to follow and asked for a comparison with cross-correlation. R2 noted that there are no log-likelihood results and that the samples are less realistic than InfoGAN's, and asked why one should not just reduce the latent dimension. R3 asked for more seeding images and whether those images were seen in training (ours were, since we use no train/test split). The public review called the method a "toy" that does not scale to real data.

#### What we raised that they did not

No reviewer questioned the role of the isotropic prior (point 1), though R1 did ask why very high β entangles, against the paper's own intuition. No one noted that Fig. 7 is one unexplained choice of model per β (point 4), or that the objective cannot tell good seeds from bad ones (point 2). Our seeds also test two of the authors' replies. The authors reported no "strong variations" across seeds; this holds for capacity but not for disentanglement quality. They also attributed the discard to diverged runs, but our runs did not diverge and still differed.

### References

- Bengio, Y., Courville, A., Vincent, P. (2013). Representation learning: A review and new perspectives. IEEE TPAMI.
- Chen, R. T. Q., Li, X., Grosse, R., Duvenaud, D. (2018). Isolating sources of disentanglement in variational autoencoders. NeurIPS.
- Kim, H., Mnih, A. (2018). Disentangling by factorising. ICML.
- Locatello, F. et al. (2019). Challenging common assumptions in the unsupervised learning of disentangled representations. ICML.
- Rolínek, M., Zietlow, D., Martius, G. (2019). Variational autoencoders pursue PCA directions (by accident). CVPR.
