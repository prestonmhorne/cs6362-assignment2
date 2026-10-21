# Phase 1 prompt — Keskar et al. 2017, Figure 3 (panels F1, C1, C2)

Paste into a NEW chat in Claude.ai or ChatGPT. Attach `assignment2/papers/keskar2017_large_batch.pdf`.
Turn OFF web search / browsing for this chat (assignment rule: no existing code in Phases 1–3).
Record: model name + version shown in the UI, and the share link when done.

---

I've attached the paper "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (Keskar et al., ICLR 2017). I want to reproduce Figure 3, panels F1, C1, and C2, from scratch in PyTorch. Please do not search the web or use any existing implementation — work only from the paper.

1. First, summarize the exact setup the paper specifies for these panels: which network and dataset each panel uses (check the network table), the layer-by-layer architecture, the small-batch and large-batch sizes, the optimizer and its hyperparameters, the number of epochs, preprocessing, the range of α, and what is plotted on each axis. Point out anything the paper leaves unspecified.
2. Then write a complete, runnable PyTorch implementation (a single Python script or notebook cells) that trains the small-batch and large-batch solutions for each panel, evaluates train and test cross-entropy loss (and accuracy, if the figure shows it) along the 1-D linear interpolation between the two solutions, and plots the three panels in the style of the paper's figure.
3. Be explicit about two choices and justify them: how batch-norm statistics are computed at the interpolated weights (if the networks use batch norm), and whether the two runs share an initialization.
4. I'm running on an Apple M4 Pro laptop (PyTorch MPS backend, 24 GB RAM), so tell me the expected runtime and clearly list every deviation from the paper.

Do not hardcode or fabricate any values in the plots; every number must come from actual training runs logged to disk.
