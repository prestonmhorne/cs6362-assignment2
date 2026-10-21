# Phase 1 prompt — Zhang et al. 2017, Figure 1 (a, b, c)

Paste into a NEW chat in Claude.ai or ChatGPT. Attach `assignment2/papers/zhang2017_rethinking_generalization.pdf`.
Turn OFF web search / browsing for this chat (assignment rule: no existing code in Phases 1–3).
Record: model name + version shown in the UI, and the share link when done.

---

I've attached the paper "Understanding deep learning requires rethinking generalization" (Zhang et al., ICLR 2017). I want to reproduce Figure 1 (panels a, b, and c) from scratch in PyTorch. Please do not search the web or use any existing implementation — work only from the paper.

1. First, summarize the exact experimental setup the paper specifies for Figure 1: dataset preprocessing, architectures (small Inception, AlexNet, MLP 1x512), optimizer, learning-rate schedule, batch size, number of steps, and how each data modification (true labels, random labels, shuffled pixels, random pixels, Gaussian, partially corrupted labels with probability p) is constructed. Point out anything the paper leaves unspecified.
2. Then write a complete, runnable PyTorch implementation (a single Python script or notebook cells) that produces all three panels:
   - (a) training loss vs. thousand steps for the Inception model under the five settings,
   - (b) time-to-overfit (relative convergence time) vs. label corruption for the three architectures,
   - (c) test error vs. label corruption for the three architectures.
3. I'm running on an Apple M4 Pro laptop (PyTorch MPS backend, 24 GB RAM), so tell me the expected runtime and, if needed, propose a reduced grid (fewer corruption levels or architectures) — but clearly list every deviation from the paper.

Do not hardcode or fabricate any values in the plots; every number must come from actual training runs logged to disk.
