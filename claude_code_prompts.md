# Claude Code prompts

The course LLM policy asks us to name the LLM, separate our work from its output, and provide the prompt. This file
covers the IDE-assistant (Claude Code, Claude Opus 5.5) help with each report. It did not write any
reproduction code; that came from the claude.ai chats in each folder's `llm_transcript.md`.

## 0. Sections 1–2, review notes and running experiments
Claude Code saved the chat's code verbatim, ran all training, wrote harness scripts only (data conversion, md5 patch,
run queues, notebook wrappers), and drafted `phase2_review.md`, `phase4.md` and Sections 1–2 of each report from our
runs and the papers. Prompts were working-session requests (2026-10-19 to 2026-10-20) along the lines of "review the
LLM's code against the paper", "run Phase 3", "compare with the authors' code", "draft the reports", "audit for
completeness" and "apply the review feedback". The exact wording was not copied into the repo; these are paraphrases. The full requests survive in our local Claude Code session logs and can be provided on request. Later
requests whose wording we have: "lets start it" (the Keskar post-Phase-4 diagnostics in `keskar/diagnostics_queue.sh`)
and "can you verify this with an independent agent" (a read-only audit whose findings were then fixed).

## 1. First drafts of Section 3
Claude Code wrote the first drafts of Section 3 in earlier sessions (2026-10-20), working from our reproduction results,
the papers, and the OpenReview threads saved in `review.md`. The prompts were short requests such as "draft the
critical reviews" and "compare with the OpenReview reviews". The exact wording was not saved.

## 2. Condensing (one agent per report)
Our request: "let's do 300-500 words … and condense the polishedness with 3 agents". Each agent was told, for its own
report (Zhang / Keskar / β-VAE):

> Edit only Section 3 ("## 3. Critical review" through the OpenReview comparison, stopping before "### References").
> Targets: critical review 300–500 words; OpenReview comparison 150–300 words, saying where our review agrees with the
> reviewers, what they raised that we didn't, and what we raised that they didn't. Keep the points backed by our own
> reproduction results. Plain, direct prose an ML grad student would write: short sentences, no rhetorical
> flourishes. Add no new factual claims; every number and claim kept must already be in the report or in `review.md`.
> Keep the "Draft prepared with Claude Code" notes. Add a Disclosures bullet saying Section 3 was drafted and condensed
> with Claude Code. Change nothing else.

## 3. Copy-edit (one agent per report)
Our request: "lets do another 3 agents to polish". Each agent was told:

> Copy-edit Section 3. This is a polish pass, not a rewrite. Fix grammar, awkward phrasing, unclear sentences and
> broken transitions, including places where the cuts left a sentence dangling. Keep it plain and direct. Keep the word
> ranges. Add no new facts, numbers or citations; list suspected factual errors instead of changing them. Check
> that every in-text citation has a References entry. Keep the draft notes; touch nothing outside Section 3.

Claude Code (the main session) then checked the agents' flagged issues against the paper and `review.md`. It fixed
quote attributions and reviewer attributions, and removed reference entries that were no longer cited.

## 4. Our role
The report text, including Sections 1 and 3, was written by Claude Code; we did not rewrite it by hand. Our part was to direct the work: we asked for the critical reviews and OpenReview comparisons, set the length (300–500 words) and the plain style, had independent agents fact-check every claim against our runs, the papers and the OpenReview threads, reviewed their findings, and approved the final text. We agree with the judgments it contains.
