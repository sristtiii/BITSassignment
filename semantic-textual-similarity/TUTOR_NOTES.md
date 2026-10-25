# Tutor Notes

## Project we are building
Semantic Textual Similarity with Transformers: predict how similar two sentences are (score 0-5) using the GLUE STS-B dataset.
Compare two models: `bert-base-uncased` and `sentence-transformers/all-MiniLM-L6-v2`.
Needs: load data, tokenize, load model, fine-tune/infer, evaluate, 5+ custom predictions, error analysis.

## Learned so far
- Starting point: basic Python, numpy, linear algebra, sigmoid, supervised learning, regression vs classification, 80/10/10 split. Has seen CBOW / skip-gram. Needs a recap of all of it.
- Tools: uses PyCharm, then moves to Google Colab for submission. Has never trained a model before.

- Gradient descent by hand: gradient 2(wx-y)x = -18, update w = w - lr*grad (2.18 at lr 0.01, 3.8 at lr 0.1), and understands that a large learning rate overshoots (loss 5.76 at w=3.8).

- Section 0 (setup): wrote the pip install cell (fixed typo and commas after hint), ran it in Colab on a T4. Learning what each library is for. Imports and GPU check next.

- Section 0 done: imports (torch, numpy, load_dataset) and GPU check run, GPU True on T4. Learned the `from X import Y` order after the SyntaxError hint. Understands fine-tuning = adapting a pretrained model; partly got "why not from scratch" (effort), needs the data and compute reason.

## Weak spots
- Has never trained a model (so training loops, loss, epochs are new).
- Hasn't used PyTorch / Hugging Face / pandas yet.
- Needs a refresher on ML basics and NLP basics.
- Confusion-matrix formulas: first attempt was wrong. After the digit trick the formulas are correct (precision tp/(tp+fp), recall tp/(tp+fn), accuracy (tp+tn)/all). Still slips on plugging in numbers: precision came out 40/70 instead of 40/50, so it used tn instead of fp. Check this again later.
- Training step: described data prep (standardize, split) instead of the loop (forward, loss, gradient, update). Exercise w=2, x=3, y=9 done: loss 9 correct, correctly said w should go up, recalled the update rule (w = w - lr * derivative). (The "-6" was a dash used as a label, not a sign slip. Tutor misread it.) Gradient attempt: got +18 instead of -18 (dropped the sign when computing 6-9), so both updates went the wrong way. Weak spot: signs with negatives, and sanity-checking an answer against earlier reasoning. Redo pending, plus the loss for lr=0.1.

## Open questions / things to check
- Assignment lists Accuracy/Precision/Recall/F1, but STS-B is regression (continuous score). Need to decide how to handle this (e.g. also report Pearson/Spearman, and/or bin scores into classes).

## Plan (5-day deadline, heavy workload, so go as far as possible each session)
Order matters because each chunk builds on the one before. Tick items off as we finish them.
- [x] 0. Warm-up: confusion-matrix formulas (done: precision 0.8, recall 40/60, accuracy 0.7)
- [ ] 1. Foundations recap: loss, training loop, embeddings, why BERT
- [ ] 2. Data: load STS-B, inspect it, tokenize
- [ ] 3. BERT fine-tune and evaluate (hardest chunk, on a Colab GPU)
- [ ] 4. MiniLM, same pipeline
- [ ] 5. Metrics: regression vs classification issue, then compare the models
- [ ] 6. 5+ custom examples and error analysis
- [ ] 7. Write-up and notebook cleanup
If we run short: cut write-up polish first. Keep the pipeline and metrics.
Rule: save the Colab notebook often. Use a GPU runtime.

## Files
- gradient_descent_warmup.ipynb: deleted at the user's request. The warm-up loop was dropped.
- STSB_assignment_template.ipynb: assignment outline. Text sections only, all code cells empty. User writes all the code themselves. Run it in Colab with a GPU.

## How the user wants to work
- Write the code by hand. Tutor gives explanations, hints and reviews, not code to paste.

## Progress log
- Session 1: read the assignment, set up these notes.
- Session 1: user answered the background questions. Deadline is in 5 days and their other workload is heavy. Waiting on a decision about pacing.
