# Interview and resume notes

## Safe resume wording

“Built PixelDebug, a visual regression tool using OpenCV/SSIM, FastAPI, React/TypeScript, and Docker; implemented screenshot localization, interactive overlays, structured reports, browser-generated defect fixtures, and automated tests. Implemented an experimental SigLIP2-to-Gemma projection bridge in PyTorch and verified gradient flow, training mechanics, checkpoint reloads, and deterministic inference using tiny randomly initialized models.”

Do not claim “trained a multimodal defect diagnosis model,” “production AI inference,” semantic defect accuracy, or performance improvements over a baseline. Gemma access was unavailable, and task-trained diagnosis is unvalidated. Describe the 77.8% number only as target-box recall on 36 synthetic held-out cases, alongside fragmentation and dataset limitations.

## Explain the design

- **Why CV first?** It provides deterministic, cheap visual evidence and an honest baseline while model training is incomplete. SSIM plus pixel differences is less sensitive than pure pixel subtraction to small rendering noise, but still not a semantic detector.
- **How do models interact?** SigLIP2 patch features are pooled, combined into reference/current/difference sequences, and projected into Gemma's embedding dimension. Gemma attends to the prefix and JSON prompt; the masked causal loss updates the bridge through the frozen decoder.
- **Why isn't a linear connector enough?** Matching dimensions only makes the forward pass valid. Training must align the embedding spaces and teach the output task. Random projector outputs cannot produce reliable diagnoses.
- **How is data generated?** Playwright renders template variants and controlled CSS mutations; DOM element unions provide labels. Structural layout families remain in separate splits. This reduces sibling-template leakage, but three simple families are too limited to establish generalization.
- **What is evaluated?** CV target recall and box IoU; model evaluation supports schema validity, one-to-one detection precision/recall, and category-correct localization. Tiny-model loss decrease establishes optimization mechanics only.
- **How is inference served?** Working CV inference runs in a thread pool behind FastAPI, with bounded input bodies and a concurrency semaphore. nginx serves the compiled UI and proxies API requests. The experimental model is offline only; `/predict` explicitly returns 503.
- **How is it deployed?** Nonroot Python and nginx containers with health checks, loopback publication, dependency locks, and a CI workflow. This is a local deployment configuration, not a hosted authenticated SaaS.
- **What are the tradeoffs?** No silent resizing, alignment opt-in, heuristics labeled honestly, frozen models to reduce training cost, pooled visual tokens at the cost of spatial precision, and separate architecture evidence from quality evidence.

## Strong next experiments

Use more independent layouts and real screenshot pairs. Add identical and benign-rendering-change negatives. Compare CV, image-only ablations, reference/current ordering, and visual-prefix removal. Measure JSON validity, localization, category recall, latency, memory, and repeatability before enabling AI serving. Keep a final test split untouched during tuning.
