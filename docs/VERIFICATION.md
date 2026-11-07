# Verification evidence

Executed on macOS arm64, Python 3.12, Node 22, and Docker Desktop on November 7, 2026. No pretrained diagnosis-quality claim is made.

| Check | Observed result |
| --- | --- |
| `pytest -q` with ML dependencies installed | 15 passed; one upstream Starlette/httpx deprecation warning |
| Tiny random SigLIP2/Gemma training | 12 steps; example NLL 3.4788 → 3.4712 |
| Gradient/freeze checks | Nonzero projector gradients; no gradients on frozen base models |
| Safe adapter checkpoint round trip | Visual prefixes and greedy outputs match after reload |
| `npm run build --prefix frontend` | TypeScript and production Vite build passed |
| Playwright dataset generation | 120 pairs, six controlled defects, 20 variants, three disjoint layout families |
| CV held-out evaluation | 36 examples; recall at IoU .5 = 28/36; mean best IoU .7343; 738 predicted regions |
| Real browser upload → API → selected overlay → JSON download | Passed; no page JavaScript errors |
| Mobile viewport | 390 px wide, no horizontal document overflow |
| Docker images and Compose health | Both images built, both containers started |
| Production nginx → FastAPI screenshot upload | Passed; one color-change region on included fixture |
| Production `/api/openapi.json` | Served correctly with `/api` server prefix |
| SigLIP2 pretrained configuration access | HTTP 200 |
| Gemma pretrained configuration access | HTTP 401; pretrained training/inference blocked |

The recorded browser demo is `demo.webm`; screenshots and report are included in this directory. The demo uses generated fixtures and the real running CV endpoint. No model outputs are substituted.

Linux GitHub Actions repeats the suite, renderer, build, browser smoke, and Docker checks. Consult the linked workflow run for its result; local macOS checks alone do not establish Linux success.

## Limitations exposed by verification

- Many detections fragment into text/edge components: 738 predictions for 36 examples. Recall does not establish precision.
- One broad structural layout family per split provides isolation but weak statistical coverage. More independent real-page layouts and unchanged negatives are needed.
- Ground truth covers the injected target union, not every element affected by reflow.
- No calibrated confidence scores, semantic AI quality, GPU training, or public hosted service has been established.
- Base-model downloads and gated access are separate from architecture tests. The training/validation and model evaluation CLIs are implemented, but their pretrained end-to-end execution is unvalidated.
- The API intentionally returns 503 for `/predict`; enabling it requires a trained, evaluated checkpoint and a separate serving implementation.
