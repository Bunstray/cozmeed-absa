---
name: lightweight-serving-docs
description: Build CPU-efficient FastAPI endpoints, Streamlit dashboards, and complete architectural documentation without Docker.
---

# Lightweight Serving & Documentation Workflow

## Runtime & Deployment Rules
- Load the IndoBERT model once at startup using `torch.inference_mode()` on CPU (`device="cpu"`).
- Keep `src/api.py` (FastAPI `POST /predict` + batch scoring) decoupled from `src/dashboard.py` (Streamlit UI) so Streamlit Community Cloud can run standalone without needing a separate background API process.
- Ensure total local environment footprint (`uv sync`) remains well below 2.5 GB.

## Documentation Deliverable (`README.md` & `docs/`)
Every completed stage must update:
- `README.md`: Overview, Mermaid architecture diagram, `uv` setup/run commands, baseline vs. IndoBERT Macro-F1 comparison table, error analysis, and deployment instructions.
- `docs/architecture.md`: Detailed data flow, schema contracts, and design trade-offs.
