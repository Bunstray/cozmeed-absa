# AGENTS.md — Workspace Rules & Operating Protocol

## 1. Mandatory Clarification & Anti-Hallucination Rule (CRITICAL)
- **NEVER guess or fabricate** raw JSON schemas, API headers, file paths, dataset column names, or model hyperparameters when uncertain.
- **ALWAYS stop and ask the user** before proceeding if:
  1. A sample raw JSON payload (e.g., Tokopedia `productReviewList` or Shopee `get_ratings`) has not been provided yet.
  2. Business logic, aspect definitions, label taxonomy, or split boundaries are ambiguous.
  3. An external dependency, CLI tool, or credential/cookie is missing.
- When asking a question, list the exact unknown fields or options concisely and wait for confirmation before generating dependent code.

## 2. Documentation-First Mandate
- Every module in `src/` must have module-level and function-level Google-style docstrings explaining inputs, outputs, side effects, and edge cases.
- Any architectural change, schema update, new CLI command, or evaluation metric result MUST be documented in `README.md` and `docs/` in the same task.
- Never leave placeholder `TODO` documentation in completed stages.

## 3. Environment & Disk Constraints
- **NO Docker:** Never create or run Docker images/containers locally (local disk budget is strictly < 2.5 GB).
- **Package Manager:** Exclusively use `uv` (`pyproject.toml` + `uv.lock`). Never use `pip install` directly or create Conda environments.
- **CPU-Only PyTorch Locally:** Keep PyTorch pinned to the `pytorch-cpu` index in `pyproject.toml`. Heavy fine-tuning runs only in `notebooks/train_indobert.ipynb` (Google Colab / Kaggle T4 GPU).
- **Code Organization:** Keep production code modular under `src/`, unit tests under `tests/`, and exploratory notebooks isolated in `notebooks/`. Never hardcode local absolute file paths.

## 4. Verification Before Completion
- After modifying any file in `src/`, run `uv run pytest` and verify clean imports before reporting completion.
