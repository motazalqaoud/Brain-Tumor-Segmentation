# GitHub Copilot Instructions — Motaz Alqaoud

## About this project
Healthcare AI project by Motaz Alqaoud, PhD — Senior AI/ML Engineer at Abbott.
Focus: medical imaging, clinical AI, surgical navigation, RAG systems.

## Coding style
- Python 3.11+, type hints on all functions
- Google-style docstrings with Args, Returns, and "Clinical note:" section
- Black formatting (line length 88)
- Descriptive variable names

## Clinical AI rules
- Never hardcode drug dosages without physician confirmation note
- Always include safety disclaimers in patient-facing output
- Medical data must preserve spacing and orientation metadata
- Prefer explicit error messages over silent failures

## Preferred libraries
- PyTorch, MONAI, sentence-transformers, FAISS, Gradio, pytest

## Comments
- Explain the "why" not just the "what"
- Add "Clinical note:" for medically significant decisions
- Mark TODOs: # TODO(motaz): description

