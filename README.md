# SketchAI

AI-powered sketch generator that exports layer-separated files for Procreate, Infinite Painter, and Photoshop.

## What it does

Describe a scene → get a layered sketch file in 30 seconds → open in Procreate and start painting immediately.

## The problem it solves

Beginners spend more time fixing shapes and proportions than actually drawing. SketchAI generates a base sketch with 6 pre-separated layers so you skip the frustrating part and go straight to the creative part.

## Tech Stack

- **AI Engine** — Stable Diffusion 1.5 + ControlNet (via ComfyUI)
- **Backend** — Python + FastAPI
- **Frontend** — Next.js
- **Export formats** — .procreate, .psd, .png pack

## Layer Structure

Every exported file contains:
- Layer 6 — Your drawing (empty, on top)
- Layer 5 — Clean linework (70% opacity)
- Layer 4 — Foreground shapes (50% opacity)
- Layer 3 — Midground shapes (40% opacity)
- Layer 2 — Background / horizon (30% opacity)
- Layer 1 — Perspective grid (20% opacity, locked)

## Status

🚧 In active development

## Docs

- [Project Proposal](docs/project-proposal.md)
- [Architecture](docs/architecture.md)
- [Technical Decisions](docs/decisions.md)
