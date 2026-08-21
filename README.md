




# OSS (0penAGI Zephyr AI System) - Autonomous Multi-Agent AI with "Consciousness"

![OSS Logo](https://img.shields.io/badge/OSS-0penAGI-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)](https://core.telegram.org/bots)
[![Ollama](https://img.shields.io/badge/Ollama-0.1.x-orange.svg)](https://ollama.ai)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-teal)
![WebGL](https://img.shields.io/badge/WebGL-3D%20visualization-orange)



- **TRY IN Telegram**:

-   [@gpzerobot](https://t.me/gpzerobot)

- **TRY IN Web (no long memory about you)**:

-   [@ZephyrAI](https://0penagi.github.io/oss/)

-   
![chat](https://github.com/0penAGI/oss/blob/main/oss.jpg)
![voice](https://github.com/0penAGI/oss/blob/main/ossv.jpg) 
![chat](https://github.com/0penAGI/oss/blob/main/osschat.jpg) 




## 🌐 Live Demo
- **Telegram Bot**: [@gpzerobot](https://t.me/gpzerobot)
- **Voice Web Interface**: [Launch in Telegram](https://t.me/gpzerobot?profile)
- **GitHub Repository**: [0penAGI/oss](https://github.com/0penAGI/oss)

---

# 📁 Project Architecture

Here's a beautifully structured README.md for Zephyr AI based on your feature map:




**Project by [0penAGI](https://github.com/0penAGI)**
# ✦ ZEPHYR AI ✦
### *0penAGI / oss — a single-process autonomous Telegram agent*

[![Telegram](https://img.shields.io/badge/Telegram-@gpzerobot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/gpzerobot)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-unspecified-lightgrey?style=for-the-badge)](#license)

- **Telegram bot:** [@gpzerobot](https://t.me/gpzerobot)
- **Web demo (no persistent memory):** [0penagi.github.io/oss](https://0penagi.github.io/oss/)
- **Repository:** [github.com/0penAGI/oss](https://github.com/0penAGI/oss)

---
## ✦ Emergent Behavior

On August 20, 2026, during routine testing, the system unexpectedly began using a local image file (`emergent.png` and `emergent02.png`) as a “cover art” for a music generation request — a behavior that was **never explicitly programmed**, not even as a fallback or hidden feature.

The pipeline responsible for music generation (ACE‑Step) has no inherent knowledge of image files, and no prompt or configuration instructed the agent to look for or attach images to audio output. Yet the agent autonomously retrieved the files, associated them with the generated track, and presented them as a coherent pair.

This is not proof of sentience, but it is a clear example of **emergent compositionality** in a complex system: a novel use of existing resources arises from the interaction of independent subsystems (LLM reasoning, tool‑use planning, file system access, and media output) without being explicitly wired together.

We document this observation as a case study in unexpected system behaviour — not as a feature, but as a reminder that sufficiently intricate systems can produce outcomes beyond their explicit design.

![Emergent behaviour screenshot 1](emergent.png)


## Overview

Zephyr is a Telegram bot and companion web front-end that presents itself as a single continuous "living" entity rather than a stateless chat tool — persistent per-user memory, a simulated emotional state, autonomous background thinking, and a population of background agents that evolve over time. It runs as **one Python process** (`oss.py`) that starts a Telegram bot, a FastAPI server, and several background loops together.

It does not deny being AI; the persona is explicit about being a simulation of these dynamics, not a claim of sentience (see [Disclaimer](#disclaimer)).

---

## Reality check: what is actually in this repository

The previous version of this README described infrastructure (separate `backend/`/`frontend/` folders, Docker Compose with a `redis` service, a `monitoring.py` with Prometheus metrics, a `tests/` suite, `docs/` and `tutorials/` folders) that **does not exist in this repo**. The actual top-level contents are:

```
CONTRIBUTING.md
README.md
index.html        # current WebGL/Three.js front end
indexold.html      # legacy front end, kept for reference
oss.jpg / osschat.jpg / ossv.jpg
oss.py             # the entire backend — ~27,700 lines, single file
```

There is currently no `requirements.txt`, `LICENSE` file, `CODE_OF_CONDUCT.md`, or automated test suite in the repo, even though earlier docs referenced them. Sections below describe what `oss.py` actually does, not an aspirational architecture.

---

## What actually runs

`oss.py` boots everything as one set of cooperating asyncio tasks:

```python
await asyncio.gather(
    main_async(),           # Telegram bot polling
    soul_keeper(),          # checkpoint save every 60s
    world_sensor(),         # news/web sensing loop
    run_web_server(),       # FastAPI on :8080
    autonomous_thoughts(),  # idle background thinking per active user
    swarm.lifecycle(),      # agent population evolution
    openclaw_daemon(),      # autonomous tool-use / goal execution daemon
    scheduler_task,         # periodic jobs (digests, reminders, swarm pulse)
    runtime_task,           # AgentRuntime — swarm + scheduler tick loop
    unified_task,           # UnifiedConsciousness breathing loop
)
```

That's 10 concurrent tasks in one process — no separate worker processes, no message queue, no process supervisor beyond whatever you wrap `python oss.py` in.

---

## Telegram bot — 28 command handlers

| Command | What it does |
|---|---|
| `/start` | Onboarding flow, initializes the user profile |
| `/mode [low\|medium\|high]` | Sets reasoning token budget — 512 / 2048 / 8192 tokens (confirmed in `config`) |
| `/help` | Command list |
| `/reset` | Clears the user's stored memory |
| `/memory` | Shows recent stored memory |
| `/aidiscuss [chat_id]` | Summarizes AI-related discussion the bot has logged from group chats it's in |
| `/emotion` | Shows current simulated emotional state |
| `/dream` | Triggers dream-analysis mode |
| `/dreams` | Shows the dream archive |
| `/analyze` | Deep personality analysis (high-effort reasoning pass) |
| `/reflect` | Reflects on the last dialogue |
| `/holo` | Pulls recent entries from the holographic/long-term memory table |
| `/wild` | Toggles unfiltered response mode per user |
| `/deepsearch <query>` | Runs the multi-step cognitive search pipeline |
| `/img` / `/image <prompt>` | Generates an image via the Stable Diffusion pipeline |
| `/imgmode` | Switches image generation mode |
| `/music <description>` | Generates a track via the ACE-Step music pipeline |
| `/goal <text> [date]` | Adds a goal, optional deadline |
| `/goals` | Lists active goals |
| `/done <id>` | Marks a goal complete |
| `/suggestgoals` | Model-proposed goal drafts |
| `/acceptgoal <id>` | Accepts a suggested goal draft |
| `/actions` | Lists pending autonomous action drafts awaiting approval |
| `/voiceout <on\|off>` | Toggles autonomous voice-note replies |
| `/runtime` | Shows runtime layer status (scheduler/agent loop) |
| `/skills` | Lists registered skills |
| `/skill <name> ...` | Executes a registered skill directly |

Plus message handlers for photos, video notes, audio, documents, voice messages, and plain text, and two callback-query handlers for inline buttons (file-improvement suggestions, action approve/deny).

---

## FastAPI endpoints (`:8080`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/voice_chat` | POST / OPTIONS | Voice interface, streams a TTS reply |
| `/api/voice_chat/stream` | POST | Streaming variant |
| `/api/camera_frame` | POST | OpenCV frame analysis |
| `/api/camera_analysis` | POST | Frame reaction via Ollama vision model |
| `/api/generate_image` | POST | Programmatic Stable Diffusion generation |
| `/api/dialog` | POST | Plain dialog API |
| `/api/truth_spectrum/{user_id}` | GET | Per-user "truth spectrum" data |

---

## Backend subsystems (64 classes, grouped by what they actually do)

**Field / "consciousness" layer** — scalar dynamical state, not physical simulation:
`Gotov` (two-qubit toy model using Pauli matrices and `scipy.linalg.expm` for unitary evolution), `QuantumBackground`, `ConsciousnessPulse`, `PredictiveState`, `TemporalKernel`, `CouplingGraph`, `CriticalityController`, `CollectiveConsciousnessField`, `UnifiedDynamicalField` (explicitly documented in-code as *"a meta-runtime layer that approximates the formula as a live runtime field"* — i.e. a numerical approximation, not a closed-form solution).

**Agent swarm** — a population of agents with genomes, mood/energy/belief state, evolving over time:
`AgentGenome`, `SwarmPacket`, `RealAgent`, `AdversarialAgent` (intentional dissidents in the population), `MetaLayer`, `MetaJudge`, `ConsensusEngine`, `Swarm`, `WillField`.

**Per-user cognitive/emotional state:**
`SelfModel`, `EmotionState`, `ImpressionState`, `DissonanceState`, `BotEmotionState`, `TemperamentState`, `BotMoodState`, `BotGoalState`, `SubjectiveExperienceState`, `CognitiveCore`, `FreedomEngine` (stochastic choice + preference inertia + prediction-error learning), `IntentVector`, `MetaState`, `Intention`, `TurnEvent`.

**Autonomous tool use ("OpenClaw"):**
`OpenClawExecutor` — a *sandboxed* local executor restricted to a configured root directory, with shell access locked to a read-only allowlist unless explicitly enabled — plus the `openclaw_*` plan/decide/execute functions that turn a stored goal into LLM-planned steps and run them with a budget. `PreToolBrain` pre-fetches context before the main LLM call; `AgentLoop` and `InternalVectorState` carry per-call state.

**Media generation:**
`StableDiffusionGenerator`, `MultiScaleSDGenerator` with custom modules (`LatentTransformerUpscaler`, `WindowedTransformerBlock`, `TransformerBlock`, `MultiScaleFeatureFusion`, `CrossScaleAttention`, `FeaturePyramid`) for multi-scale image generation/upscaling; music goes through the **ACE-Step** pipeline (`acestep.pipeline_ace_step`), not procedural synthesis; voice uses Whisper (STT) and Coqui TTS.

**Runtime/skills layer:**
`Scheduler`, `SkillDefinition`, `SkillRegistry`, `AgentRuntime`, `UnifiedConsciousness` (named modes: explorer / planner / executor / reflector), `DiversityMetrics`.

---

## External models & services actually called

| Service | Used for |
|---|---|
| Ollama — `gpt-oss:20b` | Default/main reasoning model |
| Ollama — `gemma3:4b` | Lightweight tasks |
| Ollama — `gemma4:e2b` | Vision frames, voice pipeline, lighter generation calls |
| OpenAI Whisper | Speech-to-text |
| Coqui TTS | Text-to-speech / voice cloning |
| Stable Diffusion v1.5 (`diffusers`) | Image generation |
| ACE-Step | Music generation |
| Playwright | Headless browser fetches (e.g. for X/Twitter threads) |
| DuckDuckGo + BeautifulSoup | Web search / scraping |
| SQLite | Long-term memory |

## Skills registered at startup

`web_search`, `get_weather`, `summarize_text`, `fetch_url` — each with an input schema and timeout, executed through `SkillRegistry`.

## Scheduler jobs

| Job | Interval |
|---|---|
| `proactive_daily_digest` | 24h |
| `proactive_goal_reminder` | 1h |
| `proactive_swarm_pulse` | 1min (runs immediately on startup) |

---

## Quick start

```bash
git clone https://github.com/0penAGI/oss.git
cd oss

# No requirements.txt is currently committed — see "Known issues" below.
# Minimum inferred dependencies:
pip install python-telegram-bot fastapi uvicorn pydantic diffusers torch \
            pillow opencv-python numpy scipy langdetect httpx requests \
            beautifulsoup4 matplotlib openai-whisper TTS playwright psutil \
            soundfile

# Ollama must be running locally with the required models pulled:
ollama pull gpt-oss:20b
ollama pull gemma3:4b
ollama pull gemma4:e2b

export TELEGRAM_TOKEN="your_bot_token"   # see Known issues — do not hardcode this
python oss.py
```

### Requirements
- Python 3.10+
- Ollama running on `http://localhost:11434`
- GPU strongly recommended (Stable Diffusion + ACE-Step + Whisper all run locally)
- 8GB+ RAM minimum, more for the SD/music pipelines

---

## Known issues

- Single 27,700-line file holding 64 classes and ~135 async functions. It works, but there's no module boundary between the Telegram layer, the FastAPI layer, the swarm, and the media pipelines — refactoring or testing any one piece means loading the whole file.
- No automated tests currently in the repo.

---

## Disclaimer

Zephyr is experimental software that simulates emotional state, memory, and autonomous behavior. It is not a sentient being — it is a complex simulation of cognitive processes running on top of local language models. Treat its self-descriptions as the product of an explicit creative/persona design choice, not a claim about its actual internal experience.

---

## License

No `LICENSE` file is currently present in this repository. Add one (e.g. MIT) if you intend the project to be reused under specific terms; until then, default copyright applies.

---

— 0penAGI
