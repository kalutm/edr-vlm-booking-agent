# EDR VLM Booking Agent

An academic prototype demonstrating a **Vision-Language Model (VLM) browser agent** for the Ethiopian-Djibouti Railway (EDR) booking website.

Built with **Gemini 2.5 Flash** + **Playwright** + **FastAPI**.

## Four Required Modules Demonstrated

| Module | Description |
|--------|-------------|
| **Visual Perception** | Screenshot → Gemini VLM → `PerceptionResult` (structured JSON) |
| **State Tracking** | Explicit `AgentState` dataclass + `WorkflowStep` state machine |
| **Feedback Loop** | OBSERVE → PERCEIVE → UPDATE → EXECUTE → repeat |

## Quick Start

```bash
# 1. Set your Gemini API key
export GEMINI_API_KEY=your_key_here

# 2. Create virtualenv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Install Playwright browsers
playwright install chromium

# 4. Run the POC (single feedback loop demo)
python poc.py

# 5. Run the full UI server
python -m edr_agent
# Open http://localhost:8000
```

## Architecture

```
edr_agent/
├── config.py              # UserConfig, station list, settings
├── state.py               # AgentState, WorkflowStep state machine
├── date_logic.py          # Deterministic EDR date arithmetic
├── policy.py              # NORMAL / SEAT-FIRST / DATE-FIRST policies
├── agent.py               # Feedback loop + orchestrator
├── browser/
│   └── controller.py      # Playwright browser management
├── vlm/
│   ├── client.py          # Gemini API adapter (swappable)
│   ├── schemas.py         # Pydantic schemas for all VLM I/O
│   └── perception.py      # Visual Perception Module
└── ui/
    ├── server.py          # FastAPI + WebSocket server
    └── static/
        └── index.html     # Control panel
```

## Booking Policies

| Policy | Behavior |
|--------|----------|
| **NORMAL** | Preferred date + preferred seat. Stop if seat unavailable. |
| **SEAT-FIRST** | Preferred seat drives everything. Advance date if needed. |
| **DATE-FIRST** | Date fixed. Fall through ranked seat list. |

## Running Tests

```bash
pytest tests/ -v
```

## Safety Notes

This is an **educational prototype**.

- No payment automation
- No identity credential storage  
- No CAPTCHA bypass
- Human handoff at identity/payment steps
- Behaves as a normal browser user
