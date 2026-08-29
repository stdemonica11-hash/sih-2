# SIH26104 — Voice Impersonation Detection

Prototype for Smart India Hackathon problem statement SIH26104: near-real-time
detection of AI-generated / cloned speech with a continuously updated risk score.

**Honest status:** the pipeline (audio processing, VAD, windowing, risk engine,
dashboard, streaming) is real and tested. The detection score currently comes
from a clearly-labeled **mock heuristic** — the trained AASIST model plugs in
via `VID_MODEL_PATH` with zero other changes.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

- **Dashboard: http://localhost:8000** — live mic