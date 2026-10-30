"""Run the chatbot over REAL HTTP against the mock HR service and write the Task 9/10 evidence transcripts.

    python scripts/run_chat_transcripts.py                       # starts the stdlib replica on port 9000
    python scripts/run_chat_transcripts.py --url http://127.0.0.1:9000   # use the official uvicorn mock you started

Writes reports/phase2/chat_transcripts.md.  The header says which server answered, so the evidence is honest.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import chatbot_runner_helpers as H  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default=None); a = ap.parse_args()
    H.main(a.url)
