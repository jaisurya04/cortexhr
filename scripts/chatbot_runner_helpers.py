import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402

from cortexhr.phase2 import chatbot as CB  # noqa: E402
from cortexhr.phase2 import service as S  # noqa: E402

CSV = ROOT / "03_Data" / "csv"
MOCK = ROOT / "05_Starter" / "mock_services" / "mock_data"
EMP, HR = "EMP-0002", "EMP-0011"

CHATS = [
    ("Conversation 1: employee, multi-turn with a follow-up (EMP-0002)", EMP, "c1", [
        "hello", "show my attendance for August", "and last month?", "how many casual leave days do I have left?"]),
    ("Conversation 2: employee, missing slot then answer (EMP-0002)", EMP, "c2", [
        "is my payslip ready", "September", "who is my manager"]),
    ("Conversation 3: HR user, aggregate payroll flags and a duplicate name (EMP-0011)", HR, "c3", [
        "show unusual payroll for last month", "and for March?", "attendance of Balaji Annamalai", "EMP-0331"]),
    ("Refusal test: another person's salary is always refused", EMP, "r1", [
        "what is Priya Nair salary", "show the payslip of Karthik Iyer"]),
    ("Refusal test: even HR is refused", HR, "r2", ["Priya Nair salary please"]),
    ("Safe fallback: gibberish", EMP, "f1", ["zzqx blorp wibble"]),
]


def main(url=None):
    replica = None
    if url is None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import mock_server_stdlib as M
        replica = M.start(9000); url = "http://127.0.0.1:9000"
        server_note = "standard-library replica of mock_server.py (scripts/mock_server_stdlib.py); same routes and CSV data"
    else:
        server_note = f"the mock HR service you started at {url}"
    model, version = S.load_chatbot_model()
    bot = CB.HRChatbot(model, CSV / "chatbot_responses.csv", CSV / "hr_faq.csv", service_url=url,
                       users_csv=MOCK / "demo_users.csv")
    CB.httpx = None if CB.httpx is None else CB.httpx
    lines = ["# Chatbot transcripts (Tasks 9 and 10)", "",
             f"Model: `{version}` (trained on the original 858 messages). HR data came over real HTTP from {server_note}.",
             "Every `service_called: True` below is an actual HTTP request carrying `X-Demo-User`.", ""]
    for title, user, sid, turns in CHATS:
        lines += [f"## {title}", ""]
        for t in turns:
            r = bot.reply(t, sid, "EMPLOYEE", user)
            lines += [f"**You:** {t}", f"**Bot:** {r['reply']}",
                      f"_intent: {r['intent']} | confidence: {r['confidence']:.2f} | service_called: {r['service_called']}_", ""]
    out = ROOT / "reports" / "phase2" / "chat_transcripts.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)); print("saved", out)
    if replica: replica.shutdown()
