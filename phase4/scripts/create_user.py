"""Create a PBKDF2 password hash and an environment-ready CortexHR user record."""
from __future__ import annotations
import getpass, json, secrets
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexhr_phase4.core.security import hash_password, ROLES

username = input("Username: ").strip()
if not username: raise SystemExit("Username is required")
display = input("Display name: ").strip() or username
role = input("Role (admin/hr_manager/recruiter/analyst): ").strip().lower()
if role not in ROLES: raise SystemExit("Invalid role")
p1 = getpass.getpass("Password (12+ characters): ")
p2 = getpass.getpass("Confirm password: ")
if p1 != p2: raise SystemExit("Passwords do not match")
record = {username: {"display_name": display, "role": role, "password_hash": hash_password(p1)}}
print("\nAdd this JSON record to CORTEXHR_AUTH_USERS (merge it with existing users):")
print(json.dumps(record, separators=(",", ":")))
print("\nGenerate a signing secret and keep it private:")
print(secrets.token_urlsafe(48))
print("\nDo not commit either value to Git or share them in screenshots.")
