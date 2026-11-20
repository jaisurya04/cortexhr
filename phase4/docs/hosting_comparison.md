# Day 17 step 4 - Hosting options

**Date checked:** facts gathered 2026-11 while building (searches 2026-11-17); this table was re-formatted 2026-11-18 and the pages were NOT re-fetched that day. Open the source links and re-check before you pay or decide. Where a number comes from a third-party article instead of the provider, it says so.

Features and limits change; re-check the linked pages before you decide.

| Option | Memory limit | Sleep behaviour | Build limits | Price per month | Fit for CortexHR | Source (open to verify) |
|---|---|---|---|---|---|---|
| **Render - Free web service** (API) | 512 MB RAM, 0.1 CPU | spins down after **15 min** idle; about 1 min to wake; 750 free instance hours per workspace | 500 build minutes / month (third-party summary of Render's free tier) | $0 | Works for demos if the API fits in 512 MB (see below); first request after idle is slow | https://render.com/docs/free (official: 15-min spin-down, 750 h); 512 MB / 0.1 CPU and 500 build min are from a third-party summary - confirm on https://render.com/pricing |
| **Render - Starter** | 512 MB RAM, 0.5 CPU | always on | as above | $7 | The safe choice if the free instance runs out of memory or the wake-up delay hurts | https://render.com/pricing (official price list) |
| **Streamlit Community Cloud** (console only) | official docs (Feb 2024): 690 MB minimum, 2.7 GB maximum, may change | sleeps after **12 h** without traffic; anyone can wake it | built from GitHub on every push | $0 | Good for the console; it cannot host the FastAPI service | https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app (official; figures dated Feb 2024) |
| **Hugging Face Spaces (Docker)** | 16 GB RAM, 2 vCPU (CPU Basic) | free hardware sleeps after 48 h idle | - | Hub docs now say creating a Docker or Gradio Space needs a paid plan (PRO) | Not free for a FastAPI container today | https://huggingface.co/docs/hub/spaces-overview (official) |

Sources: Render docs and pricing pages (free web services: 512 MB, 15-minute spin-down, 750 hours; Starter $7); Streamlit docs "Manage your app" (limits and 12-hour hibernation); Hugging Face Hub docs "Spaces overview" and "Spaces GPUs" (hardware table, sleep time, paid plan for Docker/Gradio Spaces). Fly.io and Railway were not evaluated.

## Memory check (measured, not guessed)
After loading every model the API process used **about 200 MB** (peak 206 MB) in a test without FastAPI, Pydantic, uvicorn and statsmodels. Those add roughly 100-200 MB (an estimate; it could not be measured offline), so the free 512 MB instance is plausible but tight. If the first deploy shows an out-of-memory restart, use Starter.

## Recommendation
* **API:** Render free web service to start (as the task says "Free is fine to start"), moving to Starter ($7) if memory or the 15-minute sleep is a problem. Configuration: `deploy/render.yaml`.
* **Console:** Streamlit Community Cloud, main file `phase4/src/cortexhr_phase4/console/app.py`, dependencies from `phase4/src/cortexhr_phase4/console/requirements.txt`, secret `CORTEXHR_API_URL` = the Render address.
* **Reason:** both are free, deploy straight from the GitHub repository, support environment variables / secrets, and the sleep behaviour is acceptable for a training demo. State this choice and reason in your report.
