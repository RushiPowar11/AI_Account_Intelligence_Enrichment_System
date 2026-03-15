# Fello AI Builder Hackathon - Submission Checklist

## Deliverables

- GitHub repository (public) with setup + architecture documentation.
- Loom demo (5-10 minutes) showing problem, system design, and live workflow.
- Deployed link (bonus) for API and/or Streamlit app.
- Optional 2-slide summary deck.

## Demo Script (Recommended)

1. Problem framing (1 min)
- Anonymous traffic + incomplete account data blocks outbound prioritization.

2. System design (1-2 min)
- Input routing for visitor/company.
- LangGraph pipeline (`route -> parallel_agents -> reasoning -> final`).
- Confidence-scored enrichment and structured JSON output.

3. Live walkthrough (4-6 min)
- Company-only example: show enrichment quality and sales action.
- Visitor example: show intent jump based on behavior (`pricing`, repeat visits, dwell time).
- Batch CSV example: upload -> enrich -> download.

4. Close (1 min)
- Mention extensibility: CRM sync, real reverse-IP provider, streaming events, historical tracking.

## Run Locally

```bash
pip install -r requirements.txt
copy .env.example .env
# add GOOGLE_API_KEY (or GEMINI_API_KEY), TAVILY_API_KEY

uvicorn app.main:app --reload --port 8000
streamlit run streamlit_app/app.py
```
