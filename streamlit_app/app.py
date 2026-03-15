"""Streamlit UI for AI Account Intelligence & Enrichment."""

import json
import os

import httpx
import pandas as pd
import streamlit as st

# Default API base URL (override with env STREAMLIT_API_URL)
API_BASE = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Account Intelligence",
    page_icon="📊",
    layout="wide",
)
st.title("AI Account Intelligence & Enrichment")
st.caption("Convert company names or visitor signals into sales-ready account intelligence.")

api_url = st.sidebar.text_input("API base URL", value=API_BASE, help="FastAPI server URL")
if not api_url.strip():
    api_url = API_BASE

tab_single, tab_batch, tab_about = st.tabs(["Single lookup", "Batch (CSV)", "About"])

# --- Single lookup ---
with tab_single:
    st.subheader("Single company or visitor")
    col1, col2 = st.columns([1, 1])
    with col1:
        company_name = st.text_input("Company name", placeholder="e.g. BrightPath Lending")
        domain = st.text_input("Domain (optional)", placeholder="e.g. brightpathlending.com")
        use_visitor = st.checkbox("Include visitor signals (JSON)")
    with col2:
        visitor_json = ""
        if use_visitor:
            visitor_json = st.text_area(
                "Visitor JSON",
                value='{"visitor_id": "001", "ip": "34.201.xxx.xxx", "pages_visited": ["/pricing", "/case-studies"], "time_on_site": "3m 42s", "visits_this_week": 3}',
                height=120,
            )

    if st.button("Get intelligence", type="primary"):
        if not company_name.strip() and not (use_visitor and visitor_json.strip()):
            st.error("Enter a company name and/or visitor JSON.")
        else:
            payload = {"company_name": company_name.strip() or None, "domain": domain.strip() or None}
            if use_visitor and visitor_json.strip():
                try:
                    payload["visitor"] = json.loads(visitor_json)
                except json.JSONDecodeError:
                    st.error("Invalid visitor JSON.")
                    st.stop()
            with st.spinner("Running enrichment pipeline..."):
                try:
                    r = httpx.post(f"{api_url.rstrip('/')}/api/enrich", json=payload, timeout=120.0)
                    r.raise_for_status()
                    data = r.json()
                except httpx.HTTPStatusError as e:
                    st.error(f"API error: {e.response.status_code} — {e.response.text[:300]}")
                    st.stop()
                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
                    st.stop()
            st.success("Done.")
            # Display sections
            st.subheader("Company")
            st.json({k: v for k, v in data.items() if k in ("company_name", "website", "domain", "industry", "company_size", "headquarters", "founding_year") and v})
            st.subheader("Intent & persona")
            st.json({k: v for k, v in data.items() if k in ("intent_score", "intent_stage", "intent_justification", "likely_persona", "persona_confidence") and v is not None})
            if data.get("key_signals_observed"):
                st.write("**Behavioral signals:** " + ", ".join(data["key_signals_observed"]))
            st.subheader("AI summary")
            st.write(data.get("ai_summary") or "—")
            st.subheader("Recommended action")
            st.write(data.get("recommended_sales_action") or "—")
            if data.get("action_steps"):
                for step in data["action_steps"]:
                    st.write(f"- {step}")
            if data.get("technology_stack") or data.get("leadership") or data.get("business_signals"):
                st.subheader("Enrichment")
                if data.get("technology_stack"):
                    st.write("**Tech stack:** " + ", ".join(data["technology_stack"]))
                if data.get("leadership"):
                    st.write("**Leadership:** " + ", ".join(data["leadership"]))
                if data.get("business_signals"):
                    st.write("**Signals:** " + ", ".join(data["business_signals"]))
            st.divider()
            st.subheader("Full JSON")
            st.json(data)

# --- Batch ---
with tab_batch:
    st.subheader("Batch enrichment from CSV")
    st.caption("Upload a CSV with a 'company_name' column (optional: 'domain'). One row per company.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()
        if "company_name" not in df.columns:
            st.error("CSV must have a 'company_name' column.")
            st.stop()
        st.dataframe(df.head(10), use_container_width=True)
        limit = min(len(df), 50)
        st.caption(f"Will process up to {limit} rows (max 50 per batch).")
        if st.button("Run batch enrichment"):
            items = []
            for _, row in df.head(50).iterrows():
                name = str(row.get("company_name", "")).strip()
                if not name:
                    continue
                items.append({"company_name": name, "domain": str(row.get("domain", "")).strip() or None})
            if not items:
                st.warning("No valid company_name rows found.")
            else:
                with st.spinner(f"Enriching {len(items)} companies..."):
                    try:
                        r = httpx.post(
                            f"{api_url.rstrip('/')}/api/enrich/batch",
                            json={"items": items},
                            timeout=300.0,
                        )
                        r.raise_for_status()
                        batch = r.json()
                    except httpx.HTTPStatusError as e:
                        st.error(f"API error: {e.response.status_code} — {e.response.text[:500]}")
                        st.stop()
                    except Exception as e:
                        st.error(f"Request failed: {str(e)}")
                        st.stop()
                st.success(f"Done: {batch['succeeded']} succeeded, {batch['failed']} failed.")
                # Build table for download
                rows = []
                for i, res in enumerate(batch["results"]):
                    if res["success"] and res.get("data"):
                        d = res["data"]
                        rows.append({
                            "company_name": d.get("company_name"),
                            "domain": d.get("domain"),
                            "industry": d.get("industry"),
                            "company_size": d.get("company_size"),
                            "founding_year": d.get("founding_year"),
                            "intent_score": d.get("intent_score"),
                            "intent_stage": d.get("intent_stage"),
                            "likely_persona": d.get("likely_persona"),
                            "ai_summary": (d.get("ai_summary") or "")[:200],
                            "recommended_action": (d.get("recommended_sales_action") or "")[:150],
                        })
                    else:
                        rows.append({
                            "company_name": items[i].get("company_name", ""),
                            "domain": items[i].get("domain", ""),
                            "industry": "",
                            "company_size": "",
                            "founding_year": "",
                            "intent_score": "",
                            "intent_stage": "",
                            "likely_persona": "",
                            "ai_summary": "",
                            "recommended_action": res.get("error", "Error"),
                        })
                out_df = pd.DataFrame(rows)
                st.dataframe(out_df, use_container_width=True)
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download results CSV", data=csv_bytes, file_name="account_intelligence.csv", mime="text/csv")
                with st.expander("Full batch JSON"):
                    st.json(batch)

# --- About ---
with tab_about:
    st.markdown("""
    **AI Account Intelligence & Enrichment** converts:
    - **Company names** (and optional domain) into enriched profiles with intent, persona, and recommended actions.
    - **Visitor signals** (IP, pages visited, time on site) into identified accounts and intent scoring.

    **Pipeline:** Input router → IP resolution (visitor) → Enrichment + Web research + Tech stack → Gemini reasoning → Structured output.

    **Confidence:** Every enriched field includes a confidence level and source (clearbit | tavily | input).

    **Run the API:** `uvicorn app.main:app --reload --port 8000`  
    **Run this UI:** `streamlit run streamlit_app/app.py`
    """)
