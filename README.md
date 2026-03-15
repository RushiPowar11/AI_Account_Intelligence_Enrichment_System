# AI Account Intelligence & Enrichment System

End-to-end AI prototype for the **Fello AI Builder Hackathon**.
Converts visitor signals or minimal company input into sales-ready account intelligence.

## 🎯 Problem Statement

Sales and marketing teams face two critical data challenges:

- **Anonymous website visitors** provide little actionable insight
- **Incomplete company data** makes it hard to prioritize accounts

This system converts raw signals into structured intelligence with recommended sales actions.

## ✨ Key Features

| Feature                          | Description                               |
| -------------------------------- | ----------------------------------------- |
| **Dual Input Modes**       | Accept visitor signals OR company names   |
| **Company Identification** | Domain, industry, size, HQ, founding year |
| **Persona Inference**      | Role detection + confidence score         |
| **Intent Scoring**         | 1-10 scale + buying stage                 |
| **Deep Enrichment**        | Tech stack, business signals, leadership  |
| **Leadership Emails**      | Contact discovery with email addresses    |
| **AI Summary**             | Contextual research summary               |
| **Sales Actions**          | Specific next steps for reps              |
| **Batch Processing**       | CSV upload for bulk enrichment            |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌─────────────────┐   ┌───────────┐   ┌───────┐ │
│  │  Route   │ → │ Parallel Agents │ → │ Reasoning │ → │ Final │ │
│  │  Node    │   │                 │   │   Node    │   │ Node  │ │
│  └──────────┘   └─────────────────┘   └───────────┘   └───────┘ │
│       │                  │                   │             │     │
│       ▼                  ▼                   ▼             ▼     │
│  IP → Company    ┌──────────────┐     LLM Reasoning   Structured │
│  Resolution      │ Enrichment   │     + Heuristics    JSON Output│
│                  │ (Apollo.io)  │                                │
│                  ├──────────────┤                                │
│                  │ Web Research │                                │
│                  │ (Tavily+LLM) │                                │
│                  ├──────────────┤                                │
│                  │ Tech Stack   │                                │
│                  │ (BuiltWith)  │                                │
│                  ├──────────────┤                                │
│                  │ Hunter.io    │                                │
│                  │ (Emails)     │                                │
│                  └──────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-LLM Fallback Chain

```
Gemini → Groq (Llama-3-70B) → OpenRouter (DeepSeek)
```

Ensures reliability when rate limits are hit.

## 📊 Requirement Coverage

### 1. Company Identification ✅

- **Apollo.io** for accurate company data (industry, size, HQ)
- **Tavily + LLM** fallback for web-based extraction
- Confidence scoring per field

### 2. Persona Inference ✅

- Behavior-based persona detection
- Page patterns → role mapping (technical, RevOps, research)
- Confidence percentage (60-80% range)

### 3. Intent Scoring ✅

- Deterministic scoring from: pages visited, dwell time, repeat visits
- Stage mapping: Awareness → Consideration → Evaluation → Decision
- Score range: 0-10 with justification

### 4. Company Profile Enrichment ✅

- Website, domain, industry, company size
- Headquarters, founding year, description
- Data confidence metadata

### 5. Technology Stack Detection ✅

- **BuiltWith API** for accurate tech detection
- Homepage fingerprinting fallback
- Categories: Analytics, CRM, Frontend, Infrastructure

### 6. Business Signals ✅

- Hiring activity, funding announcements
- M&A activity, expansion signals
- Product launches, partnerships

### 7. Leadership Discovery ✅

- CEO, Founder, VP Sales, CTO extraction
- **Hunter.io integration** for email addresses
- Email pattern detection (e.g., `{first}.{last}@company.com`)
- Email confidence scores (0-100%)

### 8. AI Summary + Sales Action ✅

- Contextual 2-3 sentence summary
- Specific recommended action
- 3 actionable next steps

### 9. Batch Processing ✅

- `POST /api/enrich/batch` (up to 50 items)
- CSV upload/download in Streamlit UI

## 🔧 Tech Stack

| Component              | Technology                                   |
| ---------------------- | -------------------------------------------- |
| **Backend**      | FastAPI, Python 3.11+                        |
| **Pipeline**     | LangGraph (multi-agent orchestration)        |
| **LLM**          | Gemini 2.5 Flash (primary), Groq, OpenRouter |
| **Enrichment**   | Apollo.io, Clearbit (legacy)                 |
| **Web Search**   | Tavily API                                   |
| **Tech Stack**   | BuiltWith API                                |
| **Email Finder** | Hunter.io                                    |
| **Frontend**     | Streamlit                                    |
| **Data Models**  | Pydantic v2                                  |

## 📁 Project Structure

```
AI_Account_Intelligence_Enrichment_System/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Environment configuration
│   ├── api/
│   │   └── routes.py           # API endpoints
│   ├── graph/
│   │   ├── pipeline.py         # LangGraph pipeline
│   │   └── state.py            # Pipeline state
│   ├── agents/
│   │   ├── enrichment.py       # Apollo.io/Clearbit enrichment
│   │   ├── tavily_fallback.py  # Web search fallback
│   │   ├── web_research.py     # Leadership + signals + Hunter
│   │   ├── tech_stack.py       # BuiltWith integration
│   │   ├── reasoning.py        # Intent/persona/summary
│   │   ├── llm_client.py       # Multi-LLM client
│   │   └── ip_resolver.py      # Demo IP resolver
│   └── models/
│       ├── inputs.py           # Request models
│       ├── enrichment.py       # Company profile
│       ├── outputs.py          # AccountIntelligence
│       └── batch.py            # Batch processing
├── streamlit_app/
│   └── app.py                  # Streamlit UI
├── data/
│   ├── sample_visitors.json
│   └── sample_companies.csv
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd AI_Account_Intelligence_Enrichment_System
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` with your API keys:

```env
# LLM (at least one required)
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key           # Fallback

# Enrichment (recommended)
APOLLO_API_KEY=your_apollo_key       # Company data
TAVILY_API_KEY=your_tavily_key       # Web search
HUNTER_API_KEY=your_hunter_key       # Email finder
BUILTWITH_API_KEY=your_builtwith_key # Tech stack
```

### 3. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health check: `GET http://localhost:8000/health`
- API docs: `GET http://localhost:8000/docs`

### 4. Run Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

## 📝 API Usage

### Single Enrichment

```bash
curl -X POST http://localhost:8000/api/enrich \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "Fractal",
    "domain": "fractal.ai",
    "visitor": {
      "pages_visited": ["/pricing", "/case-studies"],
      "time_on_site": "3m 42s",
      "visits_this_week": 3
    }
  }'
```

### Example Output

```json
{
  "company_name": "Fractal",
  "domain": "fractal.ai",
  "industry": "management consulting",
  "company_size": "5200 employees",
  "headquarters": "New York, New York, United States",
  "founding_year": "2000",
  "likely_persona": "Head of Sales Operations / RevOps",
  "persona_confidence": 76,
  "intent_score": 8.0,
  "intent_stage": "Decision",
  "ai_summary": "Fractal is a multinational AI and data analytics company...",
  "recommended_sales_action": "Launch fast follow-up with tailored proof points",
  "action_steps": [
    "Route account to AE immediately",
    "Send use-case specific proof points",
    "Propose a 30-minute discovery session"
  ],
  "technology_stack": ["Analytics: Google Tag Manager", "Frontend: React"],
  "leadership": ["Srikanth Velamakanni - CEO"],
  "leadership_contacts": [
    {
      "name": "Srikanth Velamakanni",
      "title": "CEO",
      "email": "srikanth@fractal.ai",
      "confidence": 95
    }
  ],
  "email_pattern": "{first}.{last}@fractal.ai",
  "business_signals": ["Funding signal detected", "Expansion signal detected"]
}
```

## 🎥 Demo

[Loom Demo Video - 5-10 minutes]

- Problem statement and why this matters
- Architecture walkthrough
- Live demo with real companies
- Output explanation

## 📈 What Makes This Stand Out

1. **Multi-Agent Architecture** - LangGraph orchestration, not single prompts
2. **Robust Fallbacks** - Multiple data sources, LLM providers
3. **Production-Ready** - Error handling, logging, confidence scoring
4. **Actionable Output** - Emails, next steps, not just data
5. **Real APIs** - Apollo.io, Hunter.io, BuiltWith, Tavily

## 🔮 Optional Extensions Implemented

- ✅ Multi-agent research workflows (LangGraph)
- ✅ Automated enrichment pipelines
- ✅ Data confidence scoring
- ✅ Batch processing

---

Built for the **Fello AI Builder Hackathon** 🚀
Contact:- rushikeshpowar90@gmail.com
