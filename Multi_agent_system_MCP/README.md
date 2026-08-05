# ✈️ AI Travel Planner — Production Multi-Agent System

A **production-grade multi-agent AI travel planner** built with LangGraph, Groq LLaMA 3.3 70B, and real-time data via the Model Context Protocol (MCP). Features live-streaming agent progress, human-in-the-loop approval, PostgreSQL-backed memory, and a polished Streamlit UI.

---

## ✨ Key Features

| Feature | Detail |
|---------|--------|
| **Multi-agent orchestration** | 7 specialised agents coordinated by an LLM supervisor |
| **Real-time data via MCP** | Live flight/airport info, weather forecasts, and web search |
| **Human-in-the-loop** | Review and approve/revise draft itineraries before finalising |
| **Persistent memory** | PostgreSQL-backed thread history via LangGraph checkpoints |
| **Input guardrails** | Supervisor rejects non-travel requests before any agent runs |
| **Live streaming UI** | Watch each agent complete in real-time with `st.status()` |
| **Thread history** | Load, resume, or delete any past conversation |
| **Export plans** | Download the final itinerary as a Markdown file |
| **VPS deployment** | Systemd service + nginx reverse proxy + HTTPS via certbot |

---

## 🏗️ Architecture

```
User Request
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LANGGRAPH STATE MACHINE                  │
│                                                             │
│  ┌──────────┐    Conditional routing (LLM decides)          │
│  │Supervisor│──►  flight? hotel? weather? budget?           │
│  └──────────┘                                               │
│       │                                                     │
│       ├──► ✈️  Flight Agent    (MCP: AviationStack)         │
│       ├──► 🏨 Hotel Agent     (MCP: Tavily web search)     │
│       ├──► 🌤️  Weather Agent  (MCP: OpenWeather)           │
│       ├──► 💰 Budget Agent    (LLM analysis)               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────┐                                        │
│  │ Itinerary Agent │  Builds the draft plan                 │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼   interrupt() ◄── Human reviews & approves      │
│  ┌─────────────────────┐                                     │
│  │ Human Approval Node │  Command(resume={approved, feedback})│
│  └────────┬────────────┘                                    │
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │ Final Response   │  Polished, personalised travel plan   │
│  └──────────────────┘                                       │
│                                                             │
│  Checkpointer: PostgresSaver (full state persisted to DB)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔌 MCP Servers

| Server | Transport | Data source |
|--------|-----------|-------------|
| **Tavily** | `streamable_http` | Web search — hotels, general travel info |
| **AviationStack** | `stdio` (dedicated venv) | Live flight, airport & airline data |
| **OpenWeather** | `stdio` (local FastMCP) | Current conditions & 5-day forecasts |

---

## 🛠️ Tech Stack

- **LLM** — Groq LLaMA 3.3 70B (via `langchain-groq`)
- **Orchestration** — LangGraph 1.x with `interrupt()` / `Command(resume=...)`
- **MCP** — `langchain-mcp-adapters` `MultiServerMCPClient`
- **Memory** — `langgraph-checkpoint-postgres` → PostgreSQL
- **UI** — Streamlit (live streaming, dark theme)
- **Language** — Python 3.11+

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL database
- API keys: Groq, Tavily, AviationStack, OpenWeather

### 1. Clone & set up virtual environment
```bash
git clone https://github.com/AmanKabra1/ai-travel-planner.git
cd ai-travel-planner
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. AviationStack MCP server (its own venv via uv)
```bash
pip install uv
git clone https://github.com/Pradumnasaraf/aviationstack-mcp.git
cd aviationstack-mcp && uv sync && cd ..
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 4. Create the PostgreSQL database
```sql
CREATE DATABASE langgraph_memory_demo;
```

### 5. Launch the app
```bash
streamlit run frontend.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
.
├── config.py                  # Env loading, logging setup, LLM factory
├── state.py                   # LangGraph TravelState TypedDict
├── mcp_client.py              # MultiServerMCPClient + typed async wrappers
├── agents.py                  # 7 agent functions (supervisor → final)
├── graph.py                   # StateGraph + conditional routing + PostgresSaver
├── thread_history.py          # list / load / delete thread helpers
├── frontend.py                # Streamlit UI (streaming, tabs, approval, export)
├── main.py                    # CLI runner (for testing without UI)
├── weather_mcp_server.py      # Local FastMCP server (OpenWeather)
├── aviationstack-mcp/         # Cloned MCP server (dedicated .venv via uv)
├── deploy/
│   ├── travel-planner.service # Systemd unit for VPS
│   └── nginx-travel-planner.conf
├── DEPLOYMENT.md              # Full VPS deployment guide (systemd + nginx + HTTPS)
├── requirements.txt           # Pinned dependencies
└── .env.example               # Environment variable template
```

---

## 🗺️ Agent Pipeline

```
1. Supervisor      — Validates input (guardrail), routes to needed agents
2. Flight Agent    — AviationStack MCP → airports, airlines, fare guidance
3. Hotel Agent     — Tavily web search → neighbourhood & hotel recommendations
4. Weather Agent   — OpenWeather MCP → current conditions + 5-day forecast
5. Budget Agent    — LLM cost analysis (uses outputs from agents 2-4)
6. Itinerary Agent — Builds full day-by-day draft plan
7. Human Approval  — interrupt() pauses graph; user approves or requests revisions
8. Final Response  — Produces polished, personalised final travel plan
```

---

## 🌐 Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the full VPS deployment guide covering:
- Ubuntu server setup
- PostgreSQL configuration
- Systemd service (auto-start + auto-restart)
- Nginx reverse proxy
- Free HTTPS via Let's Encrypt / certbot

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | ✅ | [app.tavily.com](https://app.tavily.com) |
| `AVIATIONSTACK_API_KEY` | ✅ | [aviationstack.com](https://aviationstack.com) |
| `OPENWEATHER_API_KEY` | ✅ | [openweathermap.org](https://openweathermap.org/api) |
| `DATABASE_URL` | ✅ | `postgresql://user:pass@host:5432/db` |
| `GROQ_MODEL` | ☐ | Default: `llama-3.3-70b-versatile` |
| `AVIATION_MCP_PYTHON` | ☐ | Override path to aviation venv python |

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

*Built with LangGraph · Groq · MCP · Streamlit*
