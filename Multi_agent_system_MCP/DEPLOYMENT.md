# Deploying the Multi-Agent Travel Planner to a VPS

This is the **Part 4** deployment, written out in full — including the systemd,
nginx, domain, and HTTPS steps the original repo README leaves blank.

Prerequisites you must have yourself:
- A VPS (Ubuntu 22.04/24.04) with root/sudo SSH access
- A domain name pointed at the VPS IP (an `A` record) — only needed for HTTPS
- Your 4 API keys (Groq, Tavily, AviationStack, OpenWeather)

Throughout, replace:
- `YOUR_VPS_IP` with your server's IP
- `your-domain.com` with your domain
- `/opt/Multi_agent_system_part_3` if you clone elsewhere

---

## Step 1 — Connect to the VPS
```bash
ssh root@YOUR_VPS_IP
```

## Step 2 — Update & install system packages
```bash
apt update && apt upgrade -y
apt install python3-pip python3-venv nginx git curl postgresql postgresql-contrib -y
python3 --version
```

## Step 3 — Set up PostgreSQL
```bash
sudo -u postgres psql -c "CREATE DATABASE langgraph_memory_demo;"
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'YOUR_DB_PASSWORD';"
```
Your `DATABASE_URL` will be:
`postgresql://postgres:YOUR_DB_PASSWORD@localhost:5432/langgraph_memory_demo`

## Step 4 — Clone the project
```bash
cd /opt
git clone <YOUR_PROJECT_GIT_URL> Multi_agent_system_part_3
cd Multi_agent_system_part_3
```
> Push this project (the `Multi_agent_system_MCP` folder contents) to your own
> Git remote first, or `scp` the files up. Do **not** commit your real `.env`.

## Step 5 — App virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 6 — AviationStack MCP server (its own venv, via uv)
Clone it **inside** the project folder so the app finds it automatically:
```bash
pip install uv
git clone https://github.com/Pradumnasaraf/aviationstack-mcp.git
cd aviationstack-mcp
uv sync                         # creates ./.venv with the package installed
cd ..
```
> `mcp_client.py` looks for `./aviationstack-mcp/.venv/bin/python`. If you put it
> elsewhere, set `AVIATION_MCP_PYTHON=/path/to/.venv/bin/python` in your `.env`.

## Step 7 — Environment variables
```bash
nano .env
```
```ini
GROQ_API_KEY=xxxx
TAVILY_API_KEY=xxxx
AVIATIONSTACK_API_KEY=xxxx
OPENWEATHER_API_KEY=xxxx
DATABASE_URL=postgresql://postgres:YOUR_DB_PASSWORD@localhost:5432/langgraph_memory_demo
```

## Step 8 — Smoke test
```bash
source venv/bin/activate
streamlit run frontend.py --server.address 0.0.0.0 --server.port 8501
```
Open `http://YOUR_VPS_IP:8501`. If it works, stop it with `CTRL+C` and continue
so it runs as a managed service instead.

## Step 9 — Run as a systemd service (auto-start + auto-restart)
```bash
# 'www-data' must be able to read the project; or set User= to your own user.
chown -R www-data:www-data /opt/Multi_agent_system_part_3

cp deploy/travel-planner.service /etc/systemd/system/travel-planner.service
systemctl daemon-reload
systemctl enable --now travel-planner
systemctl status travel-planner          # should be "active (running)"
journalctl -u travel-planner -f          # live logs
```
The service runs Streamlit bound to `127.0.0.1:8501` (private) — nginx exposes it.

## Step 10 — Nginx reverse proxy (port 80)
```bash
cp deploy/nginx-travel-planner.conf /etc/nginx/sites-available/travel-planner
# edit server_name to your domain:
nano /etc/nginx/sites-available/travel-planner
ln -s /etc/nginx/sites-available/travel-planner /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx
```
Now `http://your-domain.com` serves the app.

## Step 11 — HTTPS (free, auto-renewing)
Point your domain's `A` record at `YOUR_VPS_IP` first, then:
```bash
apt install certbot python3-certbot-nginx -y
certbot --nginx -d your-domain.com -d www.your-domain.com
```
Certbot rewrites the nginx config for port 443 and sets up auto-renewal.
Your app is now live at `https://your-domain.com`.

## Step 12 — Firewall (recommended)
```bash
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw enable
```

---

## Updating the app later
```bash
cd /opt/Multi_agent_system_part_3
git pull
source venv/bin/activate && pip install -r requirements.txt
systemctl restart travel-planner
```

## Troubleshooting
| Symptom | Check |
|---------|-------|
| 502 Bad Gateway | `systemctl status travel-planner`, `journalctl -u travel-planner -e` |
| App loads but no data | API keys in `.env`; `AVIATION_MCP_PYTHON` path; DB reachable |
| WebSocket / blank page | Nginx `Upgrade`/`Connection` headers (already in the provided conf) |
| certbot fails | DNS `A` record must resolve to the VPS before running certbot |
