# Chemical Groups Web Application

A modern, robust platform for uploading, validating, and merging chemical group CSV data. The system features a FastAPI backend and a Streamlit frontend, coordinated by a powerful launcher script (`RUN.sh`).

---

## Table of Contents
- [Features](#features)
- [Architecture & Repository Structure](#architecture--repository-structure)
- [Setup & Quickstart](#setup--quickstart)
- [Running the Application](#running-the-application)
  - [Local (localhost-only)](#1-run-locally-default)
  - [On a Local Network (LAN)](#2-run-on-local-network-lan)
  - [On the Internet (Production)](#3-run-on-the-internet-production)
- [Configuration & Customization](#configuration--customization)
- [Backend Details](#backend-details)
- [Frontend Details](#frontend-details)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## Features
- Upload and validate CSV files for specific chemical groups
- Strict schema and data validation against a reference
- Conflict detection, merge history logging, and prevention of accidental deletions
- Advanced data visualization dashboard for exploring chemical data
- Interactive filtering, sorting, and statistical analysis tools
- Chemical property visualizations and timeline analytics
- Intuitive Streamlit web interface
- REST API (FastAPI) for programmatic access
- Automated one-command launch with advanced error handling

---

## Architecture & Repository Structure
```
.
├── RUN.sh              # Main launcher script (backend + frontend)
├── src/
│   ├── backend/
│   │   └── main.py     # FastAPI backend
│   └── frontend/
│       └── app.py      # Streamlit frontend
├── data/               # Data/config directory (chemical_groups.json, Example.csv, master CSVs)
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project metadata
├── .venv/              # (Recommended) Virtual environment
└── README.md           # This file
```

---

## Setup & Quickstart
### Prerequisites
- Python 3.7+ (3.11+ recommended)
- Unix-like OS (Linux/Mac/WSL)
- [uv](https://github.com/astral-sh/uv) (optional, for faster installs)

### One-Command Launch (Recommended)
From the project root:
```bash
bash RUN.sh install
```
- Installs dependencies if needed
- Starts backend (FastAPI) and frontend (Streamlit)
- Logs to `backend.log` and `frontend.log`

To stop all services:
```bash
bash RUN.sh stop
```

---

## Running the Application

### 1. Run Locally (Default)
This is the simplest mode—accessible only from your own computer.

- **Start with:**
  ```bash
  bash RUN.sh install
  ```
- **Backend:** http://localhost:8000
- **Frontend:** http://localhost:8501
- **No changes needed.**

### 2. Run on Local Network (LAN)
To access the app from other devices on your network:

1. **Start with network mode:**
   ```bash
   bash RUN.sh network
   ```
   The script will automatically detect your IP address and configure everything.

2. **Specify a custom IP (if needed):**
   ```bash
   bash RUN.sh network 192.168.1.42
   ```
   Use this if you have multiple network interfaces or want to specify a particular IP.

3. **Access from any device on your LAN:**
   - Frontend: http://YOUR_IP:8501
   - Backend: http://YOUR_IP:8000

**Note:**
- Open firewall ports 8000 and 8501 if needed.
- For best results, set a static IP for your host machine.
- The RUN.sh script will display the network URLs when it starts.

### 3. Run on the Internet (Production)
To expose the app publicly:

1. **Use a secure server (cloud VM, etc.)**
2. **Reverse proxy with HTTPS (highly recommended):**
   - Use Nginx/Apache/Caddy to proxy `localhost:8000` and `localhost:8501` to your public domain.
   - Obtain SSL certificates (e.g., with [Let's Encrypt](https://letsencrypt.org/)).
3. **Set CORS for your public domain:**
   - Example:
     ```bash
     export ALLOWED_ORIGINS="https://yourdomain.com"
     ```
4. **Open firewall ports 80/443 (HTTP/HTTPS).**
5. **Security Best Practices:**
   - Use strong passwords for server access.
   - Restrict backend API to only needed origins.
   - Regularly update dependencies.
   - Consider running behind authentication or VPN for sensitive data.

---

## Configuration & Customization
- **Ports:**
  - Default: Backend 8000, Frontend 8501.
  - Change by editing `RUN.sh` or passing `--backend-port=XXXX`/`--frontend-port=XXXX`.
- **CORS:**
  - Set `ALLOWED_ORIGINS` env variable to allow frontend-backend communication across hosts.
- **Backend URL in Frontend:**
  - Edit `src/frontend/app.py`, change the `BACKEND` variable to your backend's address.
- **Data Files:**
  - `data/chemical_groups.json`: List of allowed chemical groups.
  - `data/Example.csv`: Reference CSV schema.
  - `data/master_<Group>.csv`: Master data for each group.
  - `data/merge_history.csv`:  
- **Entrypoint:** `src/backend/main.py`
- **Framework:** FastAPI
- **API Endpoints:**
  - `GET /chemical_groups` — List groups
  - `POST /upload_csv` — Upload/merge CSV for a group
  - `GET /health` — Health check
- **Validation:** Checks schema, ID patterns, group types, prevents deletions, detects conflicts.
- **CORS:** Controlled by `ALLOWED_ORIGINS` (default: `http://localhost:8501`).
- **Logs:** `backend.log`

## Frontend Details
- **Entrypoint:** `src/frontend/app.py`
- **Framework:** Streamlit
- **Features:**
  - File upload and group selection UI
  - Displays upload results, conflicts, troubleshooting
  - Data visualization dashboard with multiple analysis tools:
    - Data explorer with advanced filtering and search capabilities
    - Statistical analysis of numerical properties
    - Chemical property visualizations (SMILES complexity, type distribution)
    - Timeline analysis of data entries over time
  - Export functionality for filtered datasets
- **Backend URL:** Set in `app.py` (`BACKEND = ...`)
- **Logs:** `frontend.log`

---

## Dependencies
- Listed in `requirements.txt` and `pyproject.toml`
- Key packages:
  - fastapi, uvicorn
  - streamlit
  - pandas
  - requests
  - pydantic
  - python-multipart
- Install with `pip install -r requirements.txt` or via `uv`

---

## Troubleshooting
- **Backend/Frontend not starting:**
  - Check `backend.log` and `frontend.log` for errors
  - Ensure required ports (8000, 8501) are free (or let `RUN.sh` auto-select)
- **Missing data files:**
  - `RUN.sh` creates default `chemical_groups.json` if missing
  - Ensure `Example.csv` exists in `data/`
- **Dependency issues:**
  - Re-run `pip install -r requirements.txt` or `uv pip install -r requirements.txt`
- **To stop all services:**
  - `bash RUN.sh stop`
  - Or manually kill processes using `ps` and `kill`

---

## FAQ
**Q: Can I run backend and frontend on different machines?**
- Yes. Update the `BACKEND` URL in `frontend/app.py` and set `ALLOWED_ORIGINS` in backend accordingly.

**Q: How do I add a new chemical group?**
- Edit `data/chemical_groups.json` and add the group name to the `groups` list.

**Q: How do I view logs?**
- `backend.log` and `frontend.log` in the project root contain logs for each service.

**Q: How do I restart everything?**
- `bash RUN.sh restart` or stop then start manually as above.

**Q: How do I make the application accessible on my network?**
- Run `bash RUN.sh network` to configure the app for network access.
- For more control, you can specify an IP address: `bash RUN.sh network 192.168.1.42`.
- If you need to go back to local-only mode, use `bash RUN.sh local`.

**Q: How do I secure the app for production?**
- Always use HTTPS, restrict CORS, use a reverse proxy, and keep dependencies up-to-date.

---

For further details, consult the code comments or contact the maintainer.
