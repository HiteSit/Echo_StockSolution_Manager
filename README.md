# Chemical Groups Web Application

A web-based platform for uploading, validating, and merging chemical group CSV data. The application consists of a FastAPI backend and a Streamlit frontend, coordinated by a launcher script (`RUN.sh`).

---

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Manual Run Instructions](#manual-run-instructions)
- [Backend](#backend)
- [Frontend](#frontend)
- [Configuration & Data](#configuration--data)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## Features
- Upload CSV files for specific chemical groups
- Schema and data validation against a reference
- Conflict detection and merge history
- Easy-to-use web interface (Streamlit)
- REST API (FastAPI) for backend operations
- Automated and manual launch options

---

## Repository Structure
```
.
├── RUN.sh              # Main launcher script (backend + frontend)
├── src/                # Source code directory
│   ├── backend/        # FastAPI backend service
│   │   └── main.py
│   └── frontend/       # Streamlit frontend app
│       └── app.py
├── data/               # Data directory (chemical_groups.json, Example.csv, master CSVs)
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project metadata and dependencies
├── .venv/              # (Recommended) Virtual environment
└── README.md           # This file
```

---

## Quickstart
### 1. Prerequisites
- Python 3.7 or higher (3.11+ recommended)
- Unix-like OS (Linux/Mac; Windows WSL works)
- Recommended: `uv` for fast dependency install ([uv documentation](https://github.com/astral-sh/uv))

### 2. One-Command Launch (Recommended)
From the project root:
```bash
bash RUN.sh install
```
- Installs dependencies (if not already installed)
- Starts backend (FastAPI) and frontend (Streamlit)
- Logs are saved as `backend.log` and `frontend.log`

To stop all services:
```bash
bash RUN.sh stop
```

---

## Manual Run Instructions
If you want to run backend and frontend manually, follow these steps:

### 1. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Or with `uv` (if available):
```bash
uv pip install -r requirements.txt
```

### 3. Prepare Data Directory
Ensure `data/chemical_groups.json` and `data/Example.csv` exist. The launcher will create a default `chemical_groups.json` if missing.

### 4. Start Backend (FastAPI)
From the project root:
```bash
uvicorn src.backend.main:app --reload --port 8000 > backend.log 2>&1 &
```
- The backend API will be available at http://localhost:8000

### 5. Start Frontend (Streamlit)
From the project root:
```bash
streamlit run src/frontend/app.py --server.port 8501 > frontend.log 2>&1 &
```
- The frontend UI will be at http://localhost:8501

---

## Backend
- **Framework:** FastAPI
- **Entrypoint:** `src/backend/main.py`
- **API Endpoints:**
    - `GET /chemical_groups` — List available groups
    - `POST /upload_csv` — Upload and merge CSV for a group
    - `GET /health` — Health check
- **Data Validation:**
    - Checks schema, ID patterns, and group types
    - Prevents deletions and detects conflicts
- **Logs:** Output to `backend.log`

## Frontend
- **Framework:** Streamlit
- **Entrypoint:** `src/frontend/app.py`
- **Features:**
    - File upload UI
    - Group selection
    - Displays upload results and conflicts
    - Troubleshooting tips
- **Logs:** Output to `frontend.log`

---

## Configuration & Data
- `data/chemical_groups.json` — List of allowed chemical groups, e.g. `{ "groups": ["Amines", "Ethers"] }`
- `data/Example.csv` — Reference CSV schema (column names/order)
- `data/master_<Group>.csv` — Master data for each group (created/updated by backend)
- `data/merge_history.csv` — Log of merge operations

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
    - Ensure required ports (8000, 8501) are free (or use `RUN.sh` to auto-select ports)
- **Missing data files:**
    - `RUN.sh` will create a default `chemical_groups.json` if missing
    - Ensure `Example.csv` exists in `data/`
- **Dependency issues:**
    - Re-run `pip install -r requirements.txt` or `uv pip install -r requirements.txt`
- **To stop all services:**
    - `bash RUN.sh stop`
    - Or manually kill processes using `ps` and `kill`

---

## FAQ
**Q: Can I run backend and frontend on different machines?**
- Yes, but update the `BACKEND` URL in `frontend/app.py` and set CORS origins in backend accordingly.

**Q: How do I add a new chemical group?**
- Edit `data/chemical_groups.json` and add the group name to the `groups` list.

**Q: How do I view logs?**
- `backend.log` and `frontend.log` in the project root contain logs for each service.

**Q: How do I restart everything?**
- `bash RUN.sh restart` or stop then start manually as above.

---

For further questions, please check the code comments or contact the maintainer.
