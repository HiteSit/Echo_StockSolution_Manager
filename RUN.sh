#!/bin/bash

# # Activate virtual environment if it exists
# if [ -d ".venv" ]; then
#     source .venv/bin/activate
#     echo "Activated virtual environment."
# else
#     echo ".venv not found, using system Python."
# fi

# # Install requirements
# pip install -r requirements.txt

# Start backend (FastAPI with uvicorn)
echo "Starting backend (FastAPI) on http://localhost:8000 ..."
nohup python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID $BACKEND_PID. Logs: backend.log"

# Start frontend (Streamlit)
echo "Starting frontend (Streamlit) on http://localhost:8501 ..."
nohup streamlit run frontend/app.py> frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID $FRONTEND_PID. Logs: frontend.log"

# Print summary
echo "---"
echo "Backend running at: http://localhost:8000"
echo "Frontend running at: http://localhost:8501"
echo "To stop: kill $BACKEND_PID $FRONTEND_PID"
