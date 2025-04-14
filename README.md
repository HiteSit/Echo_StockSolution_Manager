# Streamlit CSV Upload App

This application allows multiple users to upload CSV files via a Streamlit interface. Uploaded files are validated by the backend and, if valid, concatenated to a master CSV file.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the backend:
   ```bash
   uvicorn backend.main:app --reload
   ```

3. Start the frontend:
   ```bash
   streamlit run frontend/app.py
   ```

## Backend Validation

The backend exposes an endpoint `/upload_csv` that accepts CSV files. A placeholder function `validate_csv(df)` is provided for custom validation logic.

## File Storage

Uploaded and validated CSVs are concatenated and stored in `data/master.csv`.
