import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# Allow Streamlit frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MASTER_CSV = os.path.join(DATA_DIR, 'master.csv')

os.makedirs(DATA_DIR, exist_ok=True)

# Load reference columns from Example_Amines.csv at startup
REFERENCE_CSV = os.path.join(DATA_DIR, 'Example_Amines.csv')
if not os.path.exists(REFERENCE_CSV):
    raise RuntimeError(f"Reference CSV not found: {REFERENCE_CSV}")
REFERENCE_COLUMNS = list(pd.read_csv(REFERENCE_CSV, nrows=0).columns)

def validate_csv(df: pd.DataFrame) -> bool:
    """
    Check if columns match the reference CSV exactly.
    Extend this with more validation logic as needed.
    """
    return list(df.columns) == REFERENCE_COLUMNS

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")
    if not validate_csv(df):
        raise HTTPException(status_code=400, detail="CSV validation failed.")
    # Concatenate to master CSV
    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV)
        master_df = pd.concat([master_df, df], ignore_index=True)
    else:
        master_df = df
    master_df.to_csv(MASTER_CSV, index=False)
    return {"message": "CSV uploaded and concatenated successfully."}
