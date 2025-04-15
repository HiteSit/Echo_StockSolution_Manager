import os
import re
from typing import Any
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    Check if columns match the reference CSV exactly (names and order).
    """
    return list(df.columns) == REFERENCE_COLUMNS

def validate_ids(df: pd.DataFrame) -> None:
    """
    Validate ID column presence, uniqueness, and sequential pattern.
    Raise HTTPException if invalid.
    """
    if "ID" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing required 'ID' column.")
    if df["ID"].duplicated().any():
        raise HTTPException(status_code=400, detail="Duplicate IDs found in uploaded file.")
    id_pattern = re.compile(r"^Amine_\d+$")
    if not df["ID"].apply(lambda x: bool(id_pattern.match(str(x)))).all():
        raise HTTPException(status_code=400, detail="All IDs must follow the sequential pattern (e.g., Amine_1, Amine_2, ...).")

def check_no_deletions(master_df: pd.DataFrame, upload_df: pd.DataFrame) -> None:
    """
    Ensure no deletions: all master IDs must be present in upload.
    Raise HTTPException if any are missing.
    """
    missing_ids = set(master_df["ID"]) - set(upload_df["ID"])
    if missing_ids:
        raise HTTPException(status_code=400, detail=f"Upload is missing IDs present in master: {sorted(missing_ids)}")

def find_conflicts(master_df: pd.DataFrame, upload_df: pd.DataFrame) -> list:
    """
    Return list of IDs where row data (excluding ID) differs between master and upload.
    """
    conflicts = []
    common_ids = set(master_df["ID"]).intersection(set(upload_df["ID"]))
    for id_val in common_ids:
        master_row = master_df.loc[master_df["ID"] == id_val].iloc[0]
        upload_row = upload_df.loc[upload_df["ID"] == id_val].iloc[0]
        for col in upload_df.columns:
            if col == "ID":
                continue
            if pd.isnull(master_row[col]) and pd.isnull(upload_row[col]):
                continue
            if not pd.isnull(master_row[col]) and not pd.isnull(upload_row[col]) and master_row[col] == upload_row[col]:
                continue
            conflicts.append(id_val)
            break
    return conflicts

def get_new_rows(master_df: pd.DataFrame, upload_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame of new rows (IDs in upload not in master).
    """
    master_ids = set(master_df["ID"])
    upload_ids = set(upload_df["ID"])
    new_ids = upload_ids - master_ids
    return upload_df[upload_df["ID"].isin(new_ids)].copy()

def append_merge_history(new_rows: pd.DataFrame, timestamp: str) -> None:
    """
    Append a summary row to merge_history.csv: timestamp, num_rows_added, ids_added.
    """
    history_path = os.path.join(DATA_DIR, "merge_history.csv")
    num_rows = len(new_rows)
    ids_added = ','.join(str(i) for i in new_rows["ID"].tolist())
    summary_row = {
        "merge_timestamp": timestamp,
        "num_rows_added": num_rows,
        "ids_added": ids_added
    }
    import pandas as pd
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        history_df = pd.concat([history_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        history_df = pd.DataFrame([summary_row])
    history_df.to_csv(history_path, index=False)


from typing import List, Any
from fastapi.responses import JSONResponse
from datetime import datetime

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)) -> Any:
    """
    Upload a CSV, validate, diff, and merge into master if valid.
    Returns a diff summary and only merges if there are no conflicts or deletions.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    # 1. Schema validation
    if not validate_csv(df):
        raise HTTPException(status_code=400, detail="CSV schema does not match reference.")

    # 2. Validate IDs (presence, uniqueness, sequentiality)
    validate_ids(df)

    # 3. Load master
    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV)
    else:
        master_df = pd.DataFrame(columns=REFERENCE_COLUMNS)

    # 4. Assert no deletions
    check_no_deletions(master_df, df)

    # 5. Find conflicts
    conflicts = find_conflicts(master_df, df)
    if conflicts:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "error_type": "conflict",
            "detail": f"Conflicting IDs found: {sorted(conflicts)}"
        })

    # 6. Find new rows
    new_rows = get_new_rows(master_df, df)

    # If nothing to merge, return summary
    if new_rows.empty:
        return {"status": "ok", "message": "No new rows to merge.", "diff": {"new": [], "conflicts": [], "deletions": []}}

    # 7. Add merge_timestamp and append to master
    timestamp = datetime.now().isoformat()
    new_rows["merge_timestamp"] = timestamp
    if "merge_timestamp" not in master_df.columns:
        master_df["merge_timestamp"] = None
    merged_df = pd.concat([master_df, new_rows], ignore_index=True)
    merged_df.to_csv(MASTER_CSV, index=False)

    # 8. Log merge history
    append_merge_history(new_rows, timestamp)

    return {
        "status": "ok",
        "message": f"Merged {len(new_rows)} new rows.",
        "diff": {
            "new": new_rows["ID"].tolist(),
            "conflicts": [],
            "deletions": []
        }
    }

