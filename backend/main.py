"""Backend API for Chemical Groups Management.

This module provides a FastAPI backend for managing chemical groups data.
It handles CSV uploads, validation, and merging into a master database.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Chemical Groups API",
    description="API for managing and merging chemical group data",
    version="1.0.0",
)

# Configure CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants and configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
CHEMICAL_GROUPS_FILE = os.path.join(DATA_DIR, "chemical_groups.json")
REFERENCE_CSV = os.path.join(DATA_DIR, "Example.csv")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Check for reference CSV at startup
if not os.path.exists(REFERENCE_CSV):
    raise RuntimeError(f"Reference CSV not found: {REFERENCE_CSV}")

# Load reference columns once at startup
REFERENCE_COLUMNS = list(pd.read_csv(REFERENCE_CSV, nrows=0).columns)


# Pydantic models for API responses
class ChemicalGroupsResponse(BaseModel):
    """Response model for chemical groups endpoint."""
    chemical_groups: List[str]


class DiffSummary(BaseModel):
    """Model for describing changes between master and uploaded data."""
    new: List[str]
    conflicts: List[str]
    deletions: List[str]


class UploadResponse(BaseModel):
    """Response model for CSV upload endpoint."""
    status: str
    message: str
    diff: DiffSummary


# Helper functions
def load_chemical_groups() -> List[str]:
    """
    Load chemical groups from JSON configuration file.
    
    Returns:
        List[str]: List of chemical group names
        
    Raises:
        RuntimeError: If file not found or contains invalid data
    """
    if not os.path.exists(CHEMICAL_GROUPS_FILE):
        raise RuntimeError(f"Chemical groups file not found: {CHEMICAL_GROUPS_FILE}")
    
    try:
        with open(CHEMICAL_GROUPS_FILE, "r") as f:
            data = json.load(f)
        
        groups = data.get("groups", [])
        if not isinstance(groups, list):
            raise ValueError("The 'groups' key must be a list.")
        
        return groups
    except Exception as e:
        raise RuntimeError(f"Error loading chemical groups JSON: {e}")


def validate_csv(df: pd.DataFrame) -> bool:
    """
    Check if DataFrame columns match the reference CSV exactly (names and order).
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if columns match reference, False otherwise
    """
    return list(df.columns) == REFERENCE_COLUMNS


def validate_ids_and_type(df: pd.DataFrame, group: str) -> None:
    """
    Validate ID column presence, uniqueness, and sequential pattern for group.
    Also validate Type column matches group (case-sensitive).
    
    Args:
        df: DataFrame to validate
        group: Chemical group name to check against
        
    Raises:
        HTTPException: If validation fails with details about the issue
    """
    if "ID" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing required 'ID' column.")
    
    if "Type" not in df.columns:
        raise HTTPException(status_code=400, detail="Missing required 'Type' column.")
    
    if df["ID"].duplicated().any():
        raise HTTPException(status_code=400, detail="Duplicate IDs found in uploaded file.")
    
    # Determine ID prefix from group name
    id_prefix = group[:-1] if group.endswith("s") else group
    id_pattern = re.compile(rf"^{id_prefix}_\d+$")
    
    # Validate ID format
    if not df["ID"].apply(lambda x: bool(id_pattern.match(str(x)))).all():
        raise HTTPException(
            status_code=400, 
            detail=f"All IDs must start with '{id_prefix}_' and be sequential (e.g., {id_prefix}_1, {id_prefix}_2, ...)."
        )
    
    # Validate Type column values
    if not (df["Type"] == id_prefix).all():
        raise HTTPException(
            status_code=400,
            detail=f"All 'Type' values must be '{id_prefix}' for group '{group}'."
        )


def check_no_deletions(master_df: pd.DataFrame, upload_df: pd.DataFrame) -> None:
    """
    Ensure no deletions occurred: all master IDs must be present in upload.
    
    Args:
        master_df: DataFrame containing existing master data
        upload_df: DataFrame containing uploaded data
        
    Raises:
        HTTPException: If any IDs are missing from upload
    """
    missing_ids = set(master_df["ID"]) - set(upload_df["ID"])
    if missing_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Upload is missing IDs present in master: {sorted(missing_ids)}"
        )


def find_conflicts(master_df: pd.DataFrame, upload_df: pd.DataFrame) -> List[str]:
    """
    Find IDs where row data (excluding ID) differs between master and upload.
    
    Args:
        master_df: DataFrame containing existing master data
        upload_df: DataFrame containing uploaded data
        
    Returns:
        List[str]: List of conflicting ID values
    """
    conflicts = []
    common_ids = set(master_df["ID"]).intersection(set(upload_df["ID"]))
    
    for id_val in common_ids:
        master_row = master_df.loc[master_df["ID"] == id_val].iloc[0]
        upload_row = upload_df.loc[upload_df["ID"] == id_val].iloc[0]
        
        for col in upload_df.columns:
            if col == "ID":
                continue
                
            # Handle null values
            if pd.isnull(master_row[col]) and pd.isnull(upload_row[col]):
                continue
                
            # Check for value equality
            if (not pd.isnull(master_row[col]) and 
                not pd.isnull(upload_row[col]) and 
                master_row[col] == upload_row[col]):
                continue
                
            conflicts.append(id_val)
            break
            
    return conflicts


def get_new_rows(master_df: pd.DataFrame, upload_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract new rows from upload (IDs not in master).
    
    Args:
        master_df: DataFrame containing existing master data
        upload_df: DataFrame containing uploaded data
        
    Returns:
        pd.DataFrame: DataFrame containing only new rows
    """
    master_ids = set(master_df["ID"])
    upload_ids = set(upload_df["ID"])
    new_ids = upload_ids - master_ids
    return upload_df[upload_df["ID"].isin(new_ids)].copy()


def append_merge_history(new_rows: pd.DataFrame, timestamp: str, chemical_group: str) -> None:
    """
    Append a summary row to merge_history.csv with details about the merge operation.
    
    Args:
        new_rows: DataFrame containing newly added rows
        timestamp: ISO-formatted timestamp for the merge operation
        chemical_group: Name of the chemical group being merged
    """
    history_path = os.path.join(DATA_DIR, "merge_history.csv")
    num_rows = len(new_rows)
    ids_added = ",".join(str(i) for i in new_rows["ID"].tolist())
    
    summary_row = {
        "chemical_group": chemical_group,
        "merge_timestamp": timestamp,
        "num_rows_added": num_rows,
        "ids_added": ids_added
    }
    
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        history_df = pd.concat([history_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        history_df = pd.DataFrame([summary_row])
        
    history_df.to_csv(history_path, index=False)


# API Routes
@app.get("/chemical_groups", response_model=ChemicalGroupsResponse)
async def get_chemical_groups() -> Dict[str, List[str]]:
    """
    Retrieve the list of available chemical groups.
    
    Returns:
        Dict containing the list of chemical group names
    """
    try:
        groups = load_chemical_groups()
        return {"chemical_groups": groups}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading chemical groups: {str(e)}")


@app.post("/upload_csv", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(...),
    chemical_group: str = Query(..., description="Chemical group to upload to (case-sensitive)")
) -> Dict[str, Any]:
    """
    Upload a CSV for a specific chemical group, validate, compare with existing data,
    and merge into master_<Group>.csv if valid.
    
    Args:
        file: Uploaded CSV file
        chemical_group: Name of the chemical group to upload to
        
    Returns:
        Dict containing status, message, and diff summary
        
    Raises:
        HTTPException: For various validation errors
    """
    # 1. Validate chemical group
    try:
        groups = load_chemical_groups()
        if chemical_group not in groups:
            raise HTTPException(status_code=400, detail=f"Unknown chemical group: {chemical_group}")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    master_csv = os.path.join(DATA_DIR, f"master_{chemical_group}.csv")

    # 2. Validate file format
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    # 3. Read and validate CSV content
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    # 4. Schema validation
    if not validate_csv(df):
        raise HTTPException(status_code=400, detail="CSV schema does not match reference.")

    # 5. Validate IDs and Type for group
    validate_ids_and_type(df, chemical_group)

    # 6. Load master for group
    if os.path.exists(master_csv):
        master_df = pd.read_csv(master_csv)
    else:
        master_df = pd.DataFrame(columns=REFERENCE_COLUMNS)

    # 7. Assert no deletions
    check_no_deletions(master_df, df)

    # 8. Find conflicts
    conflicts = find_conflicts(master_df, df)
    if conflicts:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "error_type": "conflict",
            "detail": f"Conflicting IDs found: {sorted(conflicts)}",
            "diff": {"new": [], "conflicts": conflicts, "deletions": []}
        })

    # 9. Find new rows
    new_rows = get_new_rows(master_df, df)

    # 10. If nothing to merge, return summary
    if new_rows.empty:
        return {
            "status": "ok", 
            "message": "No new rows to merge.", 
            "diff": {"new": [], "conflicts": [], "deletions": []}
        }

    # 11. Add merge_timestamp and append to master
    timestamp = datetime.now().isoformat()
    new_rows["merge_timestamp"] = timestamp
    
    if "merge_timestamp" not in master_df.columns:
        master_df["merge_timestamp"] = None
        
    merged_df = pd.concat([master_df, new_rows], ignore_index=True)
    merged_df.to_csv(master_csv, index=False)

    # 12. Log merge history
    append_merge_history(new_rows, timestamp, chemical_group)

    return {
        "status": "ok",
        "message": f"Merged {len(new_rows)} new rows.",
        "diff": {
            "new": new_rows["ID"].tolist(),
            "conflicts": [],
            "deletions": []
        }
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint.
    
    Returns:
        Dict with status information
    """
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

