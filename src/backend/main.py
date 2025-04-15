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
# Get allowed origins from environment variables or use default for development
# Default now includes wildcard for network access
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501,http://*:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict to only necessary methods
    allow_headers=["Content-Type", "Authorization"],
)

# Constants and configuration
# Get the project root directory (two levels up from the backend module)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
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


def check_for_missing_ids(master_df: pd.DataFrame, upload_df: pd.DataFrame) -> Set[str]:
    """
    Check for missing IDs (IDs in master but not in upload).
    This is now informational rather than an error.
    
    Args:
        master_df: DataFrame containing existing master data
        upload_df: DataFrame containing uploaded data
        
    Returns:
        Set[str]: Set of IDs present in master but not in upload
    """
    return set(master_df["ID"]) - set(upload_df["ID"])


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
    # Find IDs present in upload but not in master
    master_ids = set(master_df["ID"])
    upload_ids = set(upload_df["ID"])
    new_ids = upload_ids - master_ids
    
    # Return the DataFrame with only those new IDs
    return upload_df[upload_df["ID"].isin(new_ids)].copy()


def check_sequential_ids(master_df: pd.DataFrame, new_rows: pd.DataFrame, group: str) -> dict:
    """
    Check if new IDs are sequential with the existing IDs in the master file.
    
    Args:
        master_df: DataFrame containing existing master data
        new_rows: DataFrame containing new rows to be added
        group: Chemical group name
        
    Returns:
        dict: Dictionary with keys 'sequential' (bool) and 'last_id' (str)
    """
    # If master is empty or no new rows, sequential by default
    if master_df.empty or new_rows.empty:
        return {"sequential": True, "last_id": None}
    
    # Determine ID prefix from group name
    id_prefix = group[:-1] if group.endswith("s") else group
    
    # Extract numbers from IDs using regex
    id_pattern = re.compile(rf"^{id_prefix}_(\d+)$")
    
    # Function to extract number from ID
    def extract_number(id_str):
        match = id_pattern.match(str(id_str))
        if match:
            return int(match.group(1))
        return None
    
    # Get all numeric parts of IDs
    master_numbers = [extract_number(id_val) for id_val in master_df["ID"] if extract_number(id_val) is not None]
    new_numbers = [extract_number(id_val) for id_val in new_rows["ID"] if extract_number(id_val) is not None]
    
    if not master_numbers or not new_numbers:
        return {"sequential": True, "last_id": None}
    
    # Find the highest ID in master
    highest_master = max(master_numbers)
    # Find the lowest ID in new rows
    lowest_new = min(new_numbers)
    
    # Check if sequential (lowest new should be highest master + 1)
    last_id = f"{id_prefix}_{highest_master}"
    return {
        "sequential": lowest_new == highest_master + 1,
        "last_id": last_id
    }


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
    chemical_group: str = Query(..., description="Chemical group to upload to (case-sensitive)"),
    force: bool = Query(False, description="Force merge even with non-sequential IDs or conflicts")
) -> Dict[str, Any]:
    """
    Upload a CSV for a specific chemical group, validate, compare with existing data,
    and merge into master_<Group>.csv if valid.
    
    Args:
        file: Uploaded CSV file
        chemical_group: Name of the chemical group to upload to
        force: Whether to force the merge despite warnings or conflicts
        
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
        # Read the file content into memory so we can reuse it
        file_content = file.file.read()
        # Create a BytesIO object from the content for pandas to read
        import io
        df = pd.read_csv(io.BytesIO(file_content))
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

    # 7. Check for missing IDs (for information only, not an error)
    missing_ids = check_for_missing_ids(master_df, df)

    # 8. Find conflicts
    conflicts = find_conflicts(master_df, df)
    
    # 8.1 Handle conflicts based on force parameter
    if conflicts and not force:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "error_type": "conflict",
            "detail": f"Conflicting IDs found: {sorted(conflicts)}",
            "diff": {"new": [], "conflicts": conflicts, "deletions": []}
        })
    elif conflicts and force:
        # Remove conflicting rows from master before merge
        master_df = master_df[~master_df["ID"].isin(conflicts)]
        
    # 9. Find new rows - need to recalculate after conflict removal
    new_ids = set(df["ID"]) - (set(master_df["ID"]) - set(conflicts))
    new_rows = df[df["ID"].isin(new_ids)].copy()

    # 10. If nothing to merge, return summary
    if new_rows.empty:
        return {
            "status": "ok", 
            "message": "No new rows to merge.", 
            "diff": {"new": [], "conflicts": [], "deletions": list(missing_ids)}
        }
        
    # 10.1 Check if IDs are sequential
    sequence_check = check_sequential_ids(master_df, new_rows, chemical_group)
    sequential = sequence_check["sequential"]
    last_id = sequence_check["last_id"]
    
    # 10.2 Return warning if not sequential and not forced
    if not sequential and not force:
        return JSONResponse(status_code=202, content={
            "status": "warning",
            "warning_type": "non_sequential_ids",
            "detail": f"New IDs are not sequential with existing IDs. Last ID in master: {last_id}",
            "last_id": last_id,
            "diff": {
                "new": new_rows["ID"].tolist(),
                "conflicts": [],
                "deletions": list(missing_ids)
            }
        })
    
    # If force=True, we proceed with the merge even if IDs are not sequential or conflicting

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
            "conflicts": conflicts if conflicts else [],
            "deletions": list(missing_ids)
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

