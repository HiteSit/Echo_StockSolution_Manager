"""Backend API for Chemical Groups Management.

This module provides a FastAPI backend for managing chemical groups data.
It handles CSV uploads, validation, and merging into a master database.
"""

import json
import os
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# Import RDKit and related modules for volume calculation
from rdkit import Chem
from rdkit.Chem import Descriptors
import datamol as dm

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

# Add InChI to reference columns if needed but not present
# It should be positioned right after Smiles column
if "Smiles" in REFERENCE_COLUMNS and "InChI" not in REFERENCE_COLUMNS:
    smiles_idx = REFERENCE_COLUMNS.index("Smiles")
    REFERENCE_COLUMNS.insert(smiles_idx + 1, "InChI")
    logging.info("Added 'InChI' to reference columns for validation")

# Process existing master files at startup to fill missing volumes and InChI values
def process_existing_master_files():
    """Check all existing master CSV files and calculate any missing volume values and InChI values."""
    master_files = [f for f in os.listdir(DATA_DIR) if f.startswith("master_") and f.endswith(".csv")]
    
    if not master_files:
        logging.info("No existing master files found. Nothing to process for calculation.")
        return
    
    total_processed_volumes = 0
    total_processed_inchi = 0
    
    for file in master_files:
        file_path = os.path.join(DATA_DIR, file)
        try:
            # Read master file
            df = pd.read_csv(file_path)
            
            # Process InChI values
            if "InChI" not in df.columns and "Smiles" in df.columns:
                logging.info(f"No InChI column found in {file}. Calculating...")
                df = add_inchi(df)
                df.to_csv(file_path, index=False)
                total_processed_inchi += len(df)
                logging.info(f"Added InChI values to {len(df)} entries in {file}")
            
            # Check if there are missing volumes
            volume_col = [col for col in df.columns if col.startswith("Volume")]
            if not volume_col:
                logging.warning(f"No Volume column found in {file}. Skipping volume calculation.")
                continue
                
            volume_col = volume_col[0]
            missing_count = df[volume_col].isna().sum()
            
            if missing_count > 0:
                logging.info(f"Found {missing_count} missing volume values in {file}. Calculating...")
                
                # Calculate volumes for missing entries
                df_with_volumes = add_volume(df)
                
                # Save updated file
                df_with_volumes.to_csv(file_path, index=False)
                
                # Count successful calculations
                successful = missing_count - df_with_volumes[volume_col].isna().sum()
                total_processed_volumes += successful
                
                logging.info(f"Updated {successful} volume values in {file}")
            else:
                logging.info(f"No missing volume values in {file}. Skipping.")
                
        except Exception as e:
            logging.error(f"Error processing {file} for calculation: {str(e)}")
    
    if total_processed_volumes > 0 or total_processed_inchi > 0:
        logging.info(f"Startup calculation complete. Processed {total_processed_volumes} volume values and {total_processed_inchi} InChI values across all master files.")
    else:
        logging.info("No values needed processing.")

# Run the process at startup
process_existing_master_files()

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
    Handles special case for InChI column which may or may not be present in the upload.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if columns match reference, False otherwise
    """
    # Get list of columns from dataframe
    df_columns = list(df.columns)
    
    # Special case: If uploaded CSV doesn't have InChI but reference does, it's okay
    # We will calculate and add it during processing
    if "InChI" not in df_columns and "InChI" in REFERENCE_COLUMNS:
        # Remove InChI from the reference columns for comparison
        reference_without_inchi = [col for col in REFERENCE_COLUMNS if col != "InChI"]
        # Return True only if everything else matches
        return df_columns == reference_without_inchi
    
    # Otherwise, require exact match
    return df_columns == REFERENCE_COLUMNS


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
    Find IDs where row data (excluding ID, merge_timestamp and Volume) differs between master and upload.
    
    Args:
        master_df: DataFrame containing existing master data
        upload_df: DataFrame containing uploaded data
        
    Returns:
        List[str]: List of conflicting ID values
    """
    conflicts = []
    common_ids = set(master_df["ID"]).intersection(set(upload_df["ID"]))
    
    # Columns to ignore in conflict check
    ignore_columns = ["ID", "merge_timestamp"]
    # Find Volume column (if exists)
    volume_columns = [col for col in master_df.columns if col.startswith("Volume")]
    if volume_columns:
        ignore_columns.extend(volume_columns)
    
    for id_val in common_ids:
        master_row = master_df.loc[master_df["ID"] == id_val].iloc[0]
        upload_row = upload_df.loc[upload_df["ID"] == id_val].iloc[0]
        
        has_conflict = False
        for col in upload_df.columns:
            # Skip ignored columns
            if col in ignore_columns:
                continue
                
            # Handle null values - both null is considered equal
            if pd.isnull(master_row[col]) and pd.isnull(upload_row[col]):
                continue
                
            # Both values match
            if (not pd.isnull(master_row[col]) and 
                not pd.isnull(upload_row[col]) and 
                master_row[col] == upload_row[col]):
                continue
                
            # We found a real conflict
            has_conflict = True
            break
        
        if has_conflict:
            conflicts.append(id_val)
            
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


def append_merge_history(new_rows: pd.DataFrame, timestamp: str, chemical_group: str, uploader_name: str = None) -> None:
    """
    Append merge history to a log file for tracking.
    
    Args:
        new_rows: DataFrame containing new rows that were merged
        timestamp: ISO format timestamp of the merge
        chemical_group: Name of the chemical group
        uploader_name: Name of the user performing the upload
    """
    history_file = os.path.join(DATA_DIR, "merge_history.csv")
    
    # Create data for history
    history_data = {
        "timestamp": timestamp,
        "chemical_group": chemical_group,
        "num_rows": len(new_rows),
        "ids": ";".join(new_rows["ID"].astype(str).tolist()),
        "uploaded_by": uploader_name if uploader_name else "Unknown"
    }
    
    # Create or append to history file
    history_df = pd.DataFrame([history_data])
    
    if os.path.exists(history_file):
        history_df.to_csv(history_file, mode='a', header=False, index=False)
    else:
        history_df.to_csv(history_file, index=False)


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and add missing volume values using mass and concentration data.
    Uses SMILES from the dataframe to calculate molar mass using RDKit.
    Only calculates volumes for rows where Volume is NaN.
    
    Args:
        df: DataFrame containing Mass, Concentration, and SMILES columns
    
    Returns:
        DataFrame: DataFrame with calculated volumes
    """
    # Create a copy to avoid modifying the original dataframe
    df_result = df.copy()
    
    # Check if the required columns exist
    required_cols = ["Smiles", "Mass", "Conc", "Volume"]
    for col_prefix in required_cols:
        matching_cols = [col for col in df.columns if col.startswith(col_prefix)]
        if not matching_cols:
            logging.warning(f"Required column with prefix '{col_prefix}' not found. Cannot calculate volume.")
            return df_result
    
    # Extract column names and units using regex
    regex_pattern = r'\(([^)]+)\)'
    regex = re.compile(regex_pattern)
    
    # Get the actual column names
    mass_column = [col for col in df.columns if col.startswith("Mass")][0]
    concentration_column = [col for col in df.columns if col.startswith("Conc")][0]
    volume_column = [col for col in df.columns if col.startswith("Volume")][0]
    
    # Extract units from column names
    try:
        mass_unit = regex.search(mass_column).group(1)
        concentration_unit = regex.search(concentration_column).group(1)
        volume_unit = regex.search(volume_column).group(1)
        
        logging.info(f"Units detected: Mass: {mass_unit}, Concentration: {concentration_unit}, Volume: {volume_unit}")
    except (AttributeError, IndexError):
        logging.error("Failed to extract units from column names. Check column format (e.g., 'Mass (mg)').")
        return df_result
    
    # Unit conversion factors
    mass_to_g = 0.001 if mass_unit == 'mg' else 1  # Convert mg to g
    conc_to_mol_L = 1 if concentration_unit == 'M' else 1  # M is already mol/L
    L_to_target_vol = 1000000 if volume_unit == 'uL' else 1000 if volume_unit == 'mL' else 1  # Convert L to target volume unit
    
    # Calculate volume for rows with NaN in volume column
    volume_mask = df_result[volume_column].isna()
    
    if volume_mask.any():
        # Get rows with missing volume
        rows_to_calculate = df_result.loc[volume_mask]
        
        # Calculate molar mass for each compound using SMILES
        molar_masses = []
        for smiles in rows_to_calculate['Smiles']:
            try:
                # Convert SMILES to RDKit molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Calculate exact molecular weight in g/mol
                    molar_mass = Descriptors.ExactMolWt(mol)
                    molar_masses.append(molar_mass)
                else:
                    # If SMILES parsing fails, append NaN
                    molar_masses.append(np.nan)
                    logging.warning(f"Failed to parse SMILES: {smiles}")
            except Exception as e:
                molar_masses.append(np.nan)
                logging.error(f"Error calculating molar mass for {smiles}: {str(e)}")
        
        # Get mass and concentration values for rows with missing volume
        masses = rows_to_calculate[mass_column].values
        concentrations = rows_to_calculate[concentration_column].values
        
        # Calculate volume in liters: Mass (g) / (Concentration (mol/L) * Molar Mass (g/mol))
        volumes_L = []
        for mass, conc, mol_mass in zip(masses, concentrations, molar_masses):
            if np.isnan(mol_mass):
                volumes_L.append(np.nan)
            else:
                # Convert mass to grams
                mass_g = mass * mass_to_g
                # Calculate volume in liters
                volume_L = mass_g / (conc * mol_mass)
                volumes_L.append(volume_L)
        
        # Convert to target volume unit
        volumes_target_unit = [vol * L_to_target_vol if not np.isnan(vol) else np.nan for vol in volumes_L]
        
        # Round values to three decimal places
        volumes_target_unit = [round(vol, 3) if not np.isnan(vol) else np.nan for vol in volumes_target_unit]
        
        # Add calculated volumes to dataframe
        df_result.loc[volume_mask, volume_column] = volumes_target_unit
        
        # Count successful calculations
        successful_calcs = sum(~np.isnan(volumes_target_unit))
        logging.info(f"Calculated {successful_calcs} out of {volume_mask.sum()} missing volume values")
    else:
        logging.info("No missing volume values found")
    
    return df_result


def add_inchi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and add InChI values for all compounds in the dataframe.
    Uses SMILES from the dataframe to calculate InChI using datamol.
    
    Args:
        df: DataFrame containing a 'Smiles' column
    
    Returns:
        DataFrame: DataFrame with added or updated 'InChI' column
    """
    if 'Smiles' not in df.columns:
        logging.warning("Required column 'Smiles' not found. Cannot calculate InChI.")
        return df
    
    # Create a copy to avoid modifying the original dataframe
    df_result = df.copy()
    
    # Get column index of Smiles
    smiles_idx = df.columns.get_loc("Smiles")
    
    # Convert SMILES to InChI
    smiles_list = df["Smiles"].tolist()
    mols_list = [dm.to_mol(smiles) for smiles in smiles_list]
    inch_list = [dm.to_inchi(mol) for mol in mols_list]
    
    # Add or overwrite InChI column right after Smiles column
    if "InChI" in df.columns:
        df_result["InChI"] = inch_list
    else:
        df_result.insert(smiles_idx + 1, "InChI", inch_list)
        
    return df_result


def check_duplicate_inchi(df: pd.DataFrame) -> List[Tuple[str, List[str]]]:
    """
    Check for duplicate InChI values with different IDs in the dataframe.
    This indicates that the same chemical compound has been assigned multiple IDs.
    
    Args:
        df: DataFrame containing 'ID' and 'InChI' columns
        
    Returns:
        List[Tuple[str, List[str]]]: List of tuples with (duplicate_inchi, list_of_duplicate_ids)
    """
    if 'InChI' not in df.columns or 'ID' not in df.columns:
        logging.warning("Required columns 'InChI' or 'ID' not found. Cannot check for duplicates.")
        return []
    
    # Group by InChI and get corresponding IDs
    inchi_groups = df.groupby('InChI')['ID'].apply(list).reset_index()
    
    # Filter groups with more than one ID
    duplicates = inchi_groups[inchi_groups['ID'].apply(len) > 1]
    
    return [(row['InChI'], row['ID']) for _, row in duplicates.iterrows()]


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
    force: bool = Query(False, description="Force merge even with non-sequential IDs or conflicts"),
    uploader_name: str = Query(None, description="Name of the user who is uploading the file")
) -> Dict[str, Any]:
    """
    Upload a CSV for a specific chemical group, validate, compare with existing data,
    and merge into master_<Group>.csv if valid.
    
    Args:
        file: Uploaded CSV file
        chemical_group: Name of the chemical group to upload to
        force: Whether to force the merge despite warnings or conflicts
        uploader_name: Name of the user performing the upload
        
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
    
    # 5.1 Calculate InChI values for uploaded data
    df = add_inchi(df)
    
    # 5.2 Check for duplicate InChI values within the upload itself
    upload_duplicates = check_duplicate_inchi(df)
    if upload_duplicates and not force:
        # Collect all duplicated IDs
        all_duplicate_ids = []
        for _, ids in upload_duplicates:
            all_duplicate_ids.extend(ids)
        raise HTTPException(
            status_code=400, 
            detail=f"Duplicate compounds found in upload: {', '.join(all_duplicate_ids)}"
        )

    # 6. Load master for group
    if os.path.exists(master_csv):
        master_df = pd.read_csv(master_csv)
        
        # 6.1 Calculate InChI values for master data if needed
        if "InChI" not in master_df.columns:
            master_df = add_inchi(master_df)
        
        # 6.2 Check for duplicates between upload and master
        # Combine dataframes temporarily to check for duplicates
        combined_df = pd.concat([master_df, df], ignore_index=True)
        combined_duplicates = check_duplicate_inchi(combined_df)
        
        # Filter out duplicates that were already in either master or upload
        # (those were either previously allowed or just detected in step 5.2)
        new_duplicates = []
        master_inchi_to_id = dict(zip(master_df["InChI"], master_df["ID"]))
        upload_inchi_to_id = dict(zip(df["InChI"], df["ID"]))
        
        for inchi, ids in combined_duplicates:
            # Check if this duplicate involves both master and upload
            master_ids = [id for id in ids if id in master_df["ID"].values]
            upload_ids = [id for id in ids if id in df["ID"].values]
            
            if master_ids and upload_ids:
                new_duplicates.append((inchi, master_ids, upload_ids))
        
        if new_duplicates and not force:
            # Collect all duplicated IDs
            all_master_ids = []
            all_upload_ids = []
            for _, master_ids, upload_ids in new_duplicates:
                all_master_ids.extend(master_ids)
                all_upload_ids.extend(upload_ids)
            
            raise HTTPException(
                status_code=400,
                detail=f"Cross-file duplicate compounds. Upload IDs: {', '.join(all_upload_ids)}, Master IDs: {', '.join(all_master_ids)}"
            )
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

    # 10.3 Calculate volumes for new rows with missing values
    new_rows_with_volumes = add_volume(new_rows)
    
    # 11. Add merge_timestamp and append to master
    timestamp = datetime.now().isoformat()
    new_rows_with_volumes["merge_timestamp"] = timestamp
    
    # Add uploader name to new rows
    new_rows_with_volumes["uploaded_by"] = uploader_name if uploader_name else "Unknown"
    
    # Ensure master dataframe has uploaded_by column
    if "uploaded_by" not in master_df.columns:
        master_df["uploaded_by"] = None
        
    if "merge_timestamp" not in master_df.columns:
        master_df["merge_timestamp"] = None
        
    merged_df = pd.concat([master_df, new_rows_with_volumes], ignore_index=True)
    merged_df.to_csv(master_csv, index=False)

    # 12. Log merge history
    append_merge_history(new_rows_with_volumes, timestamp, chemical_group, uploader_name)

    return {
        "status": "ok",
        "message": f"Merged {len(new_rows_with_volumes)} new rows.",
        "diff": {
            "new": new_rows_with_volumes["ID"].tolist(),
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

