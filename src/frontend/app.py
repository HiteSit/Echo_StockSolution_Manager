"""Chemical Group CSV Upload Frontend App.

This Streamlit application handles the upload of CSV files for chemical groups, 
with validation and integration with the backend API.
"""

import logging
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union, Any

import requests
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration constants
# Allow configuring backend URL via environment variable
BACKEND_HOST = os.environ.get("BACKEND_HOST", "localhost")
BACKEND_PORT = os.environ.get("BACKEND_PORT", "8000")
BACKEND = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
APP_TITLE = "Chemical Groups Data Manager"
APP_DESCRIPTION = "Upload and manage chemical group data through CSV files"

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if not present
if "awaiting_confirmation" not in st.session_state:
    st.session_state.awaiting_confirmation = False
if "upload_data" not in st.session_state:
    st.session_state.upload_data = None
if "confirmation_type" not in st.session_state:
    st.session_state.confirmation_type = None


def main() -> None:
    """Main application entry point."""
    # Header section
    st.markdown(f"<h1 class='main-header'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>{APP_DESCRIPTION}</p>", unsafe_allow_html=True)
    
    # Status indicator for backend connectivity
    with st.sidebar:
        st.markdown("### System Status")
        check_backend_status()
        
        st.markdown("### Actions")
        if st.button("Refresh Chemical Groups", key="refresh_btn"):
            st.cache_data.clear()
            st.session_state.awaiting_confirmation = False
            st.session_state.upload_data = None
            st.rerun()
    
    # Fetch chemical groups
    groups, group_error = get_chemical_groups()
    
    if group_error:
        st.error(f"Backend Error: {group_error}")
        display_troubleshooting_info()
        return
        
    if not groups:
        st.warning("No chemical groups found. Please check backend configuration.")
        display_troubleshooting_info()
        return
    
    # Check if we're waiting for confirmation
    if st.session_state.awaiting_confirmation and st.session_state.upload_data:
        display_confirmation_dialog(st.session_state.upload_data)
        return
    
    # Create tabs for different functions
    upload_tab, visualize_tab = st.tabs(["Upload CSV", "Data Visualization Dashboard"])
    
    with upload_tab:
        display_upload_interface(groups)
    
    with visualize_tab:
        display_data_visualization_dashboard()


def display_upload_interface(groups: List[str]) -> None:
    """Display the CSV upload interface.
    
    Args:
        groups: List of available chemical groups
    """
    st.markdown("<h2 class='subheader'>Upload CSV Data</h2>", unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Group selection
        group = st.selectbox(
            "Select Chemical Group",
            options=groups,
            key="group_select"
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file with the correct format for the selected chemical group."
        )
    
    # Display selected information
    if uploaded_file and group:
        with col2:
            st.markdown("### Selected Information")
            st.info(
                f"**File:** {uploaded_file.name}\n\n"
                f"**Chemical Group:** {group}"
            )
            
            # Upload button
            if st.button("Process Upload", key="upload_btn", type="primary"):
                process_file_upload(uploaded_file, group)
    else:
        with col2:
            st.info("Please select a chemical group and upload a CSV file to continue.")
    
    # Instructions section
    with st.expander("CSV Format Instructions"):
        st.markdown("""
        ### CSV File Requirements
        
        Your CSV file must match the reference format with these requirements:
        
        1. **Columns**: Must match the reference column names and order exactly
        2. **ID Column**: Must follow the pattern `{Group}_#` (e.g., Amine_1, Amine_2)
        3. **Type Column**: Must match the chemical group name without the 's' suffix
        4. **No Deletions**: Cannot remove existing IDs from the master dataset
        
        Please ensure your data meets these requirements before uploading.
        """)


def display_confirmation_dialog(upload_data: Dict) -> None:
    """Display confirmation dialog for non-sequential IDs or conflicts.
    
    Args:
        upload_data: Dictionary containing file, group, and warning details
    """
    st.markdown("<h2 class='subheader'>Confirmation Required</h2>", unsafe_allow_html=True)
    
    file = upload_data["file"]
    chemical_group = upload_data["group"]
    confirmation_type = st.session_state.confirmation_type
    
    if confirmation_type == "non_sequential":
        # Non-sequential ID confirmation
        last_id = upload_data.get("last_id", "Unknown")
        new_ids = upload_data.get("new_ids", [])
        
        st.warning(f"⚠️ New IDs are not sequential with existing IDs. Last ID in master: {last_id}")
        
        if new_ids:
            st.info(f"New IDs to be added: **{', '.join(map(str, new_ids[:10]))}**")
            if len(new_ids) > 10:
                st.info(f"...and {len(new_ids) - 10} more new IDs")
    elif confirmation_type == "conflict":
        # Conflict confirmation
        conflicts = upload_data.get("conflicts", [])
        
        st.warning(f"⚠️ Conflicting IDs found. These entries will be overwritten if you proceed.")
        
        if conflicts:
            st.info(f"Conflicting IDs: **{', '.join(map(str, conflicts[:10]))}**")
            if len(conflicts) > 10:
                st.info(f"...and {len(conflicts) - 10} more conflicts")
    
    # Add confirmation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, proceed with merge", key="confirm_btn", type="primary"):
            # Execute the confirm merge
            confirm_merge(file, chemical_group, True)
            
            # Reset confirmation state
            st.session_state.awaiting_confirmation = False
            st.session_state.upload_data = None
            
    with col2:
        if st.button("❌ No, cancel merge", key="cancel_btn"):
            st.info("Merge cancelled. You can fix the issues and upload again.")
            
            # Reset confirmation state
            st.session_state.awaiting_confirmation = False
            st.session_state.upload_data = None


@st.cache_data(show_spinner=False, ttl=60)
def get_chemical_groups() -> Tuple[List[str], Optional[str]]:
    """Fetch chemical groups from the backend API.
    
    Returns:
        Tuple containing:
            - List of chemical group names (empty if error)
            - Error message (None if successful)
    """
    try:
        logger.info("Fetching chemical groups from backend")
        response = requests.get(f"{BACKEND}/chemical_groups", timeout=5)
        data = response.json()
        
        if response.status_code == 200 and "chemical_groups" in data:
            logger.info(f"Successfully retrieved {len(data['chemical_groups'])} chemical groups")
            return data["chemical_groups"], None
        else:
            error_msg = data.get("detail") or f"Status code: {response.status_code}"
            logger.error(f"Failed to get chemical groups: {error_msg}")
            return [], error_msg
    except requests.RequestException as e:
        logger.error(f"Connection error while fetching chemical groups: {str(e)}")
        return [], f"Connection error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error getting chemical groups: {str(e)}")
        return [], str(e)


def check_backend_status() -> None:
    """Check and display backend connection status."""
    try:
        response = requests.get(f"{BACKEND}/health", timeout=2)
        if response.status_code == 200:
            st.success("Backend: Connected")
            
            # Show additional backend connection details
            with st.expander("Connection Details"):
                st.code(f"Backend URL: {BACKEND}")
                st.code(f"Backend Host: {BACKEND_HOST}")
                st.code(f"Backend Port: {BACKEND_PORT}")
                
                # Show the response data if available
                try:
                    data = response.json()
                    st.code(f"Backend Response: {data}")
                except:
                    pass
        else:
            st.error(f"Backend: Unreachable (Invalid Response: {response.status_code})")
    except Exception as e:
        st.error(f"Backend: Unreachable (Connection Failed: {str(e)})")


def process_file_upload(file: Any, chemical_group: str) -> None:
    """Process the file upload to the backend.
    
    Args:
        file: The uploaded file object
        chemical_group: Selected chemical group name
    """
    with st.spinner("Processing upload..."):
        try:
            # Prepare the file upload
            files = {"file": (file.name, file, "text/csv")}
            params = {"chemical_group": chemical_group}
            
            # Send request to backend
            logger.info(f"Uploading CSV file for chemical group: {chemical_group}")
            response = requests.post(
                f"{BACKEND}/upload_csv",
                files=files, 
                params=params,
                timeout=30  # Longer timeout for file uploads
            )
            
            # Process the response
            if response.status_code == 200:
                # Standard successful response
                result = response.json()
                logger.info(f"Upload successful: {result.get('message')}")
                
                # Show success message with details
                st.success("✅ Upload and merge successful!")
                
                # Display results details
                display_upload_results(result)
                
            elif response.status_code == 202:
                # Warning response (non-sequential IDs)
                result = response.json()
                warning_type = result.get('warning_type')
                detail = result.get('detail')
                logger.warning(f"Upload warning: {detail} (Type: {warning_type})")
                
                if warning_type == "non_sequential_ids":
                    # Store data for confirmation
                    last_id = result.get('last_id', 'Unknown')
                    new_ids = result.get('diff', {}).get('new', [])
                    
                    # Save in session state for confirmation
                    st.session_state.awaiting_confirmation = True
                    st.session_state.confirmation_type = "non_sequential"
                    st.session_state.upload_data = {
                        "file": file,
                        "group": chemical_group,
                        "last_id": last_id,
                        "new_ids": new_ids
                    }
                    
                    # Force rerun to show confirmation dialog
                    st.rerun()
                            
            else:
                # Handle error response
                try:
                    response_json = response.json()
                    detail = response_json.get('detail') or str(response_json)
                    error_type = response_json.get('error_type', 'validation_error')
                except ValueError:
                    # Handle case when response is not valid JSON
                    detail = f"Invalid response (HTTP {response.status_code})"
                    error_type = "server_error"
                    logger.error(f"Response was not valid JSON: {response.text[:100]}")
                
                logger.error(f"Upload failed: {detail} (Type: {error_type})")
                st.error(f"Upload failed: {detail}")
                
                # If there are specific conflicts, show them
                if error_type == "conflict" and "diff" in response_json:
                    conflicts = response_json["diff"].get("conflicts", [])
                    if conflicts:
                        # Store data for confirmation
                        st.session_state.awaiting_confirmation = True
                        st.session_state.confirmation_type = "conflict"
                        st.session_state.upload_data = {
                            "file": file,
                            "group": chemical_group,
                            "conflicts": conflicts
                        }
                        
                        # Force rerun to show confirmation dialog
                        st.rerun()
        except requests.RequestException as e:
            logger.error(f"Connection error during upload: {str(e)}")
            st.error(f"Connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during upload: {str(e)}")
            st.error(f"Error: {str(e)}")


def confirm_merge(file: Any, chemical_group: str, force: bool) -> None:
    """Confirm the merge operation after warning about non-sequential IDs.
    
    Args:
        file: The uploaded file object
        chemical_group: Selected chemical group name
        force: Whether to force the merge despite warnings
    """
    with st.spinner("Processing confirmed merge..."):
        try:
            # Reset the file position to the beginning to reuse it
            file.seek(0)
            
            # Prepare the file upload with force parameter
            files = {"file": (file.name, file, "text/csv")}
            params = {"chemical_group": chemical_group, "force": force}
            
            # Send request to backend
            logger.info(f"Confirming upload for chemical group: {chemical_group} with force={force}")
            response = requests.post(
                f"{BACKEND}/upload_csv",
                files=files, 
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Confirmed upload successful: {result.get('message')}")
                
                # Show success message with details
                st.success("✅ Upload and merge successful!")
                
                # Display results details
                display_upload_results(result)
            else:
                # Handle error response
                try:
                    response_json = response.json()
                    detail = response_json.get('detail') or str(response_json)
                except ValueError:
                    detail = f"Invalid response (HTTP {response.status_code})"
                
                logger.error(f"Confirmed upload failed: {detail}")
                st.error(f"Upload failed: {detail}")
        except requests.RequestException as e:
            logger.error(f"Connection error during confirmation: {str(e)}")
            st.error(f"Connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during confirmation: {str(e)}")
            st.error(f"Error: {str(e)}")


def display_upload_results(result: Dict[str, Any]) -> None:
    """Display detailed upload results.
    
    Args:
        result: JSON response from the upload API
    """
    diff = result.get("diff", {})
    new_ids = diff.get("new", [])
    conflicts = diff.get("conflicts", [])
    missing_ids = diff.get("deletions", [])
    
    if new_ids:
        st.info(f"Added {len(new_ids)} new entries")
        
        if len(new_ids) <= 10:
            st.code(', '.join(map(str, new_ids)))
        else:
            st.code(', '.join(map(str, new_ids[:10])) + f"... and {len(new_ids) - 10} more")
    else:
        st.info("No new entries were added")
    
    if conflicts:
        st.info(f"Overwrote {len(conflicts)} conflicting entries")
        if len(conflicts) <= 5:
            st.code(', '.join(map(str, conflicts)))
    
    if missing_ids:
        st.info(f"This was a partial upload. There are {len(missing_ids)} IDs in the master file that were not in this upload.")
        if len(missing_ids) <= 5:
            missing_list = ', '.join(map(str, missing_ids))
            st.info(f"Missing IDs: {missing_list}")


def display_troubleshooting_info() -> None:
    """Display troubleshooting information when backend issues occur."""
    # Display backend URL for debugging
    st.warning(f"Current backend URL: {BACKEND}")
    
    with st.expander("Troubleshooting Steps"):
        st.markdown("""
        ### Troubleshooting Steps
        
        1. **Check if the backend server is running**
           - Verify that the backend process is active
           - Check backend.log for any error messages
        
        2. **Verify the chemical_groups.json file**
           - Ensure it exists in the data directory
           - Confirm it has the correct format: `{"groups": ["Group1", "Group2", ...]}`
        
        3. **Check network connectivity**
           - If using network access, ensure the backend is accessible from this machine
           - Check that CORS is properly configured in the backend
           - Verify that both machines can communicate (no firewall blocking)
        
        4. **Environment Variables**
           - You can set `BACKEND_HOST` and `BACKEND_PORT` to configure the backend connection
           - Example: `export BACKEND_HOST=192.168.1.100` before starting the app
        
        5. **Restart the application**
           - Stop both frontend and backend
           - Run the RUN.sh script again
        """)


def display_data_visualization_dashboard() -> None:
    """Display the data visualization dashboard for exploring master CSV files."""
    st.markdown("<h2 class='subheader'>Data Visualization Dashboard</h2>", unsafe_allow_html=True)
    
    # Initialize session state for visualizer if needed
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None
    if "current_df" not in st.session_state:
        st.session_state.current_df = None
    if "filtered_df" not in st.session_state:
        st.session_state.filtered_df = None
    
    # Get list of master CSV files
    master_csv_files = glob.glob(os.path.join("data", "master_*.csv"))
    if not master_csv_files:
        st.warning("No master CSV files found in the data directory.")
        return
    
    # Format file names for display (remove path and extension)
    file_display_names = [os.path.basename(f).replace("master_", "").replace(".csv", "") for f in master_csv_files]
    file_mapping = dict(zip(file_display_names, master_csv_files))
    
    # File selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_file_name = st.selectbox(
            "Select Chemical Group Data",
            options=file_display_names,
            key="viz_file_select"
        )
        
        if selected_file_name:
            file_path = file_mapping[selected_file_name]
            
            # Load the file if it's different from the currently loaded one
            if st.session_state.selected_file != file_path:
                try:
                    df = pd.read_csv(file_path)
                    st.session_state.current_df = df
                    st.session_state.filtered_df = df
                    st.session_state.selected_file = file_path
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    return
    
    # If no dataframe is loaded, return
    if st.session_state.current_df is None:
        return
    
    # Get current dataframe
    df = st.session_state.current_df
    
    # Dashboard sections using tabs
    viz_tabs = st.tabs(["Data Explorer", "Statistical Analysis", "Chemical Properties", "Timeline Analysis"])
    
    with viz_tabs[0]:  # Data Explorer
        display_data_explorer(df)
    
    with viz_tabs[1]:  # Statistical Analysis
        display_statistical_analysis(df)
    
    with viz_tabs[2]:  # Chemical Properties
        display_chemical_properties(df)
    
    with viz_tabs[3]:  # Timeline Analysis
        display_timeline_analysis(df)


def display_data_explorer(df: pd.DataFrame) -> None:
    """Display data explorer interface with filtering and searching capabilities.
    
    Args:
        df: DataFrame containing the master data
    """
    st.markdown("### Data Explorer")
    
    # Search & filter section
    with st.expander("Search & Filter", expanded=True):
        # Text search
        search_term = st.text_input("Search in all columns", key="search_input")
        
        # Apply initial text search filter
        filtered_df = df.copy()
        if search_term:
            mask = pd.Series(False, index=df.index)
            for col in df.columns:
                mask |= df[col].astype(str).str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        # Dynamic filtering based on column types
        st.markdown("#### Advanced Filters")
        
        # Create filter columns - adjust based on number of filters
        filter_cols = st.columns(3)
        col_idx = 0
        
        # Type filter (categorical)
        if 'Type' in df.columns:
            with filter_cols[col_idx % 3]:
                unique_types = sorted(df['Type'].unique().tolist())
                selected_types = st.multiselect(
                    "Filter by Type",
                    options=unique_types,
                    default=unique_types,
                    key="type_filter"
                )
                if selected_types and len(selected_types) < len(unique_types):
                    filtered_df = filtered_df[filtered_df['Type'].isin(selected_types)]
            col_idx += 1
        
        # Box filter (categorical if present)
        if 'Box' in df.columns:
            with filter_cols[col_idx % 3]:
                unique_boxes = sorted(df['Box'].dropna().unique().tolist())
                if len(unique_boxes) > 0 and len(unique_boxes) <= 20:  # Only show if reasonable number of values
                    selected_boxes = st.multiselect(
                        "Filter by Box",
                        options=unique_boxes,
                        default=unique_boxes,
                        key="box_filter"
                    )
                    if selected_boxes and len(selected_boxes) < len(unique_boxes):
                        filtered_df = filtered_df[filtered_df['Box'].isin(selected_boxes)]
            col_idx += 1
        
        # Numerical range filters (for numeric columns)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # Mass filter
        if 'Mass (mg)' in numeric_cols:
            with filter_cols[col_idx % 3]:
                min_val = float(df['Mass (mg)'].min())
                max_val = float(df['Mass (mg)'].max())
                if min_val < max_val:
                    mass_range = st.slider(
                        "Mass Range (mg)",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key="mass_filter"
                    )
                    filtered_df = filtered_df[
                        (filtered_df['Mass (mg)'] >= mass_range[0]) & 
                        (filtered_df['Mass (mg)'] <= mass_range[1])
                    ]
            col_idx += 1
        
        # Concentration filter
        if 'Conc (M)' in numeric_cols:
            with filter_cols[col_idx % 3]:
                min_val = float(df['Conc (M)'].min())
                max_val = float(df['Conc (M)'].max())
                if min_val < max_val:
                    conc_range = st.slider(
                        "Concentration Range (M)",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        key="conc_filter"
                    )
                    filtered_df = filtered_df[
                        (filtered_df['Conc (M)'] >= conc_range[0]) & 
                        (filtered_df['Conc (M)'] <= conc_range[1])
                    ]
            col_idx += 1
        
        # Volume filter
        if 'Volume (uL)' in numeric_cols:
            with filter_cols[col_idx % 3]:
                # Handle NaN values properly
                vol_data = df['Volume (uL)'].dropna()
                if not vol_data.empty:
                    min_val = float(vol_data.min())
                    max_val = float(vol_data.max())
                    if min_val < max_val:
                        vol_range = st.slider(
                            "Volume Range (uL)",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key="vol_filter"
                        )
                        filtered_df = filtered_df[
                            (filtered_df['Volume (uL)'].fillna(-1) >= vol_range[0]) & 
                            (filtered_df['Volume (uL)'].fillna(float('inf')) <= vol_range[1])
                        ]
            col_idx += 1
        
        # Add SMILES pattern filter if Smiles column exists
        if 'Smiles' in df.columns:
            with filter_cols[col_idx % 3]:
                # Create dropdown for common chemical patterns
                smiles_patterns = {
                    "All": None,
                    "Contains Nitrogen (N)": "N",
                    "Contains Oxygen (O)": "O",
                    "Contains Fluorine (F)": "F",
                    "Contains Chlorine (Cl)": "Cl",
                    "Contains Bromine (Br)": "Br",
                    "Contains Aromatic Ring": "c1" # Simple check for aromatic carbon
                }
                
                selected_pattern = st.selectbox(
                    "SMILES Pattern",
                    options=list(smiles_patterns.keys()),
                    key="smiles_pattern"
                )
                
                if selected_pattern != "All" and smiles_patterns[selected_pattern]:
                    pattern = smiles_patterns[selected_pattern]
                    filtered_df = filtered_df[filtered_df['Smiles'].str.contains(pattern, na=False)]
            col_idx += 1
        
        # Time filter if merge_timestamp column exists
        if 'merge_timestamp' in df.columns:
            with filter_cols[col_idx % 3]:
                # Convert to datetime for filtering
                df_time = df.copy()
                df_time['merge_date'] = pd.to_datetime(df_time['merge_timestamp'])
                
                min_date = df_time['merge_date'].min().date()
                max_date = df_time['merge_date'].max().date()
                
                if min_date != max_date:
                    date_range = st.date_input(
                        "Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="date_filter"
                    )
                    
                    # Apply date filter if a range is selected
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start_date, end_date = date_range
                        filtered_df = filtered_df[
                            (pd.to_datetime(filtered_df['merge_timestamp']).dt.date >= start_date) &
                            (pd.to_datetime(filtered_df['merge_timestamp']).dt.date <= end_date)
                        ]
            col_idx += 1
        
        # Set filtered dataframe in session state
        st.session_state.filtered_df = filtered_df
    
    # Show dataframe with sorting enabled
    total_rows = len(filtered_df)
    st.markdown(f"**Showing {total_rows} rows**")
    
    # Advanced table options
    table_height = st.slider("Table height", 300, 800, 500, key="table_height")
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=table_height,
        column_config={
            "ID": st.column_config.TextColumn("ID", width="medium"),
            "Smiles": st.column_config.TextColumn("Smiles", width="large"),
        }
    )
    
    # Download filtered data
    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_{os.path.basename(st.session_state.selected_file)}",
            mime="text/csv"
        )


def display_statistical_analysis(df: pd.DataFrame) -> None:
    """Display statistical analysis of numerical columns.
    
    Args:
        df: DataFrame containing the master data
    """
    st.markdown("### Statistical Analysis")
    
    # Get numerical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not numeric_cols:
        st.info("No numerical columns found for statistical analysis.")
        return
    
    # Basic statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    # Histogram plots for selected columns
    st.subheader("Distribution Analysis")
    
    selected_col = st.selectbox(
        "Select column for distribution analysis",
        options=numeric_cols,
        key="stat_column_select"
    )
    
    if selected_col:
        # Check if there are valid values
        valid_data = df[selected_col].dropna()
        
        if len(valid_data) > 0:
            # Distribution plot
            fig = px.histogram(
                df, 
                x=selected_col,
                title=f"Distribution of {selected_col}",
                nbins=20,
                opacity=0.7
            )
            fig.update_layout(
                xaxis_title=selected_col,
                yaxis_title="Count",
                bargap=0.05,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show additional statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean", f"{valid_data.mean():.2f}")
            col2.metric("Median", f"{valid_data.median():.2f}")
            col3.metric("Std Dev", f"{valid_data.std():.2f}")
        else:
            st.warning(f"No valid numerical data in column '{selected_col}'.")


def display_chemical_properties(df: pd.DataFrame) -> None:
    """Display analysis of chemical-specific properties.
    
    Args:
        df: DataFrame containing the master data
    """
    st.markdown("### Chemical Properties Visualization")
    
    # Check if we have Smiles column
    has_smiles = 'Smiles' in df.columns
    has_type = 'Type' in df.columns
    
    if not (has_smiles or has_type):
        st.info("No chemical-specific columns (Smiles, Type) found for analysis.")
        return
    
    # Type distribution if available
    if has_type:
        st.subheader("Chemical Type Distribution")
        
        type_counts = df['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        
        fig = px.pie(
            type_counts, 
            values='Count', 
            names='Type',
            title="Distribution of Chemical Types",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Smiles visualization if available
    if has_smiles:
        st.subheader("SMILES Analysis")
        
        # Calculate SMILES complexity (using length as a simple proxy)
        df_with_complexity = df.copy()
        df_with_complexity['SMILES_length'] = df['Smiles'].str.len()
        
        # Plot SMILES complexity
        fig = px.box(
            df_with_complexity,
            y='SMILES_length',
            points="all",
            title="SMILES Complexity Distribution (String Length)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show most complex molecules
        st.subheader("Top 5 Most Complex Molecules (by SMILES length)")
        complex_df = df_with_complexity.sort_values('SMILES_length', ascending=False).head(5)
        st.dataframe(
            complex_df[['ID', 'Type', 'Smiles', 'SMILES_length']],
            use_container_width=True
        )


def display_timeline_analysis(df: pd.DataFrame) -> None:
    """Display timeline analysis based on merge timestamps.
    
    Args:
        df: DataFrame containing the master data
    """
    st.markdown("### Timeline Analysis")
    
    # Check if we have merge_timestamp column
    if 'merge_timestamp' not in df.columns:
        st.info("No merge timestamp information available for timeline analysis.")
        return
    
    # Convert to datetime
    df_time = df.copy()
    df_time['merge_date'] = pd.to_datetime(df_time['merge_timestamp'])
    
    # Extract date components
    df_time['merge_year'] = df_time['merge_date'].dt.year
    df_time['merge_month'] = df_time['merge_date'].dt.month
    df_time['merge_day'] = df_time['merge_date'].dt.day
    df_time['merge_hour'] = df_time['merge_date'].dt.hour
    
    # Group by date
    st.subheader("Entries Added Over Time")
    
    # Select grouping level
    group_by = st.selectbox(
        "Group by",
        options=["Day", "Month", "Year"],
        index=0,
        key="timeline_group"
    )
    
    if group_by == "Day":
        group_col = ['merge_year', 'merge_month', 'merge_day']
        date_format = '%Y-%m-%d'
    elif group_by == "Month":
        group_col = ['merge_year', 'merge_month']
        date_format = '%Y-%m'
    else:  # Year
        group_col = ['merge_year']
        date_format = '%Y'
    
    # Count entries by date
    timeline_data = df_time.groupby(group_col).size().reset_index(name='count')
    
    # Create date string for display
    if group_by == "Day":
        timeline_data['date_str'] = timeline_data.apply(
            lambda x: f"{int(x['merge_year'])}-{int(x['merge_month']):02d}-{int(x['merge_day']):02d}", 
            axis=1
        )
    elif group_by == "Month":
        timeline_data['date_str'] = timeline_data.apply(
            lambda x: f"{int(x['merge_year'])}-{int(x['merge_month']):02d}", 
            axis=1
        )
    else:  # Year
        timeline_data['date_str'] = timeline_data['merge_year'].astype(str)
    
    # Sort by date
    timeline_data = timeline_data.sort_values('date_str')
    
    # Create timeline chart
    fig = px.bar(
        timeline_data,
        x='date_str',
        y='count',
        title=f"Number of Entries Added by {group_by}",
        labels={'date_str': 'Date', 'count': 'Number of Entries'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show entries from most recent date
    if not timeline_data.empty:
        latest_date = timeline_data['date_str'].iloc[-1]
        st.subheader(f"Most Recent Additions ({latest_date})")
        
        if group_by == "Day":
            latest_entries = df_time[
                (df_time['merge_year'] == timeline_data['merge_year'].iloc[-1]) &
                (df_time['merge_month'] == timeline_data['merge_month'].iloc[-1]) &
                (df_time['merge_day'] == timeline_data['merge_day'].iloc[-1])
            ]
        elif group_by == "Month":
            latest_entries = df_time[
                (df_time['merge_year'] == timeline_data['merge_year'].iloc[-1]) &
                (df_time['merge_month'] == timeline_data['merge_month'].iloc[-1])
            ]
        else:  # Year
            latest_entries = df_time[
                df_time['merge_year'] == timeline_data['merge_year'].iloc[-1]
            ]
        
        st.dataframe(latest_entries[df.columns], use_container_width=True)


if __name__ == "__main__":
    main()
