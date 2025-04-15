"""Chemical Group CSV Upload Frontend App.

This Streamlit application handles the upload of CSV files for chemical groups, 
with validation and integration with the backend API.
"""

import logging
import os
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
    
    # Main workflow area
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


if __name__ == "__main__":
    main()
