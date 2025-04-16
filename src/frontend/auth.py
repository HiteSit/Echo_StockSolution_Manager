"""Authentication module for the Chemical Groups Data Manager.

This module handles user authentication for the Streamlit frontend application
using streamlit-authenticator with JSON file-based credential storage.
"""

import streamlit as st
import streamlit_authenticator as stauth
import json
import os
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Constants
AUTH_FILE = "data/users.json"

def load_user_credentials():
    """Load user credentials from JSON file.
    
    Returns:
        dict: User credentials dictionary
    """
    auth_file_path = Path(AUTH_FILE)
    
    if auth_file_path.exists():
        try:
            with open(auth_file_path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {AUTH_FILE}. Using default admin credentials.")
    else:
        logger.info(f"{AUTH_FILE} not found. Creating with default admin credentials.")
        
    # Default admin credentials if file doesn't exist or is invalid
    default_credentials = {
        "credentials": {
            "usernames": {
                "admin": {
                    "name": "Admin User",
                    "email": "admin@example.com",
                    "role": "admin",
                    "password": stauth.Hasher(["adminpass123"]).generate()[0]
                },
                "user1": {
                    "name": "Regular User",
                    "email": "user@example.com",
                    "role": "user",
                    "password": stauth.Hasher(["userpass123"]).generate()[0]
                },
                "viewer1": {
                    "name": "View Only",
                    "email": "viewer@example.com",
                    "role": "viewer",
                    "password": stauth.Hasher(["viewerpass123"]).generate()[0]
                }
            }
        }
    }
    
    # Create parent directory if needed
    auth_file_path.parent.mkdir(exist_ok=True)
    
    # Save default credentials
    with open(auth_file_path, "w") as file:
        json.dump(default_credentials, file, indent=4)
    
    return default_credentials

def initialize_auth():
    """Initialize authentication object.
    
    Returns:
        tuple: (authenticator, user_credentials)
    """
    # Load credentials
    user_credentials = load_user_credentials()
    
    # Create an authentication object
    authenticator = stauth.Authenticate(
        credentials=user_credentials["credentials"],
        cookie_name="echo_stock_auth",
        key="echo_stock_manager",
        cookie_expiry_days=30
    )
    
    return authenticator, user_credentials

def check_authentication():
    """Check user authentication status.
    
    Returns:
        tuple: (is_authenticated, user_name, user_role, authenticator)
    """
    # Initialize authentication
    authenticator, user_credentials = initialize_auth()
    
    # Create a login widget
    name, authentication_status, username = authenticator.login("Login", "main")
    
    # Check authentication status
    if authentication_status == False:
        st.error("Username/password is incorrect")
        return False, None, None, authenticator
        
    elif authentication_status == None:
        st.warning("Please enter your username and password")
        return False, None, None, authenticator
        
    elif authentication_status:
        # User is authenticated
        # Get user role
        user_role = user_credentials["credentials"]["usernames"][username]["role"]
        
        # Store in session state for access throughout the app
        if "user_role" not in st.session_state:
            st.session_state["user_role"] = user_role
        if "username" not in st.session_state:
            st.session_state["username"] = username
        
        return True, name, user_role, authenticator
    
    return False, None, None, authenticator

def generate_password_hash(password: str) -> str:
    """Generate password hash for manual user management.
    
    Args:
        password: Plain text password to hash
        
    Returns:
        str: Hashed password
    """
    return stauth.Hasher([password]).generate()[0] 