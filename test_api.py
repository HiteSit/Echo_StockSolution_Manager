"""Test script for checking the force parameter behavior with FastAPI backend.

This script sends a test request to the backend with force=True to verify
how parameter conversion is working.
"""

import requests
import os

def test_force_param():
    """Test whether the force parameter is correctly handled by FastAPI."""
    # Define test parameters
    url = "http://0.0.0.0:8000/upload_csv"
    chemical_group = "Amines"
    
    # Test file path - use a sample CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(script_dir, "data", "Example.csv")
    
    if not os.path.exists(test_file_path):
        print(f"Test file not found: {test_file_path}")
        return
    
    # Option 1: Send using Python boolean
    with open(test_file_path, "rb") as file:
        files = {"file": (os.path.basename(test_file_path), file, "text/csv")}
        params = {"chemical_group": chemical_group, "force": True}
        
        print(f"Sending request with params={params}")
        response = requests.post(url, files=files, params=params)
        
        print(f"Response status: {response.status_code}")
        print(f"Response JSON: {response.json() if response.status_code < 300 else response.text}")

    # Option 2: Send using string "true"
    with open(test_file_path, "rb") as file:
        files = {"file": (os.path.basename(test_file_path), file, "text/csv")}
        params = {"chemical_group": chemical_group, "force": "true"}
        
        print(f"\nSending request with params={params}")
        response = requests.post(url, files=files, params=params)
        
        print(f"Response status: {response.status_code}")
        print(f"Response JSON: {response.json() if response.status_code < 300 else response.text}")

if __name__ == "__main__":
    test_force_param() 