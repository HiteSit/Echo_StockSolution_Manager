"""Test script for non-sequential ID handling with force parameter.

This script tests the fix for the issue where non-sequential IDs weren't being merged
even when the force parameter was set to true.
"""

import requests
import os
import json

def test_nonsequential_merge():
    """Test whether non-sequential IDs are correctly merged when force=True."""
    # Define test parameters
    url = "http://0.0.0.0:8000/upload_csv"
    chemical_group = "Amines"
    
    # Test file path - use our non-sequential CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(script_dir, "data", "NonSequential.csv")
    
    if not os.path.exists(test_file_path):
        print(f"Test file not found: {test_file_path}")
        return
    
    # First try without force - expect warning
    with open(test_file_path, "rb") as file:
        files = {"file": (os.path.basename(test_file_path), file, "text/csv")}
        params = {"chemical_group": chemical_group}
        
        print(f"Sending request WITHOUT force parameter")
        response = requests.post(url, files=files, params=params)
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 202:
            print("SUCCESS: Got expected warning about non-sequential IDs")
            result = response.json() 
            print(f"Warning type: {result.get('warning_type')}")
            print(f"New IDs: {result.get('diff', {}).get('new', [])}")
        else:
            print(f"FAILURE: Expected warning status 202, got {response.status_code}")
            print(response.text)
    
    # Now try with force=True - expect successful merge
    with open(test_file_path, "rb") as file:
        files = {"file": (os.path.basename(test_file_path), file, "text/csv")}
        params = {"chemical_group": chemical_group, "force": True}
        
        print(f"\nSending request WITH force=True")
        response = requests.post(url, files=files, params=params)
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            print("SUCCESS: Non-sequential IDs merged successfully")
            result = response.json()
            print(f"Message: {result.get('message')}")
            print(f"New IDs: {result.get('diff', {}).get('new', [])}")
        else:
            print(f"FAILURE: Expected success status 200, got {response.status_code}")
            print(response.text)

    # Now check if the IDs were actually added to the master file
    master_path = os.path.join(script_dir, "data", f"master_{chemical_group}.csv")
    if os.path.exists(master_path):
        import pandas as pd
        master_df = pd.read_csv(master_path)
        new_ids = ["Amine_20", "Amine_21", "Amine_22", "Amine_35"]
        
        print("\nChecking master file for new IDs:")
        for id_val in new_ids:
            if id_val in master_df["ID"].values:
                print(f"  ✅ {id_val} found in master file")
            else:
                print(f"  ❌ {id_val} NOT found in master file")
    else:
        print(f"\nERROR: Master file not found: {master_path}")

if __name__ == "__main__":
    test_nonsequential_merge() 