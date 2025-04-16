#!/usr/bin/env python3
"""User management utility for Chemical Groups Data Manager.

This script provides utilities for managing users in the users.json file.
It can create a new users.json file with default users or add/remove/update users.

Examples:
    # Create default users.json with admin, user1, viewer1
    python manage_users.py --init
    
    # Add a new user
    python manage_users.py --add-user username name email role password
    
    # Remove a user
    python manage_users.py --remove-user username
    
    # Update user role
    python manage_users.py --update-role username new_role
    
    # Update user password
    python manage_users.py --update-password username new_password
"""

import argparse
import json
import os
import streamlit_authenticator as stauth
from pathlib import Path

# Constants
AUTH_FILE = "data/users.json"

def create_default_users_file():
    """Create default users.json file with admin, user and viewer roles."""
    default_credentials = {
        "credentials": {
            "usernames": {
                "admin": {
                    "name": "Admin_User",
                    "email": "admin@example.com",
                    "role": "admin",
                    "password": stauth.Hasher(["adminpass123"]).generate()[0]
                },
                "user1": {
                    "name": "Regular_User",
                    "email": "user@example.com",
                    "role": "user",
                    "password": stauth.Hasher(["userpass123"]).generate()[0]
                },
                "viewer1": {
                    "name": "View_Only",
                    "email": "viewer@example.com",
                    "role": "viewer",
                    "password": stauth.Hasher(["viewerpass123"]).generate()[0]
                }
            }
        }
    }
    
    # Create parent directory if needed
    auth_file_path = Path(AUTH_FILE)
    auth_file_path.parent.mkdir(exist_ok=True)
    
    # Save credentials
    with open(auth_file_path, "w") as file:
        json.dump(default_credentials, file, indent=4)
    
    print(f"Created default users file at {AUTH_FILE}")
    print("Default users:")
    print("  - admin / adminpass123 (admin role)")
    print("  - user1 / userpass123 (user role)")
    print("  - viewer1 / viewerpass123 (viewer role)")

def load_user_credentials():
    """Load user credentials from JSON file."""
    auth_file_path = Path(AUTH_FILE)
    
    if auth_file_path.exists():
        try:
            with open(auth_file_path, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {AUTH_FILE}")
            return None
    else:
        print(f"File {AUTH_FILE} not found")
        return None

def save_user_credentials(credentials):
    """Save user credentials to JSON file."""
    with open(AUTH_FILE, "w") as file:
        json.dump(credentials, file, indent=4)
    print(f"Updated {AUTH_FILE}")

def add_user(username, name, email, role, password):
    """Add a new user to the credentials file."""
    credentials = load_user_credentials()
    if not credentials:
        print("Error loading credentials file")
        return
    
    if username in credentials["credentials"]["usernames"]:
        print(f"User '{username}' already exists")
        return
    
    if role not in ["admin", "user", "viewer"]:
        print(f"Invalid role: {role}. Must be 'admin', 'user', or 'viewer'")
        return
    
    # Hash password
    hashed_password = stauth.Hasher([password]).generate()[0]
    
    # Add user
    credentials["credentials"]["usernames"][username] = {
        "name": name,
        "email": email,
        "role": role,
        "password": hashed_password
    }
    
    save_user_credentials(credentials)
    print(f"Added user '{username}' with role '{role}'")

def remove_user(username):
    """Remove a user from the credentials file."""
    credentials = load_user_credentials()
    if not credentials:
        print("Error loading credentials file")
        return
    
    if username not in credentials["credentials"]["usernames"]:
        print(f"User '{username}' not found")
        return
    
    # Remove user
    del credentials["credentials"]["usernames"][username]
    
    save_user_credentials(credentials)
    print(f"Removed user '{username}'")

def update_role(username, new_role):
    """Update a user's role."""
    credentials = load_user_credentials()
    if not credentials:
        print("Error loading credentials file")
        return
    
    if username not in credentials["credentials"]["usernames"]:
        print(f"User '{username}' not found")
        return
    
    if new_role not in ["admin", "user", "viewer"]:
        print(f"Invalid role: {new_role}. Must be 'admin', 'user', or 'viewer'")
        return
    
    # Update role
    credentials["credentials"]["usernames"][username]["role"] = new_role
    
    save_user_credentials(credentials)
    print(f"Updated role for '{username}' to '{new_role}'")

def update_password(username, new_password):
    """Update a user's password."""
    credentials = load_user_credentials()
    if not credentials:
        print("Error loading credentials file")
        return
    
    if username not in credentials["credentials"]["usernames"]:
        print(f"User '{username}' not found")
        return
    
    # Hash password
    hashed_password = stauth.Hasher([new_password]).generate()[0]
    
    # Update password
    credentials["credentials"]["usernames"][username]["password"] = hashed_password
    
    save_user_credentials(credentials)
    print(f"Updated password for '{username}'")

def list_users():
    """List all users in the credentials file."""
    credentials = load_user_credentials()
    if not credentials:
        print("Error loading credentials file")
        return
    
    print(f"Users in {AUTH_FILE}:")
    for username, data in credentials["credentials"]["usernames"].items():
        print(f"  - {username} ({data['name']}, {data['email']}): {data['role']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage users for Echo Stock Solution Manager')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--init', action='store_true', help='Create default users.json file')
    group.add_argument('--list', action='store_true', help='List all users')
    group.add_argument('--add-user', nargs=5, metavar=('USERNAME', 'NAME', 'EMAIL', 'ROLE', 'PASSWORD'),
                       help='Add a new user')
    group.add_argument('--remove-user', metavar='USERNAME', help='Remove a user')
    group.add_argument('--update-role', nargs=2, metavar=('USERNAME', 'ROLE'),
                       help='Update a user\'s role')
    group.add_argument('--update-password', nargs=2, metavar=('USERNAME', 'PASSWORD'),
                       help='Update a user\'s password')
    
    args = parser.parse_args()
    
    if args.init:
        create_default_users_file()
    elif args.list:
        list_users()
    elif args.add_user:
        add_user(*args.add_user)
    elif args.remove_user:
        remove_user(args.remove_user)
    elif args.update_role:
        update_role(*args.update_role)
    elif args.update_password:
        update_password(*args.update_password) 