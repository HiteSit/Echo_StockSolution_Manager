#!/usr/bin/env python3
"""Password hash generator utility for user management.

This script generates bcrypt password hashes for use in the users.json file.
Run with a password argument to generate its hash.

Example:
    python password_hasher.py mypassword
"""

import argparse
import streamlit_authenticator as stauth

def hash_password(password: str) -> str:
    """Generate a secure hash for the given password.
    
    Args:
        password: Plain text password to hash
        
    Returns:
        str: Bcrypt hashed password
    """
    return stauth.Hasher([password]).generate()[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate password hash for users.json')
    parser.add_argument('password', type=str, help='Password to hash')
    args = parser.parse_args()
    
    hashed = hash_password(args.password)
    print(f"Hashed password: {hashed}") 