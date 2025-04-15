"""Main entry point for the Chemical Groups Management System.

This script serves as the main entry point for the application,
providing information about the system and how to run it.
"""

import os
import subprocess
import sys
from typing import List, Optional


def main() -> None:
    """Display information about the application and how to run it.
    
    This function provides guidance on starting the application using RUN.sh
    rather than running this file directly.
    """
    print("Chemical Groups Management System")
    print("===================================\n")
    print("This is the main package for the web application.")
    print("To start the application, please run the RUN.sh script:")
    print("\n  $ bash RUN.sh\n")
    print("This will start both the backend API and frontend interface.")
    print("\nFor more information, please refer to the README.md file.")


if __name__ == "__main__":
    main()
