#!/bin/bash
#
# Chemical Groups Web Application Launcher
# This script launches both the FastAPI backend and Streamlit frontend
# with proper error handling and environment setup.
#

set -e  # Exit immediately if a command exits with non-zero status

# Configuration variables
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT=8000
FRONTEND_PORT=8501
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
PID_FILE="${APP_DIR}/app.pid"
DATA_DIR="${APP_DIR}/data"
CHEMICAL_GROUPS_FILE="${DATA_DIR}/chemical_groups.json"

# Text formatting
BOLD="\033[1m"
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RESET="\033[0m"

# Function to display colored messages
log_info() { echo -e "${BLUE}[INFO]${RESET} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${RESET} $1"; }
log_warn() { echo -e "${YELLOW}[WARNING]${RESET} $1"; }
log_error() { echo -e "${RED}[ERROR]${RESET} $1"; }

# Function to check if a port is in use
check_port() {
    local port=$1
    if command -v lsof &> /dev/null; then
        if lsof -i:"$port" &> /dev/null; then
            return 0  # Port is in use
        fi
    elif command -v netstat &> /dev/null; then
        if netstat -tuln | grep -q ":$port "; then
            return 0  # Port is in use
        fi
    fi
    return 1  # Port is not in use
}

# Function to find next available port starting from given port
find_available_port() {
    local start_port=$1
    local max_attempts=10
    local port=$start_port
    
    for (( i=0; i<max_attempts; i++ )); do
        if ! check_port "$port"; then
            echo $port
            return 0
        fi
        port=$((port + 1))
    done
    
    # If we get here, we couldn't find an available port
    log_error "Could not find available port after $max_attempts attempts starting from $start_port"
    return 1
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python installation
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.7 or higher."
        exit 1
    fi
    
    # Check if required ports are available, if not find alternatives
    if check_port "$BACKEND_PORT"; then
        log_warn "Port $BACKEND_PORT is already in use for backend."
        local new_port=$(find_available_port $((BACKEND_PORT+1)))
        if [ $? -eq 0 ]; then
            log_info "Switching to available port $new_port for backend."
            BACKEND_PORT=$new_port
        else
            log_error "Could not find an available port for backend."
            exit 1
        fi
    fi
    
    if check_port "$FRONTEND_PORT"; then
        log_warn "Port $FRONTEND_PORT is already in use for frontend."
        local new_port=$(find_available_port $((FRONTEND_PORT+1)))
        if [ $? -eq 0 ]; then
            log_info "Switching to available port $new_port for frontend."
            FRONTEND_PORT=$new_port
        else
            log_error "Could not find an available port for frontend."
            exit 1
        fi
    fi
    
    # Ensure data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        log_info "Creating data directory..."
        mkdir -p "$DATA_DIR"
    fi
    
    # Check for chemical_groups.json
    if [ ! -f "$CHEMICAL_GROUPS_FILE" ]; then
        log_warn "Chemical groups file not found. Creating default file."
        echo '{"groups": ["Amines", "Ethers", "Alcohols"]}' > "$CHEMICAL_GROUPS_FILE"
        log_info "Created default chemical_groups.json"
    fi
}

# Function to setup virtual environment
setup_environment() {
    log_info "Setting up environment..."
    
    # Activate the existing virtual environment
    if [ -d "$VENV_DIR" ]; then
        log_info "Activating existing virtual environment"
        source "$VENV_DIR/bin/activate"
        
        # Install requirements if needed using UV
        if [ -f "$REQUIREMENTS_FILE" ] && [ "$1" == "install" ]; then
            log_info "Installing requirements with UV"
            if command -v uv &> /dev/null; then
                uv pip install -r "$REQUIREMENTS_FILE"
            else
                log_warn "UV not found, but environment already exists. Continuing..."
            fi
        fi
    else
        log_error "Virtual environment not found at $VENV_DIR"
        log_info "Proceeding with system Python"
    fi
}

# Function to start the backend server
start_backend() {
    log_info "Starting backend (FastAPI) on http://localhost:$BACKEND_PORT ..."
    cd "$APP_DIR"
    nohup python3 -m uvicorn backend.main:app --host 0.0.0.0 --port "$BACKEND_PORT" > backend.log 2>&1 &
    BACKEND_PID=$!
    
    # Check if backend started successfully
    sleep 2
    if ps -p $BACKEND_PID > /dev/null; then
        log_success "Backend started with PID $BACKEND_PID. Logs: backend.log"
    else
        log_error "Backend failed to start. Check backend.log for details."
        exit 1
    fi
}

# Function to start the frontend server
start_frontend() {
    log_info "Starting frontend (Streamlit) on http://localhost:$FRONTEND_PORT ..."
    cd "$APP_DIR"
    nohup streamlit run frontend/app.py > frontend.log 2>&1 &
    FRONTEND_PID=$!
    
    # Check if frontend started successfully
    sleep 2
    if ps -p $FRONTEND_PID > /dev/null; then
        log_success "Frontend started with PID $FRONTEND_PID. Logs: frontend.log"
    else
        log_error "Frontend failed to start. Check frontend.log for details."
        kill $BACKEND_PID  # Clean up backend if frontend fails
        exit 1
    fi
}

# Function to save PIDs to file for later shutdown
save_pids() {
    echo "BACKEND_PID=$BACKEND_PID" > "$PID_FILE"
    echo "FRONTEND_PID=$FRONTEND_PID" >> "$PID_FILE"
    log_info "PIDs saved to ${PID_FILE}"
}

# Function to clean up on script termination
cleanup() {
    log_info "Cleaning up..."
    # Add any cleanup tasks here
}

# Register cleanup function
trap cleanup EXIT

# Main script execution
echo -e "\n${BOLD}Chemical Groups Web Application Launcher${RESET}\n"

# Function to stop any running services
stop_services() {
    # First try to use PID file
    if [ -f "$PID_FILE" ]; then
        source "$PID_FILE"
        log_info "Stopping services using PIDs from $PID_FILE..."
        kill $FRONTEND_PID $BACKEND_PID 2>/dev/null || true
        rm "$PID_FILE"
        log_success "Services stopped"
    else
        # Otherwise try to find and kill processes by port
        log_info "Stopping any services running on configured ports..."
        
        # Find and kill processes on BACKEND_PORT
        if check_port "$BACKEND_PORT"; then
            local backend_pid=$(lsof -ti:"$BACKEND_PORT" 2>/dev/null || netstat -tulpn 2>/dev/null | grep ":$BACKEND_PORT " | awk '{print $7}' | cut -d'/' -f1)
            if [ -n "$backend_pid" ]; then
                log_info "Killing process $backend_pid on port $BACKEND_PORT"
                kill $backend_pid 2>/dev/null || true
            fi
        fi
        
        # Find and kill processes on FRONTEND_PORT
        if check_port "$FRONTEND_PORT"; then
            local frontend_pid=$(lsof -ti:"$FRONTEND_PORT" 2>/dev/null || netstat -tulpn 2>/dev/null | grep ":$FRONTEND_PORT " | awk '{print $7}' | cut -d'/' -f1)
            if [ -n "$frontend_pid" ]; then
                log_info "Killing process $frontend_pid on port $FRONTEND_PORT"
                kill $frontend_pid 2>/dev/null || true
            fi
        fi
        
        # Wait a moment for processes to terminate
        sleep 2
        log_success "Services stopped or were not running"
    fi
}

# Check for arguments
if [ "$1" == "stop" ]; then
    # Stop any running services
    stop_services
    exit 0
elif [ "$1" == "restart" ]; then
    # Restart services (stop and then start)
    stop_services
    # Continue with startup (fallthrough)
elif [[ "$1" == --backend-port=* ]]; then
    # Set custom backend port
    BACKEND_PORT=${1#*=}
    shift
elif [[ "$1" == --frontend-port=* ]]; then
    # Set custom frontend port
    FRONTEND_PORT=${1#*=}
    shift
fi


# Stop any existing services first
log_info "Checking for existing services..."
stop_services

# Normal startup
check_prerequisites

# Check if install flag is provided
if [ "$1" == "install" ]; then
    setup_environment "install"
else
    setup_environment
fi

# Display ports being used
log_info "Using backend port: $BACKEND_PORT"
log_info "Using frontend port: $FRONTEND_PORT"

start_backend
start_frontend
save_pids

# Print summary
echo -e "\n${BOLD}-------------------------------------${RESET}"
log_success "Services started successfully!"
echo -e "${BOLD}Backend:${RESET} http://localhost:$BACKEND_PORT"
echo -e "${BOLD}Frontend:${RESET} http://localhost:$FRONTEND_PORT"
echo -e "\nTo stop the application, run: $0 stop"
echo -e "${BOLD}-------------------------------------${RESET}\n"
