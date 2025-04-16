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

# Function to kill a process using a specific port
kill_process_on_port() {
    local port=$1
    local pid=""
    
    # Try to find PID using lsof
    if command -v lsof &> /dev/null; then
        pid=$(lsof -ti:"$port" 2>/dev/null)
    # If lsof failed or isn't available, try netstat
    elif command -v netstat &> /dev/null; then
        pid=$(netstat -tulpn 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1)
    fi
    
    # If we found a PID, try to kill it
    if [ -n "$pid" ]; then
        log_warn "Killing process $pid that is using port $port"
        kill $pid 2>/dev/null
        
        # Wait briefly to ensure the process is terminated
        sleep 1
        
        # Check if the port is now available
        if ! check_port "$port"; then
            log_success "Successfully freed port $port"
            return 0  # Successfully killed process
        else
            log_error "Failed to free port $port, process may still be running"
        fi
    else
        log_error "Could not identify process using port $port"
    fi
    
    return 1  # Failed to kill process or free port
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
    
    # Check if required ports are available, if not kill processes on those ports
    if check_port "$BACKEND_PORT"; then
        log_warn "Port $BACKEND_PORT is already in use for backend."
        
        # Try to kill the process using this port
        if kill_process_on_port "$BACKEND_PORT"; then
            log_info "Will use the original port $BACKEND_PORT for backend."
        else
            # Fallback: find an alternative port if we couldn't kill the process
            log_warn "Could not kill process on port $BACKEND_PORT. Finding alternative port..."
            local new_port=$(find_available_port $((BACKEND_PORT+1)))
            if [ $? -eq 0 ]; then
                log_info "Switching to available port $new_port for backend."
                BACKEND_PORT=$new_port
            else
                log_error "Could not find an available port for backend."
                exit 1
            fi
        fi
    fi
    
    if check_port "$FRONTEND_PORT"; then
        log_warn "Port $FRONTEND_PORT is already in use for frontend."
        
        # Try to kill the process using this port
        if kill_process_on_port "$FRONTEND_PORT"; then
            log_info "Will use the original port $FRONTEND_PORT for frontend."
        else
            # Fallback: find an alternative port if we couldn't kill the process
            log_warn "Could not kill process on port $FRONTEND_PORT. Finding alternative port..."
            local new_port=$(find_available_port $((FRONTEND_PORT+1)))
            if [ $? -eq 0 ]; then
                log_info "Switching to available port $new_port for frontend."
                FRONTEND_PORT=$new_port
            else
                log_error "Could not find an available port for frontend."
                exit 1
            fi
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
    nohup python3 -m uvicorn src.backend.main:app --host 0.0.0.0 --port "$BACKEND_PORT" > backend.log 2>&1 &
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
    nohup streamlit run src/frontend/app.py --server.address 0.0.0.0 --server.port "$FRONTEND_PORT" > frontend.log 2>&1 &
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

# Function to get the machine's IP address for network access
get_network_ip() {
    # Try multiple methods to get an IP address
    if command -v hostname &> /dev/null; then
        IP=$(hostname -I | awk '{print $1}')
        if [ -n "$IP" ]; then
            echo $IP
            return 0
        fi
    fi
    
    if command -v ip &> /dev/null; then
        IP=$(ip addr show | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | cut -d/ -f1 | head -n 1)
        if [ -n "$IP" ]; then
            echo $IP
            return 0
        fi
    fi
    
    if command -v ifconfig &> /dev/null; then
        IP=$(ifconfig | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | cut -d: -f2 | head -n 1)
        if [ -n "$IP" ]; then
            echo $IP
            return 0
        fi
    fi
    
    # If all methods fail, use localhost
    echo "localhost"
    return 1
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
elif [ "$1" == "network" ] || [ "$1" == "--network" ]; then
    # Force network mode with explicit IP
    if [ -n "$2" ]; then
        # Use provided IP if given
        NETWORK_IP="$2"
        shift 2
    else
        # Otherwise autodetect
        NETWORK_IP=$(get_network_ip)
        shift
    fi
    log_info "Network mode enabled. Using IP: $NETWORK_IP"
    # Force network mode settings
    export BACKEND_HOST="$NETWORK_IP"
    export ALLOWED_ORIGINS="http://$NETWORK_IP:8501,http://localhost:8501"
elif [ "$1" == "local" ] || [ "$1" == "--local" ]; then
    # Force local-only mode
    NETWORK_IP="localhost"
    export BACKEND_HOST="localhost"
    export ALLOWED_ORIGINS="http://localhost:8501"
    log_info "Local-only mode enabled. App will not be accessible over network."
    shift
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

# Get network IP for display
NETWORK_IP=$(get_network_ip)

# Configure frontend to find backend by setting environment variable
export BACKEND_HOST="$NETWORK_IP"
export BACKEND_PORT="$BACKEND_PORT"

# Display ports being used
log_info "Using backend port: $BACKEND_PORT"
log_info "Using frontend port: $FRONTEND_PORT"

start_backend
start_frontend
save_pids

# Print summary
echo -e "\n${BOLD}-------------------------------------${RESET}"
log_success "Services started successfully!"
echo -e "${BOLD}Local Backend:${RESET} http://localhost:$BACKEND_PORT"
echo -e "${BOLD}Local Frontend:${RESET} http://localhost:$FRONTEND_PORT"
echo -e "\n${BOLD}Network Backend:${RESET} http://$NETWORK_IP:$BACKEND_PORT"
echo -e "${BOLD}Network Frontend:${RESET} http://$NETWORK_IP:$FRONTEND_PORT"
echo -e "\nTo stop the application, run: $0 stop"
echo -e "To run in network-only mode: $0 network [optional-ip]"
echo -e "To run in local-only mode: $0 local"
echo -e "${BOLD}-------------------------------------${RESET}\n"
