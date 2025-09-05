#!/bin/bash
#SBATCH --job-name=soccer_download
#SBATCH --output=$LOG_DIR/download_log_%j.out
#SBATCH --error=$LOG_DIR/download_error_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=72:00:00

set -e

source_env_file() {
    if [ -f "$PROJECT_ROOT/.env" ]; then
        source "$PROJECT_ROOT/.env"
        echo "Environment file loaded successfully"
    else
        echo "ERROR: Environment file not found at $PROJECT_ROOT/.env"
        exit 1
    fi
}

setup_environment() {
    echo "Setting up environment..."
    
    mkdir -p "$DATA_DIR"
    mkdir -p "$LOG_DIR"
    
    cd "$PROJECT_ROOT"
    
    if [ -d "$VENV_PATH" ]; then
        echo "Activating virtual environment at $VENV_PATH"
        source "$VENV_PATH/bin/activate"
        which python
        python --version
    else
        echo "ERROR: Virtual environment not found at $VENV_PATH"
        exit 1
    fi
}

run_download() {
    echo "Starting download at $(date)"
    echo "Data will be downloaded to: $DATA_DIR"
    
    python "$PROJECT_ROOT/src/data/download_data.py" "$DATA_DIR"
    
    if [ $? -eq 0 ]; then
        echo "Download completed successfully at $(date)"
    else
        echo "Download failed with exit code $? at $(date)"
        exit 1
    fi
}

cleanup() {
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    echo "Job completed at $(date)"
}

main() {
    source_env_file
    setup_environment
    run_download
    cleanup
}

trap cleanup EXIT
main