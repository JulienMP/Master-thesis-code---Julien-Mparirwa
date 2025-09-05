#!/bin/bash
#SBATCH --job-name=extract_clips
#SBATCH --output=$LOG_DIR/extract_clips_%j.out
#SBATCH --error=$LOG_DIR/extract_clips_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00

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
    
    mkdir -p "$CLIPS_OUTPUT_DIR"
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

print_job_info() {
    echo "Starting clip extraction job"
    echo "Job started at: $(date)"
    echo "Running on node: $(hostname)"
    echo "Data directory: $DATA_DIR"
    echo "Output directory: $CLIPS_OUTPUT_DIR"
    echo "Extraction type: $EXTRACTION_TYPE"
    
    case $EXTRACTION_TYPE in
        "background")
            echo "Clips per game: ${CLIPS_PER_GAME:-3}"
            ;;
        "freekicks")
            echo "Free kick window: ${FREEKICK_WINDOW:-10} seconds"
            ;;
        "penalties")
            echo "Penalty trigger window: ${PENALTY_WINDOW:-120} seconds"
            ;;
    esac
}

run_extraction() {
    echo "Starting extraction at $(date)"
    echo "Data will be processed from: $DATA_DIR"
    echo "Clips will be saved to: $CLIPS_OUTPUT_DIR"
    
    local cmd="python $PROJECT_ROOT/src/data/extract_clips.py $DATA_DIR $CLIPS_OUTPUT_DIR --type $EXTRACTION_TYPE"
    
    case $EXTRACTION_TYPE in
        "background")
            if [ -n "$CLIPS_PER_GAME" ]; then
                cmd="$cmd --clips-per-game $CLIPS_PER_GAME"
            fi
            ;;
        "freekicks")
            if [ -n "$FREEKICK_WINDOW" ]; then
                cmd="$cmd --freekick-window $FREEKICK_WINDOW"
            fi
            ;;
        "penalties")
            if [ -n "$PENALTY_WINDOW" ]; then
                cmd="$cmd --penalty-window $PENALTY_WINDOW"
            fi
            ;;
    esac
    
    echo "Running command: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "Extraction completed successfully at $(date)"
    else
        echo "Extraction failed with exit code $? at $(date)"
        exit 1
    fi
}

cleanup() {
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    echo "Job completed at $(date)"
}

validate_extraction_type() {
    case $EXTRACTION_TYPE in
        "goals"|"background"|"freekicks"|"penalties"|"shots")
            echo "Valid extraction type: $EXTRACTION_TYPE"
            ;;
        *)
            echo "ERROR: Invalid extraction type '$EXTRACTION_TYPE'"
            echo "Valid types: goals, background, freekicks, penalties, shots"
            exit 1
            ;;
    esac
}

main() {
    source_env_file
    validate_extraction_type
    setup_environment
    print_job_info
    run_extraction
    cleanup
}

trap cleanup EXIT
main