#!/bin/bash
#SBATCH --job-name=extract_tracking_features
#SBATCH --output=$LOG_DIR/extract_tracking_features_%j.out
#SBATCH --error=$LOG_DIR/extract_tracking_features_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

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
    
    mkdir -p "$TRACKING_FEATURES_OUTPUT_DIR"
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

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU info:"
        nvidia-smi
        echo ""
    else
        echo "WARNING: No GPU detected"
    fi
}

print_job_info() {
    echo "=== TRACKING FEATURE EXTRACTION JOB ==="
    echo "Job started at: $(date)"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node(s): $SLURM_JOB_NODELIST"
    echo "Input directory: $TRACKING_OUTPUT_DIR"
    echo "Output directory: $TRACKING_FEATURES_OUTPUT_DIR"
    echo "Device: $TRACKING_DEVICE"
    echo ""
}

run_feature_extraction() {
    echo "Starting tracking feature extraction at $(date)"
    
    if [ ! -d "$TRACKING_OUTPUT_DIR" ]; then
        echo "ERROR: Input directory not found: $TRACKING_OUTPUT_DIR"
        exit 1
    fi

    START_TIME=$(date +%s)

    python "$PROJECT_ROOT/src/tracking/extract_tracking_features.py" \
        "$TRACKING_OUTPUT_DIR" \
        "$TRACKING_FEATURES_OUTPUT_DIR" \
        --device "$TRACKING_DEVICE"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    echo ""
    echo "=== JOB COMPLETION SUMMARY ==="
    echo "Exit code: $EXIT_CODE"
    echo "Total runtime: ${RUNTIME} seconds ($(($RUNTIME / 60)) minutes)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: Feature extraction completed successfully"
        show_output_summary
    else
        echo "ERROR: Feature extraction failed with exit code $EXIT_CODE"
        exit 1
    fi
}

show_output_summary() {
    echo ""
    echo "Output directory structure:"
    if command -v tree &> /dev/null; then
        tree "$TRACKING_FEATURES_OUTPUT_DIR" -L 3
    else
        find "$TRACKING_FEATURES_OUTPUT_DIR" -type f -name "*.h5" | head -10
        echo "... (showing first 10 .h5 files)"
    fi

    echo ""
    echo "Generated files:"
    h5_count=$(find "$TRACKING_FEATURES_OUTPUT_DIR" -name "*.h5" | wc -l)
    echo "  H5 feature files: $h5_count"
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
    check_gpu
    print_job_info
    run_feature_extraction
    cleanup
}

trap cleanup EXIT
main