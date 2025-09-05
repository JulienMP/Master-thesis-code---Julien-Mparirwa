#!/bin/bash
#SBATCH --job-name=extract_visual_features
#SBATCH --output=$LOG_DIR/extract_visual_features_%j.out
#SBATCH --error=$LOG_DIR/extract_visual_features_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

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
    
    mkdir -p "$VISUAL_FEATURES_OUTPUT_DIR"
    mkdir -p "$LOG_DIR"
    
    cd "$PROJECT_ROOT"
    
    if [ -d "$VISUAL_VENV_PATH" ]; then
        echo "Activating visual processing environment at $VISUAL_VENV_PATH"
        source "$VISUAL_VENV_PATH/bin/activate"
        which python
        python --version
    else
        echo "ERROR: Visual processing environment not found at $VISUAL_VENV_PATH"
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

check_dependencies() {
    echo "Checking for required packages..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
        echo "ERROR: PyTorch not found"; exit 1;
    }
    python -c "import pytorchvideo; print(f'PyTorchVideo version: {pytorchvideo.__version__}')" || {
        echo "ERROR: PyTorchVideo not found"; exit 1;
    }
    python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" || {
        echo "ERROR: OpenCV not found"; exit 1;
    }
    python -c "import h5py; print(f'h5py version: {h5py.__version__}')" || {
        echo "ERROR: h5py not found"; exit 1;
    }
}

print_job_info() {
    echo "=== VISUAL FEATURE EXTRACTION JOB ==="
    echo "Job started at: $(date)"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node(s): $SLURM_JOB_NODELIST"
    echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
    echo "Memory allocated: 64G"
    echo "Input directory: $VISUAL_INPUT_DIR"
    echo "Output directory: $VISUAL_FEATURES_OUTPUT_DIR"
    echo "Device: $VISUAL_DEVICE"
    echo ""
}

check_input_data() {
    if [ ! -d "$VISUAL_INPUT_DIR" ]; then
        echo "ERROR: Input directory not found: $VISUAL_INPUT_DIR"
        echo "Please ensure the dataset has been created first"
        exit 1
    fi

    echo "Dataset structure check:"
    for split in train val test; do
        split_path="$VISUAL_INPUT_DIR/$split"
        if [ -d "$split_path" ]; then
            video_count=$(find "$split_path" -name "*.mkv" | wc -l)
            echo "  $split: $video_count videos"
        else
            echo "  $split: directory not found"
        fi
    done

    total_videos=$(find "$VISUAL_INPUT_DIR" -name "*.mkv" | wc -l)
    echo "  Total videos: $total_videos"

    if [ $total_videos -eq 0 ]; then
        echo "ERROR: No video files found in $VISUAL_INPUT_DIR"
        exit 1
    fi
}

run_visual_feature_extraction() {
    echo ""
    echo "=== STARTING VISUAL FEATURE EXTRACTION ==="
    echo ""

    START_TIME=$(date +%s)

    python "$PROJECT_ROOT/src/visual_head/extract_visual_features.py" \
        "$VISUAL_INPUT_DIR" \
        "$VISUAL_FEATURES_OUTPUT_DIR" \
        --device "$VISUAL_DEVICE"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    echo ""
    echo "=== JOB COMPLETION SUMMARY ==="
    echo "Exit code: $EXIT_CODE"
    echo "Total runtime: ${RUNTIME} seconds ($(($RUNTIME / 60)) minutes)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: Feature extraction completed successfully"
        show_results_summary
    else
        echo "ERROR: Feature extraction failed with exit code $EXIT_CODE"
        exit 1
    fi
}

show_results_summary() {
    echo ""
    echo "Output directory structure:"
    if command -v tree &> /dev/null; then
        tree "$VISUAL_FEATURES_OUTPUT_DIR" -L 3
    else
        find "$VISUAL_FEATURES_OUTPUT_DIR" -type f -name "*.h5" | head -10
        echo "... (showing first 10 .h5 files)"
    fi

    echo ""
    echo "Generated files:"
    h5_count=$(find "$VISUAL_FEATURES_OUTPUT_DIR" -name "*.h5" | wc -l)
    json_count=$(find "$VISUAL_FEATURES_OUTPUT_DIR" -name "*.json" | wc -l)
    echo "  H5 feature files: $h5_count"
    echo "  JSON metadata files: $json_count"

    overall_stats="$VISUAL_FEATURES_OUTPUT_DIR/overall_metadata.json"
    if [ -f "$overall_stats" ]; then
        echo ""
        echo "Processing statistics:"
        python -c "
import json
try:
    with open('$overall_stats', 'r') as f:
        stats = json.load(f)
    total_stats = stats.get('total_statistics', {})
    print(f\"  Total videos: {total_stats.get('total_videos', 'N/A')}\")
    print(f\"  Successful: {total_stats.get('successful', 'N/A')}\")
    print(f\"  Failed: {total_stats.get('failed', 'N/A')}\")
    print(f\"  Average processing time: {stats.get('avg_processing_time', 0):.2f}s per video\")
    print(f\"  Feature dimension: {stats.get('feature_dimension', 'N/A')}\")
except Exception as e:
    print(f\"  Could not read stats: {e}\")
"
    fi
}

cleanup() {
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    
    echo ""
    echo "Disk usage:"
    du -sh "$VISUAL_FEATURES_OUTPUT_DIR" 2>/dev/null || echo "Could not calculate disk usage"
    
    echo ""
    echo "Job completed at $(date)"
}

main() {
    source_env_file
    setup_environment
    check_gpu
    check_dependencies
    print_job_info
    check_input_data
    run_visual_feature_extraction
    cleanup
}

trap cleanup EXIT
main