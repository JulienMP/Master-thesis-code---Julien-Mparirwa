#!/bin/bash
#SBATCH --job-name=crowd_statistical_encoder
#SBATCH --output=$LOG_DIR/crowd_statistical_encoder_%j.out
#SBATCH --error=$LOG_DIR/crowd_statistical_encoder_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

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
    
    mkdir -p "$CROWD_STATISTICAL_OUTPUT_DIR"
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

check_dependencies() {
    echo "Checking for required packages..."
    python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || {
        echo "ERROR: NumPy not found"; exit 1;
    }
    python -c "import pandas; print(f'Pandas version: {pandas.__version__}')" || {
        echo "ERROR: Pandas not found"; exit 1;
    }
    python -c "import h5py; print(f'h5py version: {h5py.__version__}')" || {
        echo "ERROR: h5py not found"; exit 1;
    }
    python -c "import scipy; print(f'SciPy version: {scipy.__version__}')" || {
        echo "ERROR: SciPy not found"; exit 1;
    }
}

print_job_info() {
    echo "=== CROWD BEHAVIOR STATISTICAL ENCODING JOB ==="
    echo "Job started at: $(date)"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node(s): $SLURM_JOB_NODELIST"
    echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
    echo "Memory allocated: 16G"
    echo "Input directory: $CROWD_BEHAVIOR_OUTPUT_DIR"
    echo "Output directory: $CROWD_STATISTICAL_OUTPUT_DIR"
    echo "Encoding method: comprehensive_statistics (94-dimensional features)"
    echo ""
}

check_input_data() {
    if [ ! -d "$CROWD_BEHAVIOR_OUTPUT_DIR" ]; then
        echo "ERROR: Input directory not found: $CROWD_BEHAVIOR_OUTPUT_DIR"
        echo "Please ensure crowd behavior analysis has completed successfully"
        exit 1
    fi

    echo "Input data structure check:"
    for split in train val test; do
        split_path="$CROWD_BEHAVIOR_OUTPUT_DIR/$split"
        if [ -d "$split_path" ]; then
            crowd_files=$(find "$split_path" -name "*_crowd_behavior.txt" | wc -l)
            echo "  $split: $crowd_files crowd behavior files"
        else
            echo "  $split: directory not found"
        fi
    done

    total_crowd_files=$(find "$CROWD_BEHAVIOR_OUTPUT_DIR" -name "*_crowd_behavior.txt" | wc -l)
    echo "  Total crowd behavior files: $total_crowd_files"

    if [ $total_crowd_files -eq 0 ]; then
        echo "ERROR: No crowd behavior files found in $CROWD_BEHAVIOR_OUTPUT_DIR"
        echo "Please ensure crowd behavior analysis has completed successfully"
        exit 1
    fi
}

run_statistical_encoding() {
    echo ""
    echo "=== STARTING CROWD BEHAVIOR STATISTICAL ENCODING ==="
    echo "Expected processing time: ~1-2 seconds per video"
    echo ""

    START_TIME=$(date +%s)

    python "$PROJECT_ROOT/src/tracking/crowd_statistical_encoder.py" \
        "$CROWD_BEHAVIOR_OUTPUT_DIR" \
        "$CROWD_STATISTICAL_OUTPUT_DIR"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    echo ""
    echo "=== JOB COMPLETION SUMMARY ==="
    echo "Exit code: $EXIT_CODE"
    echo "Total runtime: ${RUNTIME} seconds ($(($RUNTIME / 60)) minutes)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: Crowd behavior statistical encoding completed successfully"
        show_results_summary
    else
        echo "ERROR: Crowd behavior statistical encoding failed with exit code $EXIT_CODE"
        show_partial_results
        exit 1
    fi
}

show_results_summary() {
    echo ""
    echo "Output directory structure:"
    if command -v tree &> /dev/null; then
        tree "$CROWD_STATISTICAL_OUTPUT_DIR" -L 3
    else
        find "$CROWD_STATISTICAL_OUTPUT_DIR" -type f -name "*.h5" | head -10
        echo "... (showing first 10 H5 files)"
    fi
    
    echo ""
    echo "Generated files:"
    h5_files=$(find "$CROWD_STATISTICAL_OUTPUT_DIR" -name "*_crowd_features.h5" | wc -l)
    json_files=$(find "$CROWD_STATISTICAL_OUTPUT_DIR" -name "*_crowd_metadata.json" | wc -l)
    echo "  H5 feature files: $h5_files"
    echo "  JSON metadata files: $json_files"
    
    stats_file="$CROWD_STATISTICAL_OUTPUT_DIR/crowd_statistical_metadata.json"
    if [ -f "$stats_file" ]; then
        echo "  Processing statistics saved to: $stats_file"
        echo ""
        echo "Quick stats summary:"
        python -c "
import json
try:
    with open('$stats_file', 'r') as f:
        stats = json.load(f)
    total_stats = stats.get('total_statistics', {})
    print(f\"  Total videos: {total_stats.get('total_videos', 'N/A')}\")
    print(f\"  Successful: {total_stats.get('successful', 'N/A')}\")
    print(f\"  Failed: {total_stats.get('failed', 'N/A')}\")
    print(f\"  Feature dimension: {stats.get('feature_dimension', 'N/A')}\")
    print(f\"  Encoding method: {stats.get('encoding_method', 'N/A')}\")
except Exception as e:
    print(f\"  Could not read stats: {e}\")
"
    fi
    
    echo ""
    echo "Testing encoded features access:"
    python -c "
import h5py
import numpy as np
from pathlib import Path

output_dir = Path('$CROWD_STATISTICAL_OUTPUT_DIR')
h5_files = list(output_dir.glob('*/*_crowd_features.h5'))

if h5_files:
    sample_file = h5_files[0]
    print(f'Sample file: {sample_file.name}')
    
    try:
        with h5py.File(sample_file, 'r') as f:
            keys = list(f.keys())
            if keys:
                sample_key = keys[0]
                features = f[sample_key][:]
                print(f'Sample video: {sample_key}')
                print(f'Feature shape: {features.shape}')
                print(f'Feature range: [{features.min():.2f}, {features.max():.2f}]')
                print(f'Total videos in file: {len(keys)}')
            else:
                print('No datasets found in file')
    except Exception as e:
        print(f'Error reading sample file: {e}')
else:
    print('No H5 files found')
"
}

show_partial_results() {
    partial_files=$(find "$CROWD_STATISTICAL_OUTPUT_DIR" -name "*.h5" 2>/dev/null | wc -l)
    if [ $partial_files -gt 0 ]; then
        echo "Partial results: $partial_files H5 files were generated before failure"
    fi
}

cleanup() {
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    
    echo ""
    echo "Disk usage:"
    du -sh "$CROWD_STATISTICAL_OUTPUT_DIR" 2>/dev/null || echo "Could not calculate disk usage"
    
    echo ""
    echo "Job completed at $(date)"
}

main() {
    source_env_file
    setup_environment
    check_dependencies
    print_job_info
    check_input_data
    run_statistical_encoding
    cleanup
}

trap cleanup EXIT
main