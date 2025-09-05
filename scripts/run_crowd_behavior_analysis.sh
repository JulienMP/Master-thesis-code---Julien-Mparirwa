#!/bin/bash
#SBATCH --job-name=crowd_behavior_analysis
#SBATCH --output=$LOG_DIR/crowd_behavior_analysis_%j.out
#SBATCH --error=$LOG_DIR/crowd_behavior_analysis_%j.err
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
    
    mkdir -p "$CROWD_BEHAVIOR_OUTPUT_DIR"
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
    python -c "import scipy; print(f'SciPy version: {scipy.__version__}')" || {
        echo "ERROR: SciPy not found"; exit 1;
    }
}

print_job_info() {
    echo "=== CROWD BEHAVIOR ANALYSIS JOB ==="
    echo "Job started at: $(date)"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node(s): $SLURM_JOB_NODELIST"
    echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
    echo "Memory allocated: 16G"
    echo "Input directory: $TRACKING_OUTPUT_DIR"
    echo "Output directory: $CROWD_BEHAVIOR_OUTPUT_DIR"
    echo ""
}

check_input_data() {
    if [ ! -d "$TRACKING_OUTPUT_DIR" ]; then
        echo "ERROR: Input directory not found: $TRACKING_OUTPUT_DIR"
        echo "Please ensure tracking results are available before running this job"
        exit 1
    fi

    echo "Input data structure check:"
    for split in train val test; do
        split_path="$TRACKING_OUTPUT_DIR/$split"
        if [ -d "$split_path" ]; then
            video_count=$(find "$split_path" -name "*_tracking.txt" | wc -l)
            echo "  $split: $video_count tracking files"
        else
            echo "  $split: directory not found"
        fi
    done

    total_tracking_files=$(find "$TRACKING_OUTPUT_DIR" -name "*_tracking.txt" | wc -l)
    echo "  Total tracking files: $total_tracking_files"

    if [ $total_tracking_files -eq 0 ]; then
        echo "ERROR: No tracking files found in $TRACKING_OUTPUT_DIR"
        echo "Please ensure tracking processing has completed successfully"
        exit 1
    fi
}

run_crowd_behavior_analysis() {
    echo ""
    echo "=== STARTING CROWD BEHAVIOR ANALYSIS ==="
    echo "Expected processing time: ~2-5 minutes per 100 videos"
    echo ""

    START_TIME=$(date +%s)

    python "$PROJECT_ROOT/src/tracking/crowd_behavior_analysis.py" \
        "$TRACKING_OUTPUT_DIR" \
        "$CROWD_BEHAVIOR_OUTPUT_DIR"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    echo ""
    echo "=== JOB COMPLETION SUMMARY ==="
    echo "Exit code: $EXIT_CODE"
    echo "Total runtime: ${RUNTIME} seconds ($(($RUNTIME / 60)) minutes)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: Crowd behavior analysis completed successfully"
        show_results_summary
    else
        echo "ERROR: Crowd behavior analysis failed with exit code $EXIT_CODE"
        show_partial_results
        exit 1
    fi
}

show_results_summary() {
    echo ""
    echo "Output directory structure:"
    if command -v tree &> /dev/null; then
        tree "$CROWD_BEHAVIOR_OUTPUT_DIR" -L 4 | head -20
        echo "... (showing first 20 lines)"
    else
        find "$CROWD_BEHAVIOR_OUTPUT_DIR" -type f -name "*_crowd_behavior.txt" | head -10
        echo "... (showing first 10 crowd behavior files)"
    fi
    
    echo ""
    echo "Generated files:"
    crowd_files=$(find "$CROWD_BEHAVIOR_OUTPUT_DIR" -name "*_crowd_behavior.txt" | wc -l)
    echo "  Crowd behavior files: $crowd_files"
    
    stats_file="$CROWD_BEHAVIOR_OUTPUT_DIR/crowd_behavior_processing_stats.json"
    if [ -f "$stats_file" ]; then
        echo "  Processing statistics saved to: $stats_file"
        echo ""
        echo "Quick stats summary:"
        python -c "
import json
try:
    with open('$stats_file', 'r') as f:
        stats = json.load(f)
    print(f\"  Total videos: {stats.get('total_videos', 'N/A')}\")
    print(f\"  Successful: {stats.get('successful', 'N/A')}\")
    print(f\"  Failed: {stats.get('failed', 'N/A')}\")
    print(f\"  Success rate: {stats.get('success_rate', 0)*100:.1f}%\")
    print(f\"  Avg processing time: {stats.get('avg_processing_time', 0):.2f}s per video\")
except Exception as e:
    print(f\"  Could not read stats: {e}\")
"
    fi
    
    echo ""
    echo "Sample crowd behavior output:"
    sample_file=$(find "$CROWD_BEHAVIOR_OUTPUT_DIR" -name "*_crowd_behavior.txt" | head -1)
    if [ -f "$sample_file" ]; then
        echo "File: $(basename "$sample_file")"
        echo "First 5 lines:"
        head -5 "$sample_file"
    fi
}

show_partial_results() {
    partial_files=$(find "$CROWD_BEHAVIOR_OUTPUT_DIR" -name "*_crowd_behavior.txt" 2>/dev/null | wc -l)
    if [ $partial_files -gt 0 ]; then
        echo "Partial results: $partial_files files were generated before failure"
    fi
}

cleanup() {
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    
    echo ""
    echo "Disk usage:"
    du -sh "$CROWD_BEHAVIOR_OUTPUT_DIR" 2>/dev/null || echo "Could not calculate disk usage"
    
    echo ""
    echo "Job completed at $(date)"
}

main() {
    source_env_file
    setup_environment
    check_dependencies
    print_job_info
    check_input_data
    run_crowd_behavior_analysis
    cleanup
}

trap cleanup EXIT
main