#!/bin/bash
#SBATCH --job-name=soccer_prediction
#SBATCH --output=$LOG_DIR/prediction_%j.out
#SBATCH --error=$LOG_DIR/prediction_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

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
    
    local prediction_output_dir="$PREDICTION_OUTPUT_DIR/prediction_results_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$prediction_output_dir"
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
    
    export CURRENT_OUTPUT_DIR="$prediction_output_dir"
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
    echo "=== SOCCER VIDEO RUNTIME PREDICTION ==="
    echo "Job started at: $(date)"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node(s): $SLURM_JOB_NODELIST"
    echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
    echo "Memory allocated: 64G"
    echo "Output directory: $CURRENT_OUTPUT_DIR"
    echo ""
}

check_conda_environments() {
    echo "Checking conda environments..."
    source ~/anaconda3/etc/profile.d/conda.sh

    conda env list | grep "$TRACKING_ENV"
    if [ $? -ne 0 ]; then
        echo "ERROR: $TRACKING_ENV environment not found"
        exit 1
    fi

    conda env list | grep "$VISUAL_ENV"
    if [ $? -ne 0 ]; then
        echo "ERROR: $VISUAL_ENV environment not found"
        exit 1
    fi

    echo "All required environments found"
}

verify_required_components() {
    echo "Checking required components..."

    if [ ! -f "$TRAINING_OUTPUT_DIR/simple_soccer_model.pth" ]; then
        echo "ERROR: Trained model not found at $TRAINING_OUTPUT_DIR/simple_soccer_model.pth"
        exit 1
    fi

    if [ ! -d "$EMBEDDINGS_OUTPUT_DIR" ]; then
        echo "ERROR: Precomputed embeddings not found at $EMBEDDINGS_OUTPUT_DIR"
        echo "Please run precompute_embeddings.py first"
        exit 1
    fi

    if [ ! -d "$ORGANIZED_CLIPS_DIR/test" ]; then
        echo "ERROR: Test dataset not found at $ORGANIZED_CLIPS_DIR/test"
        exit 1
    fi

    echo "Trained model found: $TRAINING_OUTPUT_DIR/simple_soccer_model.pth"
    model_size=$(du -h "$TRAINING_OUTPUT_DIR/simple_soccer_model.pth" | cut -f1)
    echo "Model size: $model_size"

    embedding_files=$(find "$EMBEDDINGS_OUTPUT_DIR" -name "*_learned_features.h5" | wc -l)
    echo "Precomputed embeddings: $embedding_files files"

    test_clips=$(find "$ORGANIZED_CLIPS_DIR/test" -name "*.mkv" | wc -l)
    echo "Test dataset: $test_clips video clips"

    if [ $embedding_files -eq 0 ]; then
        echo "ERROR: No precomputed embeddings found"
        echo "Please run preprocessing first"
        exit 1
    fi

    if [ $test_clips -eq 0 ]; then
        echo "ERROR: No test clips found"
        exit 1
    fi
}

show_pipeline_info() {
    echo ""
    echo "=== RUNNING EFFICIENT PREDICTION PIPELINE ==="
    echo ""
    echo "Pipeline Architecture:"
    echo "  Input: Single test MKV video clip"
    echo "  ├── Thread 1: ByteTrack → Crowd Analysis → Statistical Encoding"
    echo "  ├── Thread 2: SlowFast Visual Feature Extraction"
    echo "  └── Fast Similarity Search using precomputed 256D embeddings"
    echo ""
    echo "Efficiency improvements:"
    echo "  - Uses precomputed learned embeddings (train+val only)"
    echo "  - No data leakage (test split excluded from similarity database)"
    echo "  - Fast similarity search in learned space"
    echo ""
}

run_prediction() {
    INPUT_CLIP=""
    if [ $# -gt 0 ]; then
        INPUT_CLIP="--input_clip $1"
        echo "Using provided input clip: $1"
    else
        echo "No input clip provided, will select random test clip"
    fi

    START_TIME=$(date +%s)

    echo "Running main prediction script..."

    if [ ! -f "$PROJECT_ROOT/src/full_pipeline/predict_runtime.py" ]; then
        echo "ERROR: predict_runtime.py not found in $PROJECT_ROOT/src/full_pipeline/"
        exit 1
    fi

    python "$PROJECT_ROOT/src/full_pipeline/predict_runtime.py" \
        "$TRAINING_OUTPUT_DIR/simple_soccer_model.pth" \
        "$EMBEDDINGS_OUTPUT_DIR" \
        "$ORGANIZED_CLIPS_DIR" \
        "$PROJECT_ROOT/src/visual_head/extract_visual_features.py" \
        "$BYTETRACK_HOME" \
        "$CURRENT_OUTPUT_DIR" \
        $INPUT_CLIP \
        --top_k 10 \
        --tracking_env "$TRACKING_ENV" \
        --visual_env "$VISUAL_ENV"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    echo ""
    echo "=== PREDICTION COMPLETION SUMMARY ==="
    echo "Exit code: $EXIT_CODE"
    echo "Total runtime: ${RUNTIME} seconds ($(($RUNTIME / 60)) minutes)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: Prediction completed successfully"
        show_prediction_results
    else
        echo "ERROR: Prediction failed with exit code $EXIT_CODE"
        show_partial_results
        exit 1
    fi
}

show_prediction_results() {
    echo ""
    echo "Output directory structure:"
    if command -v tree &> /dev/null; then
        tree "$CURRENT_OUTPUT_DIR" -L 3
    else
        echo "Contents of $CURRENT_OUTPUT_DIR:"
        find "$CURRENT_OUTPUT_DIR" -type f | head -20
    fi
    
    if [ -f "$CURRENT_OUTPUT_DIR/prediction_summary.txt" ]; then
        echo ""
        echo "=== PREDICTION SUMMARY ==="
        cat "$CURRENT_OUTPUT_DIR/prediction_summary.txt"
    fi
    
    similar_clips_dir="$CURRENT_OUTPUT_DIR/similar_clips"
    if [ -d "$similar_clips_dir" ]; then
        clip_count=$(find "$similar_clips_dir" -name "*.mkv" | wc -l)
        echo ""
        echo "Downloaded $clip_count similar clips to: $similar_clips_dir"
        echo "Similar clips:"
        ls -1 "$similar_clips_dir" | head -10
    fi
    
    if [ -f "$CURRENT_OUTPUT_DIR/prediction_report.json" ]; then
        echo ""
        echo "Detailed report available at: $CURRENT_OUTPUT_DIR/prediction_report.json"
        
        echo ""
        echo "=== KEY METRICS ==="
        python3 -c "
import json
try:
    with open('$CURRENT_OUTPUT_DIR/prediction_report.json', 'r') as f:
        report = json.load(f)
    
    query = report['query_video']
    model_pred = report['model_prediction']
    combined_pred = report['combined_prediction']
    analysis = report['similar_clips_analysis']
    
    print(f\"Query Video: {query['name']}\")
    print(f\"True Category: {query['true_category']}\")
    print(f\"\")
    print(f\"TRAINED MODEL PREDICTION:\")
    print(f\"  Goal Probability: {model_pred['goal_probability']:.3f}\")
    print(f\"  Predicted Class: {model_pred['predicted_class']}\")
    print(f\"  Attention Weights - Visual: {model_pred['attention_weights']['visual']:.3f}, Crowd: {model_pred['attention_weights']['crowd']:.3f}\")
    print(f\"\")
    print(f\"COMBINED FINAL PREDICTION:\")
    print(f\"  Goal Probability: {combined_pred['goal_probability']:.3f}\")
    print(f\"  Predicted Class: {combined_pred['predicted_class']}\")
    print(f\"  Top Similar Category: {combined_pred['top_similar_category']}\")
    print(f\"  Confidence Score: {combined_pred['confidence']:.3f}\")
    print(f\"\")
    print(f\"Similar Clips Analyzed: {analysis['total_clips']}\")
    
    print(f\"\nCategory Distribution:\")
    for cat, count in analysis['category_distribution'].items():
        pct = count / analysis['total_clips'] * 100
        print(f\"  {cat}: {count} clips ({pct:.1f}%)\")
    
    print(f\"\nEfficiency Notes:\")
    print(f\"  - Used precomputed embeddings for fast similarity search\")
    print(f\"  - Only train/val splits used (no data leakage)\")
    print(f\"  - Query processed through trained fusion model\")
        
except Exception as e:
    print(f\"Could not parse report: {e}\")
"
    fi
}

show_partial_results() {
    if [ -d "$CURRENT_OUTPUT_DIR" ]; then
        echo ""
        echo "Partial results in $CURRENT_OUTPUT_DIR:"
        find "$CURRENT_OUTPUT_DIR" -type f 2>/dev/null | head -10
    fi
}

cleanup() {
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    
    echo ""
    echo "Output directory disk usage:"
    du -sh "$CURRENT_OUTPUT_DIR" 2>/dev/null || echo "Could not calculate disk usage"
    
    echo ""
    echo "Cleaning up temporary files..."
    find "$CURRENT_OUTPUT_DIR" -name "temp" -type d -exec rm -rf {} + 2>/dev/null || true
    
    echo ""
    echo "=== FINAL SUMMARY ==="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Start time: $(date -d @$START_TIME)"
    echo "End time: $(date)"
    echo "Duration: ${RUNTIME} seconds"
    echo "Output saved to: $CURRENT_OUTPUT_DIR"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Status: SUCCESS - Efficient prediction completed"
    else
        echo "Status: FAILED - Check error log for troubleshooting"
    fi
    
    echo ""
    echo "Job completed at: $(date)"
}

main() {
    source_env_file
    setup_environment
    check_gpu
    print_job_info
    check_conda_environments
    verify_required_components
    show_pipeline_info
    run_prediction "$@"
    cleanup
}

trap cleanup EXIT
main "$@"