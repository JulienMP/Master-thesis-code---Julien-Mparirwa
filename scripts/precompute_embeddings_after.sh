#!/bin/bash
#SBATCH --job-name=precompute_embeddings
#SBATCH --output=$LOG_DIR/precompute_embeddings_%j.out
#SBATCH --error=$LOG_DIR/precompute_embeddings_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

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
    
    mkdir -p "$EMBEDDINGS_OUTPUT_DIR"
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
    echo "=== LEARNED EMBEDDINGS PREPROCESSING ==="
    echo "Job started at: $(date)"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node(s): $SLURM_JOB_NODELIST"
    echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
    echo "Memory allocated: 32G"
    echo "Model: $TRAINING_OUTPUT_DIR/simple_soccer_model.pth"
    echo "Visual features: $VISUAL_FEATURES_OUTPUT_DIR"
    echo "Crowd features: $CROWD_STATISTICAL_OUTPUT_DIR"
    echo "Output directory: $EMBEDDINGS_OUTPUT_DIR"
    echo ""
}

verify_required_components() {
    echo "Checking required components..."

    if [ ! -f "$TRAINING_OUTPUT_DIR/simple_soccer_model.pth" ]; then
        echo "ERROR: Trained model not found at $TRAINING_OUTPUT_DIR/simple_soccer_model.pth"
        exit 1
    fi

    if [ ! -d "$VISUAL_FEATURES_OUTPUT_DIR" ]; then
        echo "ERROR: Visual features not found at $VISUAL_FEATURES_OUTPUT_DIR"
        exit 1
    fi

    if [ ! -d "$CROWD_STATISTICAL_OUTPUT_DIR" ]; then
        echo "ERROR: Crowd features not found at $CROWD_STATISTICAL_OUTPUT_DIR"
        exit 1
    fi

    echo "All required components found"

    train_visual=$(find "$VISUAL_FEATURES_OUTPUT_DIR/train" -name "*_features.h5" 2>/dev/null | wc -l)
    train_crowd=$(find "$CROWD_STATISTICAL_OUTPUT_DIR/train" -name "*_crowd_features.h5" 2>/dev/null | wc -l)
    val_visual=$(find "$VISUAL_FEATURES_OUTPUT_DIR/val" -name "*_features.h5" 2>/dev/null | wc -l)
    val_crowd=$(find "$CROWD_STATISTICAL_OUTPUT_DIR/val" -name "*_crowd_features.h5" 2>/dev/null | wc -l)

    echo "Input data summary:"
    echo "  Train: $train_visual visual files, $train_crowd crowd files"
    echo "  Val: $val_visual visual files, $val_crowd crowd files"
    echo "  Test: EXCLUDED (preventing data leakage)"
}

run_preprocessing() {
    echo ""
    echo "=== STARTING PREPROCESSING ==="
    echo "Processing train and val splits only"
    echo "Expected runtime: 2-3 hours"
    echo ""

    START_TIME=$(date +%s)

    python "$PROJECT_ROOT/src/full_pipeline/precompute_embeddings.py" \
        "$TRAINING_OUTPUT_DIR/simple_soccer_model.pth" \
        "$VISUAL_FEATURES_OUTPUT_DIR" \
        "$CROWD_STATISTICAL_OUTPUT_DIR" \
        "$EMBEDDINGS_OUTPUT_DIR" \
        --device "$EMBEDDINGS_DEVICE"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    echo ""
    echo "=== PREPROCESSING COMPLETION SUMMARY ==="
    echo "Exit code: $EXIT_CODE"
    echo "Total runtime: ${RUNTIME} seconds ($(($RUNTIME / 60)) minutes)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: Learned embeddings preprocessing completed"
        show_preprocessing_results
    else
        echo "ERROR: Preprocessing failed with exit code $EXIT_CODE"
        show_partial_results
        exit 1
    fi
}

show_preprocessing_results() {
    echo ""
    echo "Output directory structure:"
    if command -v tree &> /dev/null; then
        tree "$EMBEDDINGS_OUTPUT_DIR" -L 3
    else
        find "$EMBEDDINGS_OUTPUT_DIR" -name "*.h5" | head -10
        echo "... (showing first 10 .h5 files)"
    fi
    
    embedding_files=$(find "$EMBEDDINGS_OUTPUT_DIR" -name "*_learned_features.h5" | wc -l)
    echo ""
    echo "Generated files:"
    echo "  Learned embedding files: $embedding_files"
    
    if [ -f "$EMBEDDINGS_OUTPUT_DIR/learned_embeddings_metadata.json" ]; then
        echo "  Metadata: learned_embeddings_metadata.json"
        echo ""
        echo "Processing statistics:"
        python3 -c "
import json
try:
    with open('$EMBEDDINGS_OUTPUT_DIR/learned_embeddings_metadata.json', 'r') as f:
        data = json.load(f)
    stats = data['processing_statistics']
    print(f\"  Total videos processed: {stats['total_videos']}\")
    print(f\"  Total successful: {stats['total_successful']}\")
    print(f\"  Success rate: {100 * stats['total_successful'] / stats['total_videos']:.1f}%\")
    print(f\"  Train split: {stats['train']['successful']}/{stats['train']['total_videos']}\")
    print(f\"  Val split: {stats['val']['successful']}/{stats['val']['total_videos']}\")
except Exception as e:
    print(f\"Could not read metadata: {e}\")
"
    fi
    
    echo ""
    echo "The precomputed embeddings are now ready for fast similarity search."
    echo "These 256D learned features can be used directly for inference."
}

show_partial_results() {
    partial_files=$(find "$EMBEDDINGS_OUTPUT_DIR" -name "*.h5" 2>/dev/null | wc -l)
    if [ $partial_files -gt 0 ]; then
        echo "Partial results: $partial_files embedding files were generated"
    fi
}

cleanup() {
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    
    echo ""
    echo "Disk usage:"
    du -sh "$EMBEDDINGS_OUTPUT_DIR" 2>/dev/null || echo "Could not calculate disk usage"
    
    echo ""
    echo "Job completed at: $(date)"
}

main() {
    source_env_file
    setup_environment
    check_gpu
    print_job_info
    verify_required_components
    run_preprocessing
    cleanup
}

trap cleanup EXIT
main