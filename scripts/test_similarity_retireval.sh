#!/bin/bash
#SBATCH --job-name=test_similarity_retrieval
#SBATCH --output=$LOG_DIR/test_similarity_retrieval_%j.out
#SBATCH --error=$LOG_DIR/test_similarity_retrieval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
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
    
    mkdir -p "$TESTING_OUTPUT_DIR"
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
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    else
        echo "WARNING: nvidia-smi not available"
    fi
}

print_job_info() {
    echo "=== TEST SIMILARITY RETRIEVAL ==="
    echo "Job started at: $(date)"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node(s): $SLURM_JOB_NODELIST"
    echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
    echo "Memory allocated: 16G"
    echo "GPU allocated: $CUDA_VISIBLE_DEVICES"
    echo "Model: $TRAINING_OUTPUT_DIR/simple_soccer_model.pth"
    echo "Visual features: $VISUAL_FEATURES_OUTPUT_DIR"
    echo "Crowd features: $CROWD_STATISTICAL_OUTPUT_DIR"
    echo "Output directory: $TESTING_OUTPUT_DIR"
    echo ""
}

verify_required_files() {
    echo "Checking required files and directories..."

    if [ -f "$TRAINING_OUTPUT_DIR/simple_soccer_model.pth" ]; then
        echo "✓ Model file found: $TRAINING_OUTPUT_DIR/simple_soccer_model.pth"
        model_size=$(du -h "$TRAINING_OUTPUT_DIR/simple_soccer_model.pth" | cut -f1)
        echo "  Model size: $model_size"
    else
        echo "✗ ERROR: Model file not found: $TRAINING_OUTPUT_DIR/simple_soccer_model.pth"
        echo "Please ensure the training script completed successfully and saved the model"
        exit 1
    fi

    if [ -d "$VISUAL_FEATURES_OUTPUT_DIR" ]; then
        test_visual_files=$(find "$VISUAL_FEATURES_OUTPUT_DIR/test" -name "*_features.h5" 2>/dev/null | wc -l)
        echo "✓ Visual features directory found: $VISUAL_FEATURES_OUTPUT_DIR"
        echo "  Test visual files: $test_visual_files"
    else
        echo "✗ ERROR: Visual features directory not found: $VISUAL_FEATURES_OUTPUT_DIR"
        exit 1
    fi

    if [ -d "$CROWD_STATISTICAL_OUTPUT_DIR" ]; then
        test_crowd_files=$(find "$CROWD_STATISTICAL_OUTPUT_DIR/test" -name "*_crowd_features.h5" 2>/dev/null | wc -l)
        echo "✓ Crowd features directory found: $CROWD_STATISTICAL_OUTPUT_DIR"
        echo "  Test crowd files: $test_crowd_files"
    else
        echo "✗ ERROR: Crowd features directory not found: $CROWD_STATISTICAL_OUTPUT_DIR"
        exit 1
    fi
}

verify_test_data() {
    echo ""
    echo "Quick test data verification:"
    python3 << EOF
import h5py
from pathlib import Path

test_visual_dir = Path("$VISUAL_FEATURES_OUTPUT_DIR/test")
test_crowd_dir = Path("$CROWD_STATISTICAL_OUTPUT_DIR/test")

categories = ['background', 'before_goal', 'free_kicks_goals', 'penalties', 'shots_no_goals']
total_samples = 0

for category in categories:
    visual_file = test_visual_dir / f"{category}_features.h5"
    crowd_file = test_crowd_dir / f"{category}_crowd_features.h5"
    
    if visual_file.exists() and crowd_file.exists():
        with h5py.File(visual_file, 'r') as vf, h5py.File(crowd_file, 'r') as cf:
            visual_count = len(vf.keys())
            crowd_count = len(cf.keys())
            common_count = len(set(vf.keys()).intersection(set(cf.keys())))
            total_samples += common_count
            print(f"  {category}: {common_count} samples (V:{visual_count}, C:{crowd_count})")
    else:
        print(f"  {category}: FILES MISSING")

print(f"Total test samples available: {total_samples}")
EOF
}

run_similarity_testing() {
    echo ""
    echo "=== STARTING SIMILARITY RETRIEVAL TESTING ==="
    echo "Expected runtime: 5-15 minutes"
    echo ""

    START_TIME=$(date +%s)

    python "$PROJECT_ROOT/src/multitask_model/test_similarity_retrieval.py" \
        "$TRAINING_OUTPUT_DIR/simple_soccer_model.pth" \
        "$VISUAL_FEATURES_OUTPUT_DIR" \
        "$CROWD_STATISTICAL_OUTPUT_DIR" \
        "$TESTING_OUTPUT_DIR" \
        --device "$TESTING_DEVICE"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    echo ""
    echo "=== TEST COMPLETION SUMMARY ==="
    echo "Exit code: $EXIT_CODE"
    echo "Total runtime: ${RUNTIME} seconds ($(($RUNTIME / 60)) minutes)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: Similarity retrieval testing completed!"
        show_test_results
    else
        echo "ERROR: Testing failed with exit code $EXIT_CODE"
        show_debug_info
        exit 1
    fi
}

show_test_results() {
    echo ""
    echo "Generated files in ${TESTING_OUTPUT_DIR}:"
    if command -v tree &> /dev/null; then
        tree "$TESTING_OUTPUT_DIR" -L 2
    else
        find "$TESTING_OUTPUT_DIR" -name "*.png" -o -name "*.json" | head -10
        echo "... (showing test result files)"
    fi
    
    echo ""
    echo "Results preview:"
    if [ -f "$TESTING_OUTPUT_DIR/test_similarity_results.json" ]; then
        echo "=== TEST SUMMARY ==="
        python3 -c "
import json
with open('$TESTING_OUTPUT_DIR/test_similarity_results.json', 'r') as f:
    data = json.load(f)
print(f\"Total test samples: {data['model_info']['total_test_samples']}\")
print(f\"Attention weights - Visual: {data['model_info']['average_attention_weights']['visual']:.3f}, Crowd: {data['model_info']['average_attention_weights']['crowd']:.3f}\")
print(f\"Number of similarity tests performed: {len(data['similarity_tests'])}\")
" 2>/dev/null || echo "Could not parse results JSON"
        echo ""
        echo "Detailed results: $TESTING_OUTPUT_DIR/test_similarity_results.json"
        echo "t-SNE visualization: $TESTING_OUTPUT_DIR/test_tsne_visualization.png"
    fi
    
    echo ""
    echo "Test features:"
    echo "  Loaded trained model and test data"
    echo "  Extracted 256D embeddings for all test clips"
    echo "  Built cosine and Euclidean similarity indices"
    echo "  Tested similarity retrieval on sample clips"
    echo "  Generated detailed results with file paths"
    echo "  Created t-SNE visualization of test data"
}

show_debug_info() {
    echo ""
    echo "Debug information:"
    echo "Model path: $TRAINING_OUTPUT_DIR/simple_soccer_model.pth"
    echo "Model exists: $([ -f "$TRAINING_OUTPUT_DIR/simple_soccer_model.pth" ] && echo "YES" || echo "NO")"
    echo "Test data directories:"
    echo "  Visual: $([ -d "$VISUAL_FEATURES_OUTPUT_DIR/test" ] && echo "EXISTS" || echo "MISSING")"
    echo "  Crowd: $([ -d "$CROWD_STATISTICAL_OUTPUT_DIR/test" ] && echo "EXISTS" || echo "MISSING")"
}

cleanup() {
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    
    echo ""
    echo "Resource usage summary:"
    if command -v nvidia-smi &> /dev/null; then
        echo "Final GPU status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    fi
    
    echo ""
    echo "Job completed at: $(date)"
}

main() {
    source_env_file
    setup_environment
    check_gpu
    print_job_info
    verify_required_files
    verify_test_data
    run_similarity_testing
    cleanup
}

trap cleanup EXIT
main