#!/bin/bash
#SBATCH --job-name=soccer_training_final
#SBATCH --output=$LOG_DIR/soccer_training_final_%j.out
#SBATCH --error=$LOG_DIR/soccer_training_final_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

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
    
    mkdir -p "$TRAINING_OUTPUT_DIR"
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
        echo "WARNING: nvidia-smi not available, cannot check GPU status"
    fi
}

check_dependencies() {
    echo "Checking for required packages..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
        echo "ERROR: PyTorch not found"; exit 1;
    }
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || {
        echo "WARNING: Could not check CUDA availability"
    }
    python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || {
        echo "ERROR: NumPy not found"; exit 1;
    }
    python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')" || {
        echo "ERROR: scikit-learn not found"; exit 1;
    }
    python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')" || {
        echo "ERROR: Matplotlib not found"; exit 1;
    }
    python -c "import h5py; print(f'h5py version: {h5py.__version__}')" || {
        echo "ERROR: h5py not found"; exit 1;
    }
}

print_job_info() {
    echo "=== SOCCER MULTI-TASK MODEL WITH T-SNE ==="
    echo "Job started at: $(date)"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Running on node(s): $SLURM_JOB_NODELIST"
    echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
    echo "Memory allocated: 32G"
    echo "GPU allocated: $CUDA_VISIBLE_DEVICES"
    echo "Visual features: $VISUAL_FEATURES_OUTPUT_DIR"
    echo "Crowd features: $CROWD_STATISTICAL_OUTPUT_DIR"
    echo "Output directory: $TRAINING_OUTPUT_DIR"
    echo ""
}

verify_data_structure() {
    echo "Data structure verification:"

    if [ -d "$VISUAL_FEATURES_OUTPUT_DIR" ]; then
        visual_train_files=$(find "$VISUAL_FEATURES_OUTPUT_DIR/train" -name "*_features.h5" 2>/dev/null | wc -l)
        visual_val_files=$(find "$VISUAL_FEATURES_OUTPUT_DIR/val" -name "*_features.h5" 2>/dev/null | wc -l)
        echo "  Visual features - Train: $visual_train_files files, Val: $visual_val_files files"
    else
        echo "ERROR: Visual features directory not found: $VISUAL_FEATURES_OUTPUT_DIR"
        exit 1
    fi

    if [ -d "$CROWD_STATISTICAL_OUTPUT_DIR" ]; then
        crowd_train_files=$(find "$CROWD_STATISTICAL_OUTPUT_DIR/train" -name "*_crowd_features.h5" 2>/dev/null | wc -l)
        crowd_val_files=$(find "$CROWD_STATISTICAL_OUTPUT_DIR/val" -name "*_crowd_features.h5" 2>/dev/null | wc -l)
        echo "  Crowd features - Train: $crowd_train_files files, Val: $crowd_val_files files"
    else
        echo "ERROR: Crowd features directory not found: $CROWD_STATISTICAL_OUTPUT_DIR"
        exit 1
    fi

    echo ""
    echo "Quick data sample check:"
    python3 << EOF
import h5py
import numpy as np
from pathlib import Path

visual_dir = Path("$VISUAL_FEATURES_OUTPUT_DIR/train")
crowd_dir = Path("$CROWD_STATISTICAL_OUTPUT_DIR/train")

visual_file = visual_dir / "background_features.h5"
crowd_file = crowd_dir / "background_crowd_features.h5"

if visual_file.exists() and crowd_file.exists():
    with h5py.File(visual_file, 'r') as vf, h5py.File(crowd_file, 'r') as cf:
        visual_videos = list(vf.keys())[:3]
        crowd_videos = list(cf.keys())[:3]
        
        print(f"  Visual sample videos: {visual_videos}")
        print(f"  Crowd sample videos: {crowd_videos}")
        
        if visual_videos:
            visual_sample = vf[visual_videos[0]][:]
            print(f"  Visual feature shape: {visual_sample.shape}")
        
        if crowd_videos:
            crowd_sample = cf[crowd_videos[0]][:]
            print(f"  Crowd feature shape: {crowd_sample.shape}")
else:
    print("  WARNING: Could not find sample files for verification")
EOF
}

run_training() {
    echo ""
    echo "=== STARTING MULTI-TASK TRAINING WITH T-SNE ==="
    echo "Model features:"
    echo "  - Attention-based fusion layer"
    echo "  - Multi-task learning (clustering + goal prediction)"
    echo "  - Standard K-means clustering evaluation"
    echo "  - t-SNE visualization of learned embeddings"
    echo "  - Comprehensive result analysis and plots"
    echo "Expected runtime: 3-4 hours (including t-SNE computation)"
    echo ""

    START_TIME=$(date +%s)

    python "$PROJECT_ROOT/src/multitask_model/soccer_training_final.py" \
        "$VISUAL_FEATURES_OUTPUT_DIR" \
        "$CROWD_STATISTICAL_OUTPUT_DIR" \
        "$TRAINING_OUTPUT_DIR" \
        --fusion-dim "$TRAINING_FUSION_DIM" \
        --num-clusters "$TRAINING_NUM_CLUSTERS" \
        --epochs "$TRAINING_EPOCHS" \
        --batch-size "$TRAINING_BATCH_SIZE" \
        --device "$TRAINING_DEVICE"

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    echo ""
    echo "=== JOB COMPLETION SUMMARY ==="
    echo "Exit code: $EXIT_CODE"
    echo "Total runtime: ${RUNTIME} seconds ($(($RUNTIME / 60)) minutes)"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "SUCCESS: Soccer multi-task training with t-SNE completed!"
        show_results_summary
    else
        echo "ERROR: Training failed with exit code $EXIT_CODE"
        show_debug_info
        exit 1
    fi
}

show_results_summary() {
    echo ""
    echo "Generated files in ${TRAINING_OUTPUT_DIR}:"
    if command -v tree &> /dev/null; then
        tree "$TRAINING_OUTPUT_DIR" -L 2
    else
        find "$TRAINING_OUTPUT_DIR" -name "*.png" -o -name "*.json" -o -name "*.txt" -o -name "*.pth" | head -10
        echo "... (showing main output files)"
    fi
    
    echo ""
    echo "Key results:"
    if [ -f "$TRAINING_OUTPUT_DIR/TRAINING_SUMMARY.txt" ]; then
        echo "=== FINAL PERFORMANCE ==="
        grep -A 6 "FINAL PERFORMANCE:" "$TRAINING_OUTPUT_DIR/TRAINING_SUMMARY.txt" | tail -5
        echo ""
        echo "=== OPTIMAL CLUSTERING ==="
        grep -A 10 "CLUSTERING EVALUATION:" "$TRAINING_OUTPUT_DIR/TRAINING_SUMMARY.txt" | head -8
        echo ""
        echo "Full summary: $TRAINING_OUTPUT_DIR/TRAINING_SUMMARY.txt"
    fi
    
    echo ""
    echo "Model features implemented:"
    echo "  Attention-based fusion with modality balancing"
    echo "  Multi-task learning (clustering + goal prediction)"
    echo "  Comprehensive clustering evaluation (k=2 to 10)"
    echo "  t-SNE visualization of 256D embeddings"
    echo "  Attention weight tracking over training"
    echo "  Model saved and comprehensive results generated"
    echo "  2x4 subplot layout with all visualizations"
    
    if [ -f "$TRAINING_OUTPUT_DIR/training_results.png" ]; then
        echo "  ✓ t-SNE plots generated successfully"
    fi
}

show_debug_info() {
    echo ""
    echo "Debug information:"
    echo "Python environment: $(which python)"
    echo "Current directory: $(pwd)"
    
    if [ -f "$TRAINING_OUTPUT_DIR/simple_soccer_model.pth" ]; then
        echo "NOTE: Model file was saved despite error"
    fi
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
    check_dependencies
    print_job_info
    verify_data_structure
    run_training
    cleanup
}

trap cleanup EXIT
main