#!/bin/bash
#SBATCH --job-name=create_subset
#SBATCH --output=$LOG_DIR/create_subset_%j.out
#SBATCH --error=$LOG_DIR/create_subset_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
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
    
    mkdir -p "$SUBSET_OUTPUT_DIR"
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
    echo "Starting subset creation job"
    echo "Job started at: $(date)"
    echo "Running on node: $(hostname)"
    echo "Source directory: $ORGANIZED_CLIPS_DIR"
    echo "Target directory: $SUBSET_OUTPUT_DIR"
    echo "Total videos: $SUBSET_TOTAL_VIDEOS"
    echo "Proportions:"
    echo "  Background: $SUBSET_BACKGROUND_PROP"
    echo "  Before goal: $SUBSET_BEFORE_GOAL_PROP"
    echo "  Free kicks: $SUBSET_FREE_KICKS_PROP"
    echo "  Penalties: $SUBSET_PENALTIES_PROP"
    echo "  Shots no goals: $SUBSET_SHOTS_NO_GOALS_PROP"
}

run_subset_creation() {
    echo "Starting subset creation at $(date)"
    
    python "$PROJECT_ROOT/src/data/create_subset_dataset.py" \
        "$ORGANIZED_CLIPS_DIR" \
        "$SUBSET_OUTPUT_DIR" \
        "$SUBSET_TOTAL_VIDEOS" \
        --background-prop "$SUBSET_BACKGROUND_PROP" \
        --before-goal-prop "$SUBSET_BEFORE_GOAL_PROP" \
        --free-kicks-prop "$SUBSET_FREE_KICKS_PROP" \
        --penalties-prop "$SUBSET_PENALTIES_PROP" \
        --shots-no-goals-prop "$SUBSET_SHOTS_NO_GOALS_PROP"
    
    if [ $? -eq 0 ]; then
        echo "Subset creation completed successfully at $(date)"
    else
        echo "Subset creation failed with exit code $? at $(date)"
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
    print_job_info
    run_subset_creation
    cleanup
}

trap cleanup EXIT
main