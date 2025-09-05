#!/bin/bash
#SBATCH --job-name=track_videos
#SBATCH --output=$LOG_DIR/track_videos_%j.out
#SBATCH --error=$LOG_DIR/track_videos_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
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
    
    mkdir -p "$TRACKING_OUTPUT_DIR"
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
    
    export PYTHONPATH="$BYTETRACK_HOME:$PYTHONPATH"
    cd "$BYTETRACK_HOME"
}

print_job_info() {
    echo "Starting video tracking job"
    echo "Job started at: $(date)"
    echo "Running on node: $(hostname)"
    echo "Input directory: $TRACKING_INPUT_DIR"
    echo "Output directory: $TRACKING_OUTPUT_DIR"
    echo "ByteTrack home: $BYTETRACK_HOME"
    echo "Experiment file: $TRACKING_EXP_FILE"
    echo "Checkpoint: $TRACKING_CHECKPOINT"
    echo "Test size: $TRACKING_TEST_SIZE"
}

process_videos_in_directory() {
    local dir_path="$1"
    local counter=0
    local total_videos=0

    while IFS= read -r -d '' video_file; do
        ((total_videos++))
    done < <(find "$dir_path" -name "*.mkv" -print0)

    echo "Found $total_videos MKV files in $dir_path"

    while IFS= read -r -d '' video_file; do
        ((counter++))

        echo "[$counter/$total_videos] Processing video: $(basename "$video_file") at $(date)"
        echo "Full path: $video_file"

        python "$PROJECT_ROOT/src/tracking/track_videos.py" \
            "$video_file" \
            "$TRACKING_OUTPUT_DIR" \
            --exp-file "$TRACKING_EXP_FILE" \
            --ckpt "$TRACKING_CHECKPOINT" \
            --fp16 --fuse --save-result \
            --tsize "$TRACKING_TEST_SIZE"

        if [ $? -ne 0 ]; then
            echo "First attempt failed, trying with tsize=800..."
            python "$PROJECT_ROOT/src/tracking/track_videos.py" \
                "$video_file" \
                "$TRACKING_OUTPUT_DIR" \
                --exp-file "$TRACKING_EXP_FILE" \
                --ckpt "$TRACKING_CHECKPOINT" \
                --fp16 --fuse --save-result \
                --tsize 800
        fi

        if [ $? -ne 0 ]; then
            echo "Second attempt failed, trying without explicit tsize..."
            python "$PROJECT_ROOT/src/tracking/track_videos.py" \
                "$video_file" \
                "$TRACKING_OUTPUT_DIR" \
                --exp-file "$TRACKING_EXP_FILE" \
                --ckpt "$TRACKING_CHECKPOINT" \
                --save-result
        fi

        if [ $? -eq 0 ]; then
            echo "Successfully processed $(basename "$video_file")"
        else
            echo "ERROR: Failed to process $(basename "$video_file") with exit code $?"
            echo "Continuing with next video..."
        fi

        echo "---"
    done < <(find "$dir_path" -name "*.mkv" -print0)

    return $counter
}

run_tracking() {
    echo "Starting tracking at $(date)"
    
    if [ ! -d "$TRACKING_INPUT_DIR" ]; then
        echo "ERROR: Input directory not found at $TRACKING_INPUT_DIR"
        exit 1
    fi

    total_processed=0

    for split_dir in "$TRACKING_INPUT_DIR"/*; do
        if [ -d "$split_dir" ] && [[ "$(basename "$split_dir")" =~ ^(train|val|test)$ ]]; then
            split_name=$(basename "$split_dir")
            echo "Processing split: $split_name"

            for category_dir in "$split_dir"/*; do
                if [ -d "$category_dir" ]; then
                    category_name=$(basename "$category_dir")
                    echo "Processing category: $split_name/$category_name"

                    process_videos_in_directory "$category_dir"
                    category_count=$?
                    total_processed=$((total_processed + category_count))

                    echo "Completed processing $category_count videos in $split_name/$category_name"
                fi
            done
        fi
    done

    if [ $total_processed -eq 0 ]; then
        echo "WARNING: No videos were processed"
    else
        echo "Processing completed for $total_processed videos at $(date)"
    fi
}

generate_summary() {
    echo "Generating summary report..."
    
    local summary_file="$TRACKING_OUTPUT_DIR/tracking_summary.txt"
    
    find "$TRACKING_OUTPUT_DIR" -name "*.txt" | wc -l > "$summary_file"
    echo "Total tracking result files: $(cat "$summary_file")" >> "$summary_file"
    echo "Processing completed at: $(date)" >> "$summary_file"
    
    echo "Output directory structure:" >> "$summary_file"
    find "$TRACKING_OUTPUT_DIR" -type d >> "$summary_file"
    
    echo "Summary saved to: $summary_file"
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
    run_tracking
    generate_summary
    cleanup
}

trap cleanup EXIT
main