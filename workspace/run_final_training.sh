#!/bin/bash
# Run final model training on multiple GPUs in parallel

# Configuration
GPUS=(0 1 2 3)
COMMANDS_FILE="workspace/commands/commands_final_model_training_clean.txt"
LOG_DIR="workspace/logs/final_training"

# Create log directory
mkdir -p "$LOG_DIR"

# Read commands into array
mapfile -t COMMANDS < "$COMMANDS_FILE"
TOTAL_COMMANDS=${#COMMANDS[@]}

echo "=================================================="
echo "Final Model Training - Parallel GPU Execution"
echo "=================================================="
echo "Total commands: $TOTAL_COMMANDS"
echo "GPUs: ${GPUS[@]}"
echo "Commands file: $COMMANDS_FILE"
echo "Log directory: $LOG_DIR"
echo "=================================================="
echo ""

# Function to run command on specific GPU
run_command() {
    local cmd=$1
    local gpu=$2
    local idx=$3
    local dataset_name=$(echo "$cmd" | grep -oP '(?<=--dataset_name )\S+')
    
    echo "[GPU $gpu] Starting: $dataset_name (command $idx/$TOTAL_COMMANDS)"
    
    # Modify command to add device_no
    local modified_cmd=$(echo "$cmd" | sed "s/--category/--device_no $gpu --category/")
    
    # Run the modified command
    eval "$modified_cmd" > "$LOG_DIR/${dataset_name}.log" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu] ✓ Completed: $dataset_name"
    else
        echo "[GPU $gpu] ✗ Failed: $dataset_name (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Track running jobs
declare -A GPU_JOBS
for gpu in "${GPUS[@]}"; do
    GPU_JOBS[$gpu]=""
done

# Command index
cmd_idx=0

# Main loop
while [ $cmd_idx -lt $TOTAL_COMMANDS ] || [ ${#GPU_JOBS[@]} -gt 0 ]; do
    # Check for available GPUs and start new jobs
    for gpu in "${GPUS[@]}"; do
        # Check if GPU is free
        if [ -z "${GPU_JOBS[$gpu]}" ] || ! kill -0 "${GPU_JOBS[$gpu]}" 2>/dev/null; then
            # GPU is free, start new job if available
            if [ $cmd_idx -lt $TOTAL_COMMANDS ]; then
                cmd="${COMMANDS[$cmd_idx]}"
                ((cmd_idx++))
                
                # Start job in background
                run_command "$cmd" "$gpu" "$cmd_idx" &
                GPU_JOBS[$gpu]=$!
            else
                # No more commands, remove GPU from tracking
                unset GPU_JOBS[$gpu]
            fi
        fi
    done
    
    # Sleep before checking again
    sleep 2
done

echo ""
echo "=================================================="
echo "All training jobs completed!"
echo "=================================================="
echo "Logs saved to: $LOG_DIR"
echo ""
echo "Next steps:"
echo "  1. Check logs: ls -lh $LOG_DIR"
echo "  2. Collect results: conda run -n ADMET python workspace/src/collect_final_results.py"
echo "=================================================="
