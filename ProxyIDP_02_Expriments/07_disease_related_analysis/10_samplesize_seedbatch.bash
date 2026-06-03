#!/bin/bash

# ===================== Configuration =====================
# Path to the Python script (ensure the path is correct)
CODE_FILE="14_samplesize_control_batch_dispred_comparison.py"
# Name of the log directory
LOG_DIR="samplesize_analysis_logs"
# Seed range (0-9)
SEED_START=0
SEED_END=9
# Starting CPU core number (seed 0 uses 20, seed 1 uses 21 ... seed 9 uses 29)
CPU_START=20
# ==========================================================

# Create log directory
mkdir -p ${LOG_DIR}
echo "Log directory created: ${LOG_DIR}"

# Clear the process ID record file (avoid duplication)
> ${LOG_DIR}/process_ids.txt

# Submit seed tasks in a loop (bind to corresponding CPU cores)
for SEED in $(seq ${SEED_START} ${SEED_END})
do
    # Calculate the CPU core number for the current seed
    CPU_CORE=$((CPU_START + SEED))
    # Define log file name (includes seed number)
    LOG_FILE="${LOG_DIR}/seed_${SEED}_cpu${CPU_CORE}.log"
    
    # Run the Python script in the background, bind to the specified CPU core,
    # redirect output and errors to the log file
    echo "Submitting task for seed ${SEED} (bound to CPU ${CPU_CORE}), log file: ${LOG_FILE}"
    nohup taskset -c ${CPU_CORE} python ${CODE_FILE} ${SEED} > ${LOG_FILE} 2>&1 &
    
    # Record the mapping of process ID, seed, and CPU core
    echo "Seed ${SEED} | CPU ${CPU_CORE} | Process ID: $!" >> ${LOG_DIR}/process_ids.txt
done

# Information prompt
echo -e "\n===== Task Submission Completed ====="
echo "All seed tasks (${SEED_START}-${SEED_END}) are running in the background, bound to CPU cores ${CPU_START}-$((CPU_START+SEED_END))"
echo "Check task status: ps -ef | grep ${CODE_FILE}"
echo "Check CPU binding: ps -o pid,pcpu,cmd,psr -p \$(cat ${LOG_DIR}/process_ids.txt | awk '{print \$8}')"
echo "Check log for a single seed: tail -f ${LOG_DIR}/seed_XXX_cpuYY.log (replace XXX with seed, YY with CPU)"
echo "View all process/CPU mappings: cat ${LOG_DIR}/process_ids.txt"
echo "Stop all tasks: pkill -f ${CODE_FILE}"