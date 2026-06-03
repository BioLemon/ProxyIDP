#!/bin/bash

# ===================== Configuration =====================
# Path to the code file
CODE_FILE="17_samplesize_lower_than_real.py"
# Log directory name
LOG_DIR="samplesize_analysis_logs"
SEED_START=0
SEED_END=9
CPU_START=20
# =========================================================

# Create log directory
mkdir -p ${LOG_DIR}
echo "Log directory created: ${LOG_DIR}"

# Clear the process ID record file
> ${LOG_DIR}/process_ids.txt

# Submit seed tasks in a loop 
for SEED in $(seq ${SEED_START} ${SEED_END})
do
    # Calculate the CPU core number for the current seed
    CPU_CORE=$((CPU_START + SEED))
    # Define log file name
    LOG_FILE="${LOG_DIR}/seed_${SEED}_low_cpu${CPU_CORE}.log"
    
    # Run the Python script in the background, bind to the specified CPU core,
    # redirect output and errors to the log file
    echo "Submitting task for seed ${SEED} (bind to CPU ${CPU_CORE}), log file: ${LOG_FILE}"
    nohup taskset -c ${CPU_CORE} python ${CODE_FILE} ${SEED} > ${LOG_FILE} 2>&1 &
    
    # Record the mapping of process ID, seed, and CPU core
    echo "Seed ${SEED} | CPU ${CPU_CORE} | Process ID: $!" >> ${LOG_DIR}/process_ids.txt
done

