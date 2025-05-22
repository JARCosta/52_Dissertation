#!/bin/bash


log_folder="logs"
old_log_folder="logs.old"

# Check if the 'logs' folder exists.
if [ -d "$log_folder" ]; then
  # Create the 'logs.old' folder if it doesn't exist.
  mkdir -p "$old_log_folder"

  # Move all files within the 'logs' folder to 'logs.old'.
  find "$log_folder" -type f -exec mv {} "$old_log_folder" \;

  echo "Moved all log files from '$log_folder' to '$old_log_folder'."
else
  echo "The folder '$log_folder' does not exist. No log files to move."
fi



# Find all the run_mvu_n*.sh scripts in the current directory
script_files=$(find . -name "run_mvu_n*.sh" | sort)

# Check if any scripts were found
if [ -z "$script_files" ]; then
  echo "No run_mvu_n*.sh scripts found in the current directory."
  exit 1
fi

# Loop through the found scripts and submit them using sbatch
for script_file in $script_files; do
  # Remove the "./" prefix
  script_name=$(basename "$script_file")
  sbatch "$script_name"
  echo "Submitted job: $script_name"
done

echo "All scripts submitted."
sleep 1
squeue --format="%.18i %.9P %p %.25j %.8u %.8T %.10M %.9l %.6D %R %c %b "