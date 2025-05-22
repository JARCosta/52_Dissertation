#!/bin/bash

# This script cancels all Slurm jobs belonging to the user 'u035724'
# and then moves log files from the 'logs' folder to 'logs.old'.

user="u035724"
log_folder="logs"
old_log_folder="logs.old"

# Get the job IDs for the specified user.
job_ids=$(squeue -u "$user" --noheader | awk '{print $1}')

# Check if any jobs were found for the user.
if [ -z "$job_ids" ]; then
  echo "No running or pending jobs found for user '$user'."
else
  echo "Cancelling the following jobs for user '$user':"
  echo "$job_ids"

  # Cancel each job ID.
  for job_id in $job_ids; do
    scancel "$job_id"
    echo "Cancelled job ID: $job_id"
  done

  echo "All jobs for user '$user' have been cancelled."
fi

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

exit 0