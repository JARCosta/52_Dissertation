#!/bin/bash

# Array of n_points values
n_points_values=(200 210 220 230 240 250)

# Loop through the array
for n_points in "${n_points_values[@]}"; do
  # Define the output file name based on n_points
  output_file="run_mvu_n${n_points}.sh"

  # Create the script content with the current n_points value
  script_content=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=mvu-n${n_points}
#SBATCH --mem=30G # ram
#SBATCH --mincpus=3
#SBATCH --cpus-per-task=3
#SBATCH --output=logs/job.%A.mvu.n${n_points}.out # %a
#SBATCH --gres=shard:0
#SBATCH --time=24:00:00

source .venv/bin/activate
python code/launcher.py --paper mvu --n_points ${n_points} #--threaded
EOF
)

  # Write the script content to the output file
  echo "$script_content" > "$output_file"

  # Make the script executable (optional, but recommended)
  chmod +x "$output_file"

  # Print a message indicating that the file has been created
  echo "Created script: $output_file with n_points = $n_points"
done

echo "All scripts created successfully."
