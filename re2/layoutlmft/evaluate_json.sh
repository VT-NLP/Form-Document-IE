#!/bin/bash

# Change directory to the folder containing the subfolders with checkpoints
cd ...

# Initialize variables to hold the maximum f1 score and the name of the folder with that score
max_score=0
max_folder=""

# Loop through each folder in the current directory
for folder in */ ; do
    # Change directory to the current folder
    cd "$folder"

    # Check if the "all_results.json" file exists in the current folder
    if [ -f "all_results.json" ]; then
        # Get the eval_f1 score from the "all_results.json" file
        eval_f1=$(jq '.eval_f1' all_results.json)

        # Check if the eval_f1 score is greater than the current maximum score
        if awk 'BEGIN {exit !('"$eval_f1"' > '"$max_score"')}'; then
            # Update the maximum score and folder name
            max_score=$eval_f1
            max_folder="$folder"
        fi
    fi

    # Change directory back to the parent folder
    cd ..
done

# Print the folder with the highest eval_f1 score
echo "Folder with highest eval_f1 score: $max_folder"
