#!/bin/bash

cd ../build/test/

# Loop through each executable that starts with 'test'
for test_script in test*; do
    # Check if the file is an executable before running
    if [[ -x "$test_script" ]]; then
        ./"$test_script"
    else
        echo "Skipping non-executable file: $test_script"
    fi
done