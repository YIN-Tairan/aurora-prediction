#!/bin/bash

INPUT_FILE="omni_1min_data_1981_2025.txt"
OUTPUT_FILE="omni_1min_cleaned.txt"

echo "Starting to clean the dataset..."

awk '{
    # Check if the line is a valid data line starting with a year (19xx or 20xx)
    if ($1 ~ /^(19|20)[0-9]{2}$/) {
        missing_count = 0;
        
        # In your format, Column 1-4 are Time. Column 5 to 40 are Satellite Data. Column 41 is AE-index.
        # We will count how many satellite data columns are missing (checking cols 5 through 40).
        for (i = 5; i <= 40; i++) {
            if ($i ~ /^9+\.?9*$/) {
                missing_count++;
            }
        }
        
        # There are 36 satellite columns here. 
        # If 34 or more of them are missing, we consider the line effectively empty.
        if (missing_count >= 15) {
            printf "Deleted empty line -> Year: %s, DOY: %s, Hour: %s, Minute: %s (AE-index was %s)\n", $1, $2, $3, $4, $41 > "/dev/stderr"
        } else {
            # Print lines that have actual satellite data
            print $0
        }
    } else {
        # Keep the header and HTML tags intact for now (if you still need them)
        print $0
    }
}' "$INPUT_FILE" > "$OUTPUT_FILE"

echo "Done! Cleaned data saved to $OUTPUT_FILE"
