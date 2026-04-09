INPUT_FILE="hro_1981_2025_1min.txt"
OUTPUT_FILE="omni_1min_cleaned.txt"

echo "Starting to clean the dataset..."

awk '{
    # Check if the line is a valid data line starting with a year (19xx or 20xx)
    if ($1 ~ /^(19|20)[0-9]{2}$/) {
        
        # In this format, Columns 1-4 are Time. Columns 5 to 50 are Data.
        # We check if BOTH column 5 and column 6 indicate missing data (composed of 9s).
        if ($5 ~ /^9+\.?9*$/ && $6 ~ /^9+\.?9*$/) {
            # Delete the line (by not printing it) and log it to stderr
            printf "Deleted empty line -> Year: %s, DOY: %s, Hour: %s, Minute: %s (Col 5: %s, Col 6: %s)\n", $1, $2, $3, $4, $5, $6 > "/dev/stderr"
        } else {
            # Keep the line if at least one of them has valid data
            print $0
        }
    } else {
        # Keep the header and HTML tags intact for now (if you still need them)
        print $0
    }
}' "$INPUT_FILE" > "$OUTPUT_FILE"

echo "Done! Cleaned data saved to $OUTPUT_FILE"