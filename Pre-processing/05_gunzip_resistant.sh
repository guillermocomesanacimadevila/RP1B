#!bin/bash

# Create the output directory if it doesn't exist
mkdir -p gunziped_resistant

# Loop through all .vcf.gz files
# Keep the original .vcf.gz just in case | parse .vcf -> new dir
for file in *.vcf.gz; do
    if [[ -f "$file" ]]; then
        gunzip -c "$file" > "gunziped_resistant/${file%.gz}"
        echo "Uncompressed: $file -> gunziped_resistant/${file%.gz}"
    fi
done

echo "All files have been gunzipped into the 'gunziped' directory."
