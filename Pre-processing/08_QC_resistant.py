# Remove INDEls and Uninformative SNPs from Susceptible sample VCFs
import os
import pandas as pd

# Input and output dirs
input_dit = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_resistant/gunziped_resistant/csv_converted"
output_dir = "/home/jovyan/DOUBLE_DESCENT/Data/downloaded_resistant/gunziped_resistant/csv_converted/post_QC_resistant"

# Ensure output dir exsists
os.makedirs(output_dir, exist_ok=True)

# List all 4 DNA nucleotides to filter through ALT & REF
valid_nucleotides = ["A", "T", "C", "G"]

# Process each CSV-converted VCF within input directory
for file in os.listdir(input_dit):
    if file.endswith(".csv"):
        input_file_path = os.path.join(input_dit, file)
        output_file_path = os.path.join(output_dir, file)

        # Read CSV file
        df = pd.read_csv(input_file_path)

        # Keep only SNPs with 1 index - avoid INDELs
        if "REF" in df.columns and "ALT" in df.columns:
            df_filtered = df[
                (df["REF"].str.len() == 1) &
                (df["ALT"].str.len() == 1) &
                (df["REF"].isin(valid_nucleotides)) &
                (df["ALT"].isin(valid_nucleotides))
            ]

            # Save back to CSV
            df_filtered.to_csv(output_file_path, index=False)

            print(f"Processed: {file} -> Saved to {output_file_path}")
        else:
            print(f"Skipped {file}: Missing REF or ALT columns")

print("Done!")

# === Additional notes === #
# QC
# Quality = HIGH - Isonazid
# 250 samples = Resistant - 250 = susceptible
# Iterate through list(vcf1["ALT"].unique()) and look for = NULL positions
# Make script that iterates through both positions of all CSV-converted VCFs and removes INDELs
