# Writes a FASTA (>id|gene|cycle) for every train/val/test split in BGF_clustering/train_test_val_final.

# Load necessary libraries
if (!require("Biostrings")) BiocManager::install("Biostrings")
library(data.table)
library(Biostrings)
library(tidyverse)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# 1. Configuration
input_dir <- "../../BGF_clustering/train_test_val_final"
thresholds <- c(10, 20, 30, 40, 50, 60, 70, 80, 90)
sets <- c("train", "val", "test")

# 2. Loop through each identity threshold and each split set
for (val in thresholds) {
  for (s in sets) {
    
    # Construct filename
    csv_file <- file.path(input_dir, paste0("bgf_", s, "_", val, ".csv"))
    fasta_file <- file.path(input_dir, paste0("bgf_", s, "_", val, ".fasta"))
    
    if (file.exists(csv_file)) {
      message(paste("Converting:", csv_file, "to FASTA..."))
      
      # Read CSV quickly with fread
      dt <- fread(csv_file)
      
      # Create headers in the format >id|gene|cycle
      # This keeps your metadata attached to the sequence for the model
      headers <- paste(dt$id, dt$gene, dt$cycle, sep = "|")
      
      # Create an AAStringSet (Amino Acid) or DNAStringSet
      # Using AAStringSet because BioGeoFormer works with protein sequences
      seq_set <- AAStringSet(dt$sequence)
      names(seq_set) <- headers
      
      # Write out as FASTA
      writeXStringSet(seq_set, filepath = fasta_file, format = "fasta")
      
      # Free up memory before next iteration
      rm(dt, seq_set, headers)
      gc()
      
      message(paste("Successfully wrote:", fasta_file))
    } else {
      warning(paste("File not found, skipping:", csv_file))
    }
  }
}