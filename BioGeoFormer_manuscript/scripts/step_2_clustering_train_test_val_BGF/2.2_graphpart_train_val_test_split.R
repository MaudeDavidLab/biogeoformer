# Consumes raw GraphPart output (per-threshold two-partition assignment:
# cluster 0 = train, cluster 1 = test+val combined), then splits the
# test+val partition 50/50 stratified by cycle.
#

# The GraphPart invocation itself that produces
# BioGeoFormer_stratified_splits_<N>.csv is 2.1_run_graphpart.slurm (run once
# per threshold on the HPC, graph-part CLI, mmseqs2 backend).
library(data.table)
library(tidyverse)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# 1. Load the main database once and convert to data.table for speed
bgf_dt <- fread("../../BioGeoFormer_db/BioGeoFormer_db.csv")
bgf_dt[, id := as.character(id)]

# 2. Define the thresholds
thresholds <- c(10, 20, 30, 40, 50, 60, 70, 80, 90)

# 3. Process each threshold
for (val in thresholds) {

  input_file <- paste0("../../BGF_clustering/graphpart_raw/BioGeoFormer_stratified_splits_", val, ".csv")
  
  if (file.exists(input_file)) {
    message(paste("Processing identity threshold:", val, "%"))
    
    # Read GraphPart output
    gp_raw <- fread(input_file)
    
    # Clean GraphPart data: Col 1 is ID, last col is Cluster
    gp_clean <- gp_raw[, .(id = as.character(get(names(gp_raw)[1])), 
                           cluster = get(names(gp_raw)[ncol(gp_raw)]))]
    
    # Fast join using data.table
    merged_dt <- merge(bgf_dt, gp_clean, by = "id")
    
    # Train = cluster 0, Test/Val combined = cluster 1
    train_dt <- merged_dt[cluster == 0]
    testval_dt <- merged_dt[cluster == 1]
    
    # Split test/val 50/50 stratified by class (label column)
    set.seed(42)
    val_dt <- testval_dt[, .SD[sample(.N, floor(.N / 2))], by = cycle]
    test_dt <- testval_dt[!id %in% val_dt$id]
    
    # Define output directory
    out_dir <- "../../BGF_clustering/graphpart_raw/"
    
    # Save splits
    fwrite(train_dt, file.path(out_dir, paste0("bgf_train_", val, ".csv")))
    fwrite(val_dt,   file.path(out_dir, paste0("bgf_val_", val, ".csv")))
    fwrite(test_dt,  file.path(out_dir, paste0("bgf_test_", val, ".csv")))
    
    message(paste("Successfully saved splits for", val, "% identity."))
    message(paste("  Train:", nrow(train_dt), "| Val:", nrow(val_dt), "| Test:", nrow(test_dt)))
    
  } else {
    warning(paste("File missing:", input_file))
  }
}
