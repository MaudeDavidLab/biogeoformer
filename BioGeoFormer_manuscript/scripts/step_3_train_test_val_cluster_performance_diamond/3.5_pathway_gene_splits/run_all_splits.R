#!/usr/bin/env Rscript
# =============================================================================
# Splits every bgf_train_<N>.csv into per-pathway, per-gene FASTA/CSV files under
# BGF_clustering/train_test_val_fasta_bypathway/ (input for HMM building).
# =============================================================================

suppressMessages(library(data.table))

script_dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
if (length(script_dir) == 0 || script_dir == "") script_dir <- "scripts/step_3_train_test_val_cluster_performance_diamond/3.5_pathway_gene_splits"

source(file.path(script_dir, "utils_fasta.R"))
source(file.path(script_dir, "split_by_gene_pathway.R"))

repo_root <- normalizePath(file.path(script_dir, "..", "..", ".."))
input_dir <- file.path(repo_root, "BGF_clustering", "train_test_val_final")
out_root <- file.path(repo_root, "BGF_clustering", "train_test_val_fasta_bypathway")

splits <- sprintf("bgf_train_%d", seq(10, 90, by = 10))

all_manifests <- vector("list", length(splits))

for (i in seq_along(splits)) {
  csv_path <- file.path(input_dir, paste0(splits[i], ".csv"))
  if (!file.exists(csv_path)) {
    warning(sprintf("Skipping missing file: %s", csv_path))
    next
  }
  all_manifests[[i]] <- split_one_file(csv_path, out_root)
}

combined <- rbindlist(all_manifests, use.names = TRUE)
fwrite(combined, file.path(out_root, "manifest_all_splits.csv"))

message(sprintf(
  "\nDone. %d splits processed, %d gene/pathway/split rows in manifest_all_splits.csv",
  uniqueN(combined$split), nrow(combined)
))
