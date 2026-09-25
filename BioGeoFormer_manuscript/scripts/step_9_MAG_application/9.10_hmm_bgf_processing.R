library(dplyr)
library(readr)

# Reshapes the filtered hmmscan hits (9.4b) into the target_name/pathway table used by 9.14.

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

good_hits_fp <- "../../cold_seep_MAG_application/HMM_BGF_output/MAGs_vs_bgf_train_50_good_hits.csv"
out_fp <- "../../cold_seep_MAG_application/HMM_BGF_processed/HMM_BGF_processed.csv"

good_hits <- read_csv(good_hits_fp, show_col_types = FALSE)

hmm_bgf_processed <- good_hits %>%
  rename(
    target_name = query_name,   # MAG protein ID
    pathway     = predicted_pathway,
    gene        = predicted_gene
  )

dir.create(dirname(out_fp), recursive = TRUE, showWarnings = FALSE)
write.csv(hmm_bgf_processed, out_fp, row.names = FALSE)

message(sprintf("Wrote %d protein annotations to %s", nrow(hmm_bgf_processed), out_fp))
