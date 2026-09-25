#!/usr/bin/env Rscript
# Keeps a pathway at a given threshold only if train > 100, val > 30 and test > 30
# sequences; otherwise it is removed from all three sets.
# Reads BGF_clustering/graphpart_raw (train) and BGF_clustering/cleaned (val/test);
# writes BGF_clustering/train_test_val_final.

library(data.table)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

MIN_TRAIN <- 100
MIN_VAL   <- 30
MIN_TEST  <- 30

thresholds <- seq(10, 90, by = 10)

out_dir <- "../../BGF_clustering/train_test_val_final"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

summary_rows <- vector("list", length(thresholds))

for (i in seq_along(thresholds)) {
  T <- thresholds[i]
  message(sprintf("── Threshold %d%% ──────────────────────────────", T))

  train <- fread(sprintf("../../BGF_clustering/graphpart_raw/bgf_train_%d.csv", T))
  val   <- fread(sprintf("../../BGF_clustering/cleaned/bgf_val_%d.csv", T))
  test  <- fread(sprintf("../../BGF_clustering/cleaned/bgf_test_%d.csv", T))

  train_n <- train[, .N, by = cycle]
  val_n   <- val[,   .N, by = cycle]
  test_n  <- test[,  .N, by = cycle]

  all_cycles <- sort(unique(c(train_n$cycle, val_n$cycle, test_n$cycle)))

  get_n <- function(dt, cyc) {
    v <- dt$N[dt$cycle == cyc]
    if (length(v) == 0) 0L else v
  }

  keep_cycles <- character(0)
  for (cyc in all_cycles) {
    t_n  <- get_n(train_n, cyc)
    v_n  <- get_n(val_n, cyc)
    te_n <- get_n(test_n, cyc)
    if (t_n > MIN_TRAIN && v_n > MIN_VAL && te_n > MIN_TEST) {
      keep_cycles <- c(keep_cycles, cyc)
    }
  }

  train_f <- train[cycle %in% keep_cycles]
  val_f   <- val[cycle %in% keep_cycles]
  test_f  <- test[cycle %in% keep_cycles]

  fwrite(train_f, file.path(out_dir, sprintf("bgf_train_%d.csv", T)), quote = TRUE)
  fwrite(val_f,   file.path(out_dir, sprintf("bgf_val_%d.csv", T)), quote = TRUE)
  fwrite(test_f,  file.path(out_dir, sprintf("bgf_test_%d.csv", T)), quote = TRUE)

  message(sprintf("  Cycles: %d total, %d kept, %d removed",
                   length(all_cycles), length(keep_cycles),
                   length(all_cycles) - length(keep_cycles)))
  message(sprintf("  Train: %d -> %d | Val: %d -> %d | Test: %d -> %d",
                   nrow(train), nrow(train_f), nrow(val), nrow(val_f),
                   nrow(test), nrow(test_f)))

  # Per-cycle before/after breakdown for the summary table
  per_cycle <- list()
  for (set_name in c("Train", "Val", "Test")) {
    dt <- switch(set_name, Train = train_n, Val = val_n, Test = test_n)
    for (cyc in keep_cycles) {
      per_cycle[[paste0(set_name, "_", cyc)]] <- get_n(dt, cyc)
    }
  }

  summary_rows[[i]] <- c(
    list(
      Threshold      = T,
      Cycles_Total   = length(all_cycles),
      Cycles_Kept    = length(keep_cycles),
      Cycles_Removed = length(all_cycles) - length(keep_cycles),
      Train_Before   = nrow(train),
      Train_After    = nrow(train_f),
      Test_Before    = nrow(test),
      Test_After     = nrow(test_f),
      Val_Before     = nrow(val),
      Val_After      = nrow(val_f)
    ),
    per_cycle
  )
}

summary_dt <- rbindlist(summary_rows, fill = TRUE)
fwrite(summary_dt, file.path(out_dir, "filtering_summary.csv"))
message("Wrote ", file.path(out_dir, "filtering_summary.csv"))
