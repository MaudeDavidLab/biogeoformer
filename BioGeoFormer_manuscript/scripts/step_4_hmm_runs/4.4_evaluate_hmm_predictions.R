#!/usr/bin/env Rscript
# =============================================================================
# Scores HMM pathway predictions on val/test sets from hmmscan output (HMM building and
# hmmscan were run on the HPC). Prediction = pathway of the best hit's gene; queries
# with no hit are NO_HIT and count as wrong. Methods: "thresholded" (hmmscan defaults)
# and "no_threshold" (--max, splits 10-40). Writes hmm_runs/pathway_prediction_*.csv.
# =============================================================================

suppressMessages(library(data.table))

# ---- paths (adjust if running elsewhere) ------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
repo_root    <- normalizePath("../..")
manifest_fp  <- file.path(repo_root, "BGF_clustering", "train_test_val_fasta_bypathway", "manifest_all_splits.csv")
tblout_dir   <- file.path(repo_root, "hmm_runs", "scan_results")
besthit_dir  <- file.path(repo_root, "hmm_runs", "scan_results_max_besthit")
query_dir    <- file.path(repo_root, "BGF_clustering", "train_test_val_final")
out_dir      <- file.path(repo_root, "hmm_runs")

splits <- sprintf("%d", seq(10, 90, by = 10))
kinds  <- c("val", "test")

# Splits 10-40 also have hmmscan --max runs, pre-reduced to one best hit per query.
no_threshold_splits <- c("10", "20", "30", "40")

# ---- gene -> canonical pathway lookup ---------------------------------------
build_gene_pathway_lookup <- function(manifest_fp) {
  m <- fread(manifest_fp)
  gp <- unique(m[, .(gene, pathway)])
  dupe <- gp[, .N, by = gene][N > 1]
  if (nrow(dupe) > 0) {
    warning(sprintf("%d gene(s) map to more than one pathway; using the first seen.", nrow(dupe)))
    gp <- gp[!duplicated(gene)]
  }
  gp
}

# ---- tblout parsing ----------------------------------------------------------
# hmmscan --tblout columns (whitespace-delimited, first 18 fixed, 19th is the
# free-text target description and is dropped here since our HMMs have none):
#   target_name target_accession query_name query_accession
#   evalue_full score_full bias_full evalue_best1dom score_best1dom bias_best1dom
#   exp reg clu ov env dom rep inc
tblout_cols <- c(
  "target_name", "target_accession", "query_name", "query_accession",
  "evalue_full", "score_full", "bias_full",
  "evalue_best1dom", "score_best1dom", "bias_best1dom",
  "exp", "reg", "clu", "ov", "env", "dom", "rep", "inc"
)

read_tblout <- function(path) {
  lines <- readLines(path)
  lines <- lines[!startsWith(lines, "#")]
  if (length(lines) == 0) {
    return(data.table(matrix(character(0), ncol = length(tblout_cols),
                              dimnames = list(NULL, tblout_cols))))
  }
  # Vectorized split; per-line strsplit was too slow on the large tblout files.
  n <- length(tblout_cols)
  fields <- tstrsplit(lines, "\\s+", fixed = FALSE)[seq_len(n)]
  dt <- setDT(fields)
  setnames(dt, tblout_cols)
  dt[, `:=`(
    evalue_full = as.numeric(evalue_full),
    score_full  = as.numeric(score_full)
  )]
  dt
}

# Pre-reduced --max results: one row per query already (query_name,
# target_name, evalue_full, score_full), tab-separated, no header -- see
# reduce_max_tblout.sh. No further "best hit per query" reduction needed.
read_besthit_tsv <- function(path) {
  dt <- fread(path, header = FALSE, sep = "\t",
              col.names = c("query_name", "target_name", "evalue_full", "score_full"))
  dt
}

# gene is the last "__"-delimited token of the HMM/target name
# e.g. "bgf_train_10__aceti_met__acdA" -> "acdA"
target_to_gene <- function(target_name) {
  parts <- strsplit(target_name, "__", fixed = TRUE)
  vapply(parts, function(p) p[length(p)], character(1))
}

# query name is the original fasta header "id|gene|cycle"
parse_query_name <- function(query_name) {
  parts <- strsplit(query_name, "|", fixed = TRUE)
  data.table(
    id        = vapply(parts, `[`, character(1), 1),
    true_gene = vapply(parts, `[`, character(1), 2),
    true_pathway = vapply(parts, `[`, character(1), 3)
  )
}

# Parse every query's ground-truth (id, gene, pathway) straight from the
# fasta headers ">id|gene|cycle" -- this is the full query set, including
# ones that never got a single hmmscan hit.
parse_fasta_truth <- function(fasta_path) {
  lines <- readLines(fasta_path)
  headers <- sub("^>", "", lines[startsWith(lines, ">")])
  parts <- strsplit(headers, "|", fixed = TRUE)
  data.table(
    id           = vapply(parts, `[`, character(1), 1),
    true_gene    = vapply(parts, `[`, character(1), 2),
    true_pathway = vapply(parts, `[`, character(1), 3)
  )
}

# Macro precision/recall/F1 and multi-class MCC (Gorodkin). NO_HIT is added as a
# predicted-only class so the confusion matrix is square; it is not scored itself.
compute_class_metrics <- function(full_detail) {
  classes     <- sort(unique(full_detail$true_pathway))
  pred_levels <- c(classes, "NO_HIT")
  K <- length(classes)

  true_f <- factor(full_detail$true_pathway, levels = classes)
  pred_f <- factor(full_detail$predicted_pathway, levels = pred_levels)
  cm <- table(true = true_f, pred = pred_f)  # K x (K+1)

  TP <- diag(cm[, classes, drop = FALSE])
  FP <- colSums(cm[, classes, drop = FALSE]) - TP
  FN <- rowSums(cm) - TP

  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  f1        <- 2 * precision * recall / (precision + recall)
  precision[is.nan(precision)] <- NA
  recall[is.nan(recall)]       <- NA
  f1[is.nan(f1)]                <- NA

  per_class <- data.table(
    pathway = classes, TP = as.integer(TP), FP = as.integer(FP), FN = as.integer(FN),
    precision = as.numeric(precision), recall = as.numeric(recall), f1 = as.numeric(f1)
  )

  # ---- multi-class MCC (Gorodkin 2004), computed on the square confusion
  # matrix: classes x classes, plus NO_HIT as an all-zero true-row/column ----
  cm_square <- matrix(0, nrow = K + 1, ncol = K + 1, dimnames = list(pred_levels, pred_levels))
  cm_square[classes, ] <- cm
  n_all  <- sum(cm_square)
  c_diag <- sum(diag(cm_square))
  t_k <- rowSums(cm_square)
  p_k <- colSums(cm_square)
  mcc_num <- c_diag * n_all - sum(t_k * p_k)
  mcc_den <- sqrt((n_all^2 - sum(p_k^2)) * (n_all^2 - sum(t_k^2)))
  mcc <- if (mcc_den == 0) NA_real_ else mcc_num / mcc_den

  list(
    macro = data.table(
      precision_macro = mean(precision, na.rm = TRUE),
      recall_macro    = mean(recall, na.rm = TRUE),
      f1_macro        = mean(f1, na.rm = TRUE),
      mcc             = mcc,
      n_classes       = K
    ),
    per_class = per_class
  )
}

# ---- score one split x kind (val/test) --------------------------------------
# "thresholded" reads scan_results/*.tblout; "no_threshold" reads
# scan_results_max_besthit/*.tsv (splits 10-40 only).
score_one <- function(split, kind, gene_pathway, method = "thresholded") {
  fasta_fp <- file.path(query_dir, sprintf("bgf_%s_%s.fasta", kind, split))
  all_truth <- parse_fasta_truth(fasta_fp)
  n_total   <- nrow(all_truth)

  if (method == "thresholded") {
    tblout_fp <- file.path(tblout_dir, sprintf("bgf_%s_%s.tblout", kind, split))
    hits <- read_tblout(tblout_fp)
  } else {
    besthit_fp <- file.path(besthit_dir, sprintf("bgf_%s_%s.besthit.tsv", kind, split))
    hits <- read_besthit_tsv(besthit_fp)
  }

  if (nrow(hits) == 0) {
    detail <- copy(all_truth)
    detail[, `:=`(predicted_gene = NA_character_, predicted_pathway = "NO_HIT",
                  evalue_full = NA_real_, score_full = NA_real_, correct = FALSE,
                  split = split, kind = kind, method = method)]
  } else {
    if (method == "thresholded") {
      setorder(hits, query_name, evalue_full, -score_full)
      best <- hits[, .SD[1], by = query_name]  # reduce to best hit per query
    } else {
      best <- hits  # already one row per query (pre-reduced on the cluster)
    }

    best[, predicted_gene := target_to_gene(target_name)]
    best <- merge(best, gene_pathway, by.x = "predicted_gene", by.y = "gene", all.x = TRUE)
    setnames(best, "pathway", "predicted_pathway")

    truth <- parse_query_name(best$query_name)
    best <- cbind(best, truth)
    best[, correct := predicted_pathway == true_pathway]

    scored <- best[, .(id, true_gene, true_pathway, predicted_gene, predicted_pathway,
                        evalue_full, score_full, correct)]

    unscored_ids <- setdiff(all_truth$id, scored$id)
    unscored <- all_truth[id %in% unscored_ids]
    unscored[, `:=`(predicted_gene = NA_character_, predicted_pathway = "NO_HIT",
                     evalue_full = NA_real_, score_full = NA_real_, correct = FALSE)]

    detail <- rbind(scored, unscored, use.names = TRUE)
    detail[, `:=`(split = split, kind = kind, method = method)]
  }

  n_scored  <- sum(detail$predicted_pathway != "NO_HIT")
  n_correct <- sum(detail$correct, na.rm = TRUE)

  metrics <- compute_class_metrics(detail)
  metrics$per_class[, `:=`(split = split, kind = kind, method = method)]

  list(
    summary = cbind(
      data.table(split = split, kind = kind, method = method, n_total = n_total,
                 n_scored = n_scored, n_correct = n_correct,
                 accuracy = n_correct / n_total),
      metrics$macro
    ),
    detail = detail,
    per_class = metrics$per_class
  )
}

# ---- run across all splits x kinds ------------------------------------------
gene_pathway <- build_gene_pathway_lookup(manifest_fp)

all_summaries  <- list()
all_details    <- list()
all_per_class  <- list()
i <- 0L
for (split in splits) {
  for (kind in kinds) {
    i <- i + 1L
    message(sprintf("scoring bgf_%s_%s (thresholded) ...", kind, split))
    res <- score_one(split, kind, gene_pathway, method = "thresholded")
    all_summaries[[i]] <- res$summary
    all_details[[i]]   <- res$detail
    all_per_class[[i]] <- res$per_class

    if (split %in% no_threshold_splits) {
      i <- i + 1L
      message(sprintf("scoring bgf_%s_%s (no_threshold) ...", kind, split))
      res <- score_one(split, kind, gene_pathway, method = "no_threshold")
      all_summaries[[i]] <- res$summary
      all_details[[i]]   <- res$detail
      all_per_class[[i]] <- res$per_class
    }
  }
}

summary_dt   <- rbindlist(all_summaries)
detail_dt    <- rbindlist(all_details, use.names = TRUE, fill = TRUE)
per_class_dt <- rbindlist(all_per_class, use.names = TRUE, fill = TRUE)

summary_dt <- summary_dt[order(kind, as.integer(split), method)]
fwrite(summary_dt, file.path(out_dir, "pathway_prediction_summary.csv"))
fwrite(detail_dt, file.path(out_dir, "pathway_prediction_detail.csv"))
fwrite(per_class_dt, file.path(out_dir, "pathway_prediction_per_class.csv"))

print(summary_dt)
message(sprintf("\nWrote summary to %s", file.path(out_dir, "pathway_prediction_summary.csv")))
message(sprintf("Wrote per-class precision/recall/F1 to %s", file.path(out_dir, "pathway_prediction_per_class.csv")))
message(sprintf("Wrote per-query detail to %s", file.path(out_dir, "pathway_prediction_detail.csv")))
