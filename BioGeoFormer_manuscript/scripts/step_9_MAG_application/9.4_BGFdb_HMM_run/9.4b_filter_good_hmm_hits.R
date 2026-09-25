#!/usr/bin/env Rscript
# Keeps the best hmmscan hit per MAG ORF with full-sequence E-value < 1e-5 and
# bit score > 50, and maps its gene to a pathway. Run with the tblout from
# 9.4a (coverage columns stay NA; they need the domtblout).

suppressMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)

file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg))) else
  dirname(rstudioapi::getActiveDocumentContext()$path)
repo_root <- normalizePath(file.path(script_dir, "..", "..", ".."))
mag_dir <- file.path(repo_root, "cold_seep_MAG_application", "HMM_BGF_output")
hit_fp    <- if (length(args) >= 1) args[1] else
  file.path(mag_dir, "MAGs_vs_bgf_train_50.tblout")
manifest_fp <- if (length(args) >= 2) args[2] else
  file.path(repo_root, "BGF_clustering", "train_test_val_fasta_bypathway", "manifest_all_splits.csv")
out_fp <- if (length(args) >= 3) args[3] else
  file.path(mag_dir, "MAGs_vs_bgf_train_50_good_hits.csv")
evalue_cutoff          <- if (length(args) >= 4) as.numeric(args[4]) else 1e-5
score_cutoff           <- if (length(args) >= 5) as.numeric(args[5]) else 50
hmm_coverage_cutoff    <- if (length(args) >= 6) as.numeric(args[6]) else 0
query_coverage_cutoff  <- if (length(args) >= 7) as.numeric(args[7]) else 0

is_domtblout <- grepl("\\.domtblout$", hit_fp)

# ---- gene -> canonical pathway lookup (same logic as evaluate_hmm_predictions.R) ----
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

# ---- tblout parsing -------------------------------------------------------
# hmmscan --tblout columns (whitespace-delimited, first 18 fixed, 19th is
# free-text and dropped):
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
  n <- length(tblout_cols)
  fields <- tstrsplit(lines, "\\s+", fixed = FALSE)[seq_len(n)]
  dt <- setDT(fields)
  setnames(dt, tblout_cols)
  dt[, `:=`(evalue_full = as.numeric(evalue_full), score_full = as.numeric(score_full))]
  dt
}

# ---- domtblout parsing -----------------------------------------------------
# hmmscan --domtblout columns (whitespace-delimited, first 22 fixed, 23rd+
# is free-text and dropped):
#   target_name target_accession tlen query_name query_accession qlen
#   evalue_full score_full bias_full
#   dom_num dom_of evalue_c evalue_i score_dom bias_dom
#   hmm_from hmm_to ali_from ali_to env_from env_to acc
domtblout_cols <- c(
  "target_name", "target_accession", "tlen", "query_name", "query_accession", "qlen",
  "evalue_full", "score_full", "bias_full",
  "dom_num", "dom_of", "evalue_c", "evalue_i", "score_dom", "bias_dom",
  "hmm_from", "hmm_to", "ali_from", "ali_to", "env_from", "env_to", "acc"
)

read_domtblout <- function(path) {
  lines <- readLines(path)
  lines <- lines[!startsWith(lines, "#")]
  n <- length(domtblout_cols)
  fields <- tstrsplit(lines, "\\s+", fixed = FALSE)[seq_len(n)]
  dt <- setDT(fields)
  setnames(dt, domtblout_cols)
  num_cols <- c("tlen", "qlen", "evalue_full", "score_full",
                "hmm_from", "hmm_to", "ali_from", "ali_to")
  dt[, (num_cols) := lapply(.SD, as.numeric), .SDcols = num_cols]
  dt
}

# gene is the last "__"-delimited token of the HMM/target name
# e.g. "bgf_train_50__aceti_met__acdA" -> "acdA"
target_to_gene <- function(target_name) {
  parts <- strsplit(target_name, "__", fixed = TRUE)
  vapply(parts, function(p) p[length(p)], character(1))
}

# ---- run ----------------------------------------------------------------
gene_pathway <- build_gene_pathway_lookup(manifest_fp)

message(sprintf("reading %s ...", hit_fp))

if (is_domtblout) {
  dom <- read_domtblout(hit_fp)
  message(sprintf("%d raw domain rows", nrow(dom)))

  # evalue_full/score_full are constant across all domain rows of the same
  # (query, target) pair, so this filters at the hit level same as tblout.
  good <- dom[evalue_full < evalue_cutoff & score_full > score_cutoff]
  message(sprintf(
    "%d domain rows pass evalue_full < %.0e & score_full > %g",
    nrow(good), evalue_cutoff, score_cutoff
  ))

  # Collapse to one row per (query, target) hit, unioning domain regions to
  # get overall profile/query span for coverage.
  hit <- good[, .(
    evalue_full = evalue_full[1], score_full = score_full[1],
    tlen = tlen[1], qlen = qlen[1],
    hmm_from = min(hmm_from), hmm_to = max(hmm_to),
    ali_from = min(ali_from), ali_to = max(ali_to)
  ), by = .(query_name, target_name)]

  hit[, `:=`(
    hmm_coverage   = (hmm_to - hmm_from + 1) / tlen,
    query_coverage = (ali_to - ali_from + 1) / qlen
  )]

  hit <- hit[hmm_coverage >= hmm_coverage_cutoff & query_coverage >= query_coverage_cutoff]
  message(sprintf(
    "%d hits pass hmm_coverage >= %g & query_coverage >= %g (%d unique queries)",
    nrow(hit), hmm_coverage_cutoff, query_coverage_cutoff, uniqueN(hit$query_name)
  ))
} else {
  hits <- read_tblout(hit_fp)
  message(sprintf("%d raw hit rows", nrow(hits)))

  hit <- hits[evalue_full < evalue_cutoff & score_full > score_cutoff]
  message(sprintf(
    "%d hits pass evalue_full < %.0e & score_full > %g (%d unique queries) -- tblout input, coverage not available",
    nrow(hit), evalue_cutoff, score_cutoff, uniqueN(hit$query_name)
  ))
  hit[, `:=`(hmm_coverage = NA_real_, query_coverage = NA_real_)]
}

hit[, predicted_gene := target_to_gene(target_name)]
hit <- merge(hit, gene_pathway, by.x = "predicted_gene", by.y = "gene", all.x = TRUE)
setnames(hit, "pathway", "predicted_pathway")

setorder(hit, query_name, evalue_full, -score_full)
best <- hit[, .SD[1], by = query_name]  # single best-scoring qualifying hit per query

out <- best[, .(query_name, predicted_gene, predicted_pathway,
                evalue_full, score_full, hmm_coverage, query_coverage)]

dir.create(dirname(out_fp), recursive = TRUE, showWarnings = FALSE)
fwrite(out, out_fp)
message(sprintf(
  "\nWrote %d best-hit rows (one per query with at least one good hit) to %s",
  nrow(out), out_fp
))
