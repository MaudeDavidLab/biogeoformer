#!/usr/bin/env Rscript
# =============================================================================
# Defines split_one_file(): writes one training split as per-gene FASTA/CSV files grouped
# by pathway. Sourced by run_all_splits.R (needs utils_fasta.R).
# =============================================================================

suppressMessages(library(data.table))

# split_one_file(csv_path, out_root): returns and writes the split's manifest.csv.
split_one_file <- function(csv_path, out_root) {
  split_name <- tools::file_path_sans_ext(basename(csv_path))
  message(sprintf("[%s] reading %s", split_name, csv_path))

  dt <- fread(
    csv_path,
    colClasses = list(character = c("id", "gene", "cycle", "sequence"))
  )

  split_dir <- file.path(out_root, split_name)
  dir.create(split_dir, recursive = TRUE, showWarnings = FALSE)

  groups <- split(dt, by = c("cycle", "gene"), sorted = TRUE, keep.by = TRUE)

  manifest <- vector("list", length(groups))

  for (i in seq_along(groups)) {
    sub <- groups[[i]]
    pathway <- sub$cycle[1]
    gene <- sub$gene[1]

    pathway_dir <- file.path(split_dir, sanitize_filename(pathway))
    dir.create(pathway_dir, recursive = TRUE, showWarnings = FALSE)

    gene_safe <- sanitize_filename(gene)
    fasta_path <- file.path(pathway_dir, paste0(gene_safe, ".fasta"))
    csv_path_out <- file.path(pathway_dir, paste0(gene_safe, ".csv"))

    write_gene_fasta(sub, fasta_path)
    fwrite(sub, csv_path_out)

    manifest[[i]] <- data.table(
      split = split_name,
      pathway = pathway,
      gene = gene,
      n_sequences = nrow(sub),
      fasta_path = file.path(split_name, sanitize_filename(pathway), paste0(gene_safe, ".fasta")),
      csv_path = file.path(split_name, sanitize_filename(pathway), paste0(gene_safe, ".csv"))
    )
  }

  manifest <- rbindlist(manifest)
  setorder(manifest, pathway, gene)
  fwrite(manifest, file.path(split_dir, "manifest.csv"))

  message(sprintf(
    "[%s] wrote %d gene files across %d pathways (%d sequences total)",
    split_name, nrow(manifest), uniqueN(manifest$pathway), sum(manifest$n_sequences)
  ))

  invisible(manifest)
}
