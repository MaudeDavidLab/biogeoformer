#!/usr/bin/env Rscript
# =============================================================================
# FASTA helpers for the gene/pathway splitting scripts (headers >id|gene|cycle).
# =============================================================================

# Make a gene or pathway label safe to use as a file name.
sanitize_filename <- function(x) {
  gsub("[^A-Za-z0-9_.-]", "_", x)
}

# Write one gene's rows (id, gene, cycle, sequence) to a FASTA file.
write_gene_fasta <- function(dt, path) {
  lines <- character(nrow(dt) * 2L)
  lines[c(TRUE, FALSE)] <- paste0(">", dt$id, "|", dt$gene, "|", dt$cycle)
  lines[c(FALSE, TRUE)] <- dt$sequence
  writeLines(lines, path)
}
