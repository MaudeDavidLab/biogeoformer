library(data.table)
library(ggplot2)
library(scales)

# S Fig 11: identity of each MAG ORF's top DIAMOND hit against BGFdb, broken y-axis.
# The y-axis is linear from 0 to lo_max (covers every identity bin), then skips the
# range lo_max..up_min and resumes linear around the no-hit count. The two linear
# sections have different scales (see ty()); the gap is marked with break symbols.
# The no-hit ORFs (no DIAMOND alignment at all; the .m8 only lists ORFs that got one)
# are drawn as one bin to the left of the identity histogram. No filters are applied.

if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
} else {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) setwd(dirname(sub("^--file=", "", file_arg)))
}

m8      <- "../../cold_seep_MAG_application/DIAMOND_BGF_output/mags_diamond_output_sep10.m8"
bgf_csv <- "../../cold_seep_MAG_application/BGF_annotations/cleaned_MAGS_forbgf_output_50.csv"
out_dir <- "../../results/figures/mags_diamond_pid_distribution"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# columns 1 = query id, 3 = percent_identity
aln <- fread(m8, header = FALSE, select = c(1, 3), col.names = c("seq_id", "percent_identity"))

# every ORF that went into BGF / DIAMOND (only the id column is read)
all_ids <- fread(bgf_csv, select = "seq_id")$seq_id
stopifnot(!anyDuplicated(all_ids), all(aln$seq_id %in% all_ids))
n_orfs   <- length(all_ids)
n_hits   <- nrow(aln)
n_no_hit <- n_orfs - n_hits
min_pid  <- min(aln$percent_identity)

# 1-point bins, right-closed (same binning as geom_histogram(binwidth = 1, boundary = 0))
bins <- aln[, .(n = .N), by = .(bin_end = ceiling(percent_identity))][order(bin_end)]
bins[, `:=`(xmin = bin_end - 1, xmax = bin_end)]

# ---- broken y-axis -------------------------------------------------------------
# Display units 0-100: lower section [0, lo_max] -> [0, lo_units], gap, upper section
# [up_min, up_max] -> [lo_units + gap_units, 100]. Counts inside the gap are never drawn.
lo_max <- 55e3;  up_min <- 4.6e6;  up_max <- 5.2e6
lo_units <- 80;  gap_units <- 4
ty <- function(y) {
  stopifnot(all(y <= lo_max | y >= up_min), all(y <= up_max))
  ifelse(y <= lo_max,
         y / lo_max * lo_units,
         lo_units + gap_units + (y - up_min) / (up_max - up_min) * (100 - lo_units - gap_units))
}
y_counts <- c(0, 1e4, 2e4, 3e4, 4e4, 5e4, 5e6)
y_labels <- c("0", "10K", "20K", "30K", "40K", "50K", "5M")

# no-hit bin drawn against the left end of the histogram. Its x-position and width are
# arbitrary (it is a category, not an identity range); it is nudged in over the 17-19%
# bins, which hold only 50 ORFs in total and are invisible at this y-scale.
nohit_xmin <- 16
nohit_xmax <- 19
nohit_x    <- (nohit_xmin + nohit_xmax) / 2
x_left     <- nohit_xmin - 1.5

col_hist <- "#2a78d6"   # blue for both the histogram and the no-hit bin

x_breaks <- c(nohit_x, seq(20, 100, 10))
x_labels <- c("no hit", seq(20, 100, 10))

# break marks: one set of two parallel diagonal cuts across the no-hit bar, with the gap
# between them cut out in white (parallelogram slightly wider than the bar)
gap_lo <- lo_units
gap_hi <- lo_units + gap_units
bx0 <- nohit_xmin - 0.3   # left/right ends of the cuts
bx1 <- nohit_xmax + 0.3
dy  <- 1.3                # half of the rise of each cut
cut_line <- function(y) {
  annotate("segment", x = bx0, xend = bx1, y = y - dy, yend = y + dy,
           linewidth = 0.5, colour = "grey25")
}

p <- ggplot() +
  geom_rect(data = bins, aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = ty(n)), fill = col_hist) +
  annotate("rect", xmin = nohit_xmin, xmax = nohit_xmax, ymin = 0, ymax = ty(n_no_hit), fill = col_hist) +
  annotate("polygon", x = c(bx0, bx1, bx1, bx0),
           y = c(gap_lo - dy, gap_lo + dy, gap_hi + dy, gap_hi - dy), fill = "white") +
  cut_line(gap_lo) + cut_line(gap_hi) +
  annotate("text", x = nohit_xmax + 0.6, y = ty(n_no_hit) - 6, hjust = 0, vjust = 0.5, size = 3.2,
           lineheight = 0.95,
           label = sprintf("%s ORFs with no hit\n(no alignment; < %.1f%% identity)",
                           format(n_no_hit, big.mark = ","), min_pid)) +
  annotate("text", x = 55, y = 60, hjust = 0, vjust = 0.5, size = 3.2, lineheight = 0.95,
           label = sprintf("%s ORFs with a hit\n(median %.1f%% identity)",
                           format(n_hits, big.mark = ","), median(aln$percent_identity))) +
  scale_x_continuous(breaks = x_breaks, labels = x_labels) +
  scale_y_continuous(breaks = ty(y_counts), labels = y_labels) +
  # limits set on the coord (not the scales) so the axis-edge break marks are not dropped
  coord_cartesian(xlim = c(x_left, 100.9), ylim = c(0, 100), expand = FALSE, clip = "off") +
  labs(x = "DIAMOND top-hit percent identity (%)",
       y = "Number of ORFs",
       title = "Percent identity of DIAMOND hits, cold seep MAGs vs BioGeoFormer database",
       subtitle = sprintf("%s ORFs in total; all alignments, one top hit per ORF, 1 percentage-point bins. Y-axis is broken between 50K and 5M.",
                          format(n_orfs, big.mark = ","))) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        plot.title = element_text(face = "bold", size = 13),
        plot.title.position = "plot")

ggsave(file.path(out_dir, "mags_diamond_pid_distribution_broken_axis.png"), p,
       width = 10, height = 5, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "mags_diamond_pid_distribution_broken_axis.pdf"), p,
       width = 10, height = 5)
