library(dplyr)
library(stringr)
library(tidyverse)
library(scales)


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()



short_pathways <- c(
  #methane cycle
  "cenmetpat",
  "cenmetpat,aom",
  "aceti_met",
  "methyl_met",
  "oxid_met_c1",
  "oxid_formaldehyde",
  "oxid_formate",
  "serine",
  "rump",
  
  
  # nitrogen cycles
  "nitrification", 
  "denitrification", 
  "assnitred", 
  "dissnitred", 
  "nitfix", 
  "annamox", 
  "odegsyn", 
  "nitrogen_other",
  
  
  # phosphorus cycles 
  "pyruvate", 
  "pentose", 
  "phosphotransferase", 
  "ox_phosphorylation", 
  "phosph_met", 
  "two_comp", 
  "transporters", 
  "org_phos_hyd",
  "phos_other", 
  "purine", 
  "pyrimidine",
  
  #sulfur cycles 
  "asssulred",
  "dsro",
  "sulred",
  "SOX", 
  "sulfur_ox", 
  "sulfur_dis", 
  "org_sul_trans", 
  "in_or_sul",  
  "sul_other")

long_pathways <- c("Central methanogenic pathway",
                   "Central methanogenic pathway, AOM",
                   "Aceticlastic methanogenesis",
                   "Methylotrophic methanogenesis",
                   "Oxidation of methane and C1 compounds",
                   "Oxidation of formaldehyde", 
                   "Oxidation of formate", 
                   "Serine cycle", 
                   "RuMP cycle", 
                   
                   
                   "Nitrification",
                   "Denitrication",
                   "Assimilatory nitrate reduction",
                   "Dissimilatory nitrate reduction",
                   "Nitrogen fixation",
                   "Annamox",
                   "Organic degradation and synthesis",
                   "Related Nitrogen genes",
                   
                   
                   "Pyruvate metabolism",
                   "Pentose phosphate pathway",
                   "Phosphotransferase system",
                   "Oxidative phosphorylation",
                   "Phosphonate and phosphinate metabolism",
                   "Two-component system",
                   "Transporters",
                   "Organic phosphoester hydrolysis",
                   "Related phosphorus genes",
                   "Purine metabolism",
                   "Pyrimidine metabolism",
                   
                   
                   "Assimilatory sulphate reduction",
                   "Dissimilatory sulphur reduction and oxidation",
                   "Sulphur reduction",
                   "SOX systems",
                   "Sulphur oxidation",
                   "Sulphur disproportionation",
                   "Organic sulphur transformation",
                   "Linkages between inorganic and organic sulphur transformation",
                   "Related sulphur genes"
                   
)


short_pathways <- as.data.frame(short_pathways)
long_pathways <- as.data.frame(long_pathways)

pathwaymap <- cbind(short_pathways, long_pathways)


joined_predictions <- read.csv("../../cold_seep_MAG_application/joined_predictions_coldseep_mags.csv")

long_preds <- joined_predictions %>%
  pivot_longer(
    cols = starts_with("prediction_"),
    names_to = "method",
    names_prefix = "prediction_",
    values_to = "pathway"
  ) %>%
  filter(!is.na(pathway))  # Remove NAs

# Step 3: Map short names to long pathway names
# Assume you have a dataframe `pathwaymap` with short to long names
long_preds <- left_join(long_preds, pathwaymap, by = c("pathway" = "short_pathways")) %>%
  mutate(pathway = long_pathways) %>%
  select(-long_pathways)

# Step 4: Count occurrences
pathway_counts <- long_preds %>%
  group_by(pathway, method) %>%
  summarise(count = n(), .groups = "drop")

# Step 5: Rename methods for display
pathway_counts <- pathway_counts %>%
  mutate(method = recode(method,
                         "diamond" = "Alignment",
                         "bgf" = "BGF",
                         "hmm" = "HMM",
                         "kegg" = "KEGG"))

# Step 6: Assign cycles to each pathway
long_pathways <- c(
  ## methane
  "Central methanogenic pathway" = "Methane",
  "Central methanogenic pathway, AOM" = "Methane",
  "Aceticlastic methanogenesis" = "Methane",
  "Methylotrophic methanogenesis" = "Methane",
  "Oxidation of methane and C1 compounds" = "Methane",
  "Oxidation of formaldehyde" = "Methane",
  "Oxidation of formate" = "Methane",
  "Serine cycle" = "Methane",
  "RuMP cycle" = "Methane",
  ## nitrogen
  "Nitrification" = "Nitrogen",
  "Denitrication" = "Nitrogen",
  "Assimilatory nitrate reduction" = "Nitrogen",
  "Dissimilatory nitrate reduction" = "Nitrogen",
  "Nitrogen fixation" = "Nitrogen",
  "Annamox" = "Nitrogen",
  "Organic degradation and synthesis" = "Nitrogen",
  "Related Nitrogen genes" = "Nitrogen",
  ## phosphorus
  "Pyruvate metabolism" = "Phosphorus",
  "Pentose phosphate pathway" = "Phosphorus",
  "Phosphotransferase system" = "Phosphorus",
  "Oxidative phosphorylation" = "Phosphorus",
  "Phosphonate and phosphinate metabolism" = "Phosphorus",
  "Two-component system" = "Phosphorus",
  "Transporters" = "Phosphorus",
  "Organic phosphoester hydrolysis" = "Phosphorus",
  "Related phosphorus genes" = "Phosphorus",
  "Purine metabolism" = "Phosphorus",
  "Pyrimidine metabolism" = "Phosphorus",
  ## sulphur
  "Assimilatory sulphate reduction" = "Sulphur",
  "Dissimilatory sulphur reduction and oxidation" = "Sulphur",
  "Sulphur reduction" = "Sulphur",
  "SOX systems" = "Sulphur",
  "Sulphur oxidation" = "Sulphur",
  "Sulphur disproportionation" = "Sulphur",
  "Organic sulphur transformation" = "Sulphur",
  "Linkages between inorganic and organic sulphur transformation" = "Sulphur",
  "Related sulphur genes" = "Sulphur"
)

# Canonical, ordered pathway list -- used to force the bubble and bar
# panels below onto the exact same set of y-axis rows in the exact same
# order. Without this, each ggplot panel independently derives its axis
# categories from whatever pathways happen to be present in its own data,
# and since bar_plot's data (BGF-unique calls only) is a much narrower
# subset than bubble_plot's (all methods), any pathway missing from one but
# not the other shifts every row below it out of alignment between panels.
canonical_pathways <- names(long_pathways)

pathway_counts <- pathway_counts %>%
  mutate(cycle = recode(pathway, !!!long_pathways))

pathway_counts <- filter(pathway_counts, !is.na(cycle))

# Step 7: Plot
bubble_plot <- ggplot(pathway_counts, aes(x = method, y = pathway, size = count, color = cycle)) + 
  geom_point(alpha = 0.7) +
  xlab("Prediction Method") +
  ylab("Metabolic Pathway") +
  theme_minimal() +
  theme(axis.text = element_text(size = 12.5),
        axis.title = element_text(size = 14),
        plot.title = element_text(size = 16, hjust = 0.5),
        legend.position = "right") +
  scale_size_continuous(name = "Gene Count", range = c(3, 15))


ggsave("../../results/figures/misc/bubble_plot.png", bubble_plot, width = 10.5, height = 9, dpi = 600)


comparison <- pathway_counts %>%
  pivot_wider(names_from = method, values_from = count) %>%
  mutate(
    vs_Alignment = BGF/Alignment ,
    vs_HMM       = BGF/ HMM ,
    vs_KEGG      = BGF/KEGG
  ) %>%
  select(cycle, pathway, starts_with("vs_"))

write.csv(comparison, "../../results/tables/summary_stats_comparison.csv")

print(comparison)


median_comparison <- comparison %>%
  summarise(
    median_vs_Alignment = median(vs_Alignment, na.rm = TRUE),
    median_vs_HMM       = median(vs_HMM, na.rm = TRUE),
    median_vs_KEGG      = median(vs_KEGG, na.rm = TRUE)
  )

print(median_comparison)


mean_comparison <- comparison %>%
  summarise(
    mean_vs_Alignment = mean(vs_Alignment, na.rm = TRUE),
    mean_vs_HMM       = mean(vs_HMM, na.rm = TRUE),
    mean_vs_KEGG      = mean(vs_KEGG, na.rm = TRUE)
  )

print(mean_comparison)






selected_pathways <- c(
  "Two-component system",
  "Transporters",
  "Sulphur reduction",
  "Nitrification",
  "Denitrication",
  "Central methanogenic pathway",
  "Aceticlastic methanogenesis"
)

# Filter the dataset
filtered_pathway_data <- pathway_counts %>%
  filter(pathway %in% selected_pathways)


bubble_plot_filtered <- ggplot(filtered_pathway_data, aes(x = method, y = pathway, size = count, color = cycle)) + 
  geom_point(alpha = 0.7) +
  xlab("Prediction Method") +
  ylab("Metabolic Pathway") +
  theme_minimal() +
  theme(axis.text = element_text(size = 12.5),
        axis.title = element_text(size = 14),
        plot.title = element_text(size = 16, hjust = 0.5),
        legend.position = "right") +
  scale_size_continuous(name = "Gene Count", range = c(3, 15))




df <- read.csv("../../cold_seep_MAG_application/model_predictions_mags_allmodels_df.csv")


library(dplyr)
library(ggplot2)

# 1. Subset to only BGF predictions
bgf_only <- df %>%
  filter(!is.na(prediction_bgf) &                       # bgf made a call
           is.na(prediction_diamond) &
           is.na(prediction_hmm) &
           is.na(prediction_kegg)) %>%
  select(query_id, prediction_bgf)

# 2. Count number of cycles predicted
cycle_counts <- bgf_only %>%
  count(prediction_bgf, name = "n")

cycle_counts_long <- cycle_counts %>%
  left_join(pathwaymap, by = c("prediction_bgf" = "short_pathways")) %>%
  mutate(pathway = ifelse(is.na(long_pathways), prediction_bgf, long_pathways)) %>%
  select(pathway, n)

cycle_counts_long

# Fill in every canonical pathway BGF never uniquely called (n = 0), rather
# than the old one-off patch for a single pathway -- guarantees this panel
# always has the same row set as bubble_plot's, regardless of which
# pathways happen to be BGF-unique for whatever model/threshold is in use.
cycle_counts_long <- cycle_counts_long %>%
  complete(pathway = canonical_pathways, fill = list(n = 0))

cycle_counts_long <- cycle_counts_long %>%
  mutate(cycle = recode(pathway, !!!long_pathways))



library(ggplot2)
library(patchwork)

# Custom scientific formatter -- renders as "1 %*% 10^4" plotmath (proper
# "1 x 10^4" superscript) instead of R's plain "1e+04" text. Moved up here
# (originally defined further down, near barplot_total) so bar_plot below
# can use it too.
sci_labeller <- function(x) {
  parse(text = gsub("e\\+?", " %*% 10^", scientific_format()(x)))
}

pathway_counts <- pathway_counts %>%
  mutate(method = recode(method,
                         "Alignment"   = "DIAMOND-BGFdb",
                         "HMM"       = "HMM",
                         "KEGG"      = "DIAMOND-KEGG"
                         # "BGF" already reads as "BGF" from the earlier recode -- no-op dropped
  ))


pathway_counts$method <- factor(
  pathway_counts$method,
  levels = c("DIAMOND-BGFdb", "BGF", "HMM", "DIAMOND-KEGG")
)

# Same canonical pathway levels as cycle_counts_long below, so the two
# panels' y-axes line up row-for-row.
pathway_counts$pathway <- factor(pathway_counts$pathway, levels = canonical_pathways)
cycle_counts_long$pathway <- factor(cycle_counts_long$pathway, levels = canonical_pathways)



bubble_plot <- ggplot(pathway_counts, aes(x = method, y = pathway, size = count, color = cycle)) +
  geom_point(alpha = 0.7) +
  scale_y_discrete(drop = FALSE) +   # keep every canonical pathway row even if empty here
  xlab(NULL) +
  ylab(NULL) +   # removed y-axis label
  theme_minimal() +
  theme(
    axis.text.y.left = element_text(size = 12.5),
    axis.text.x.bottom = element_text(
      size = 12.5, 
      angle = 45, 
      vjust = 1,             # vertical adjust (1 = down/right for angled text)
      hjust = 1,             # horizontal adjust (1 = right align)
      margin = margin(t = 8) # 👈 add top margin to push text down
    ), 
    axis.title.x = element_text(size = 14),
    plot.title = element_text(size = 16, hjust = 0.5),
    
    # Legend tweaks
    legend.position = "left",
    legend.justification = "top",
    legend.box.just = "left",
    legend.margin = margin(0, 0, 0, 0),         # shrink inside spacing
    legend.box.margin = margin(r = -15, l = -5), # pull closer to plot
    
    # Bigger cycle legend
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 16),
    legend.key.size = unit(1.2, "cm"),
    legend.spacing.y = unit(0.5, "cm"),
    
    panel.grid.major.y = element_line(color = "grey80"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  scale_size_continuous(name = "Gene Count", range = c(3, 15))

# Bar plot (to the right) -- set bar_x_log_scale to FALSE for a plain linear
# x-axis instead; both versions share the same angled-label styling.
bar_x_log_scale <- FALSE

bar_x_scale <- if (bar_x_log_scale) {
  # pseudo_log instead of log10 -- n includes real zeros (from complete()
  # above), and log10(0) is -Inf, which would silently drop those bars with
  # a warning; pseudo_log handles zero gracefully while still reading as
  # log-scaled for the larger counts. Explicit breaks/labels because
  # pseudo_log_trans()'s automatic breaks overlap and render garbled at the
  # high end otherwise.
  scale_x_continuous(
    trans = scales::pseudo_log_trans(base = 10),
    breaks = c(0, 100, 10000, 300000),
    labels = sci_labeller
  )
} else {
  # Explicit breaks here too: coord_cartesian(xlim = c(0, 55000)) below
  # re-derives breaks over the zoomed range, and 55k doesn't divide nicely,
  # so the automatic algorithm drops to 20k steps.
  scale_x_continuous(
    breaks = seq(0, 60000, by = 10000),
    labels = sci_labeller
  )
}

bar_plot <- ggplot(cycle_counts_long, aes(x = n, y = pathway, fill = cycle)) +
  geom_col(color = NA) +   # no bold black border
  scale_y_discrete(drop = FALSE) +   # same canonical row set as bubble_plot
  bar_x_scale +
  # Truncate the view at 50k rather than a proper axis break -- Purine's bar
  # (168,059) just gets clipped at the panel edge instead of drawn full
  # length; the underlying data/geom is untouched, only the visible range is
  # cut, so it's easy to annotate/finish by hand afterward. Linear axis only
  # (xlim here is in transformed space under pseudo_log, so skip it there).
  { if (!bar_x_log_scale) coord_cartesian(xlim = c(0, 65000)) } +
  labs(x = "Unique BGF Predictions", y = NULL) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.y = element_blank(),    # hide y text
    axis.ticks.y = element_blank(),   # hide ticks
    axis.title.y = element_blank(),   # no y title
    axis.text.x = element_text(size = 10, angle = 45, hjust = 1),  # fewer + angled to fit this narrow panel
    panel.grid.major.y = element_line(color = "grey80"),  # same horizontal grid
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "none"
  )


combined <- bubble_plot + bar_plot + plot_layout(widths = c(3, 1.6))
combined



ggsave("../../results/figures/misc/bubble_plot_andbar.png", combined, width = 12, height = 9, dpi = 600)
ggsave("../../results/figures/misc/bubble_plot_andbar.svg", combined, width = 12, height = 9, dpi = 600)




ggsave("../../results/figures/misc/bubble_plot_filtered.png", bubble_plot_filtered, width = 7, height = 4, dpi = 600)


bubble_plot_filtered <- ggplot(filtered_pathway_data, aes(x = method, y = pathway, size = count, color = cycle)) + 
  geom_point(alpha = 0.7) +
  xlab("Prediction Method") +
  ylab("Metabolic Pathway") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 12.5),
    axis.text.x = element_text(angle = 45, hjust = 1),  # angled x-axis
    axis.title = element_text(size = 14),
    plot.title = element_text(size = 16, hjust = 0.5),
    legend.position = "right"
  ) +
  scale_size_continuous(name = "Gene Count", range = c(3, 15))

ggsave("../../results/figures/misc/bubble_plot_filtered.png", bubble_plot_filtered, width = 7, height = 4, dpi = 600)




# Count non-NA predictions per method from the same `joined_predictions` dataframe
annotation_counts <- joined_predictions %>%
  pivot_longer(
    cols = starts_with("prediction_"),
    names_to = "method",
    names_prefix = "prediction_",
    values_to = "pathway"
  ) %>%
  filter(!is.na(pathway)) %>%
  count(method) %>%
  mutate(method = recode(method,
                         "diamond" = "Diamond",
                         "bgf" = "BGF",
                         "hmm" = "HMM",
                         "kegg" = "KEGG"))

# Plot barplot of total annotations per method
library(scales)

# Reorder method factor
annotation_counts$method <- factor(annotation_counts$method, levels = c("KEGG", "BGF", "Diamond", "HMM"))

# Plot
barplot_total <- ggplot(annotation_counts, aes(x = method, y = n)) +
  geom_col(fill = "#333333", alpha = 0.9) +
  coord_flip() +
  scale_y_continuous(labels = sci_labeller) +  # <- fixed scientific format
  labs(x = "", y = "Total Gene Annotations", title = " ") +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    plot.title = element_text(size = 16, hjust = 0.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "none"
  )

# Save at high resolution
ggsave("../../results/figures/misc/barplot.png", barplot_total, width = 9, height = 5, dpi = 600)


ggsave(
  "../../results/figures/misc/barplot.svg",
  barplot_total,
  width = 9,
  height = 5
)



# ---------------------------------------------------------------------------
# Barplot: pathways where HMM and BGF independently agree on the same call
# (both methods made a prediction for the protein, and it's the same
# pathway). Uses the same canonical_pathways factor + complete() pattern as
# cycle_counts_long above so every pathway gets a row (n = 0 if HMM and BGF
# never agreed on it), rather than only showing pathways with at least one
# agreement.
# ---------------------------------------------------------------------------
hmm_bgf_agree_counts <- df %>%
  filter(!is.na(prediction_hmm) & !is.na(prediction_bgf) & prediction_hmm == prediction_bgf) %>%
  count(prediction_hmm, name = "n") %>%
  left_join(pathwaymap, by = c("prediction_hmm" = "short_pathways")) %>%
  mutate(pathway = ifelse(is.na(long_pathways), prediction_hmm, long_pathways)) %>%
  select(pathway, n) %>%
  complete(pathway = canonical_pathways, fill = list(n = 0)) %>%
  mutate(
    cycle = recode(pathway, !!!long_pathways),
    pathway = factor(pathway, levels = canonical_pathways)
  )

hmm_bgf_agree_plot <- ggplot(hmm_bgf_agree_counts, aes(x = pathway, y = n, fill = cycle)) +
  geom_col() +
  coord_flip() +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(labels = sci_labeller) +
  labs(x = NULL, y = "HMM & BGF Agreement Count",
       title = "Pathways Where HMM and BGF Independently Agree") +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.y = element_text(size = 10),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right"
  )

ggsave("../../results/figures/misc/hmm_bgf_agreement_barplot.png", hmm_bgf_agree_plot,
       width = 9, height = 10, dpi = 600)



# ---------------------------------------------------------------------------
# Barplot: pathways where HMM and BGF disagree (both methods made a call,
# but a different one). Unlike agreement, a single disagreement event
# involves two different pathway labels -- HMM's call and BGF's call -- so
# there's no single "the pathway" to group by. Instead this shows, per
# pathway, how often that pathway was HMM's call while BGF disagreed, next
# to how often it was BGF's call while HMM disagreed -- i.e. which pathways
# each method tends to uniquely claim within the disagreement set.
# ---------------------------------------------------------------------------
hmm_bgf_disagree <- df %>%
  filter(!is.na(prediction_hmm) & !is.na(prediction_bgf) & prediction_hmm != prediction_bgf)

hmm_disagree_counts <- hmm_bgf_disagree %>%
  count(prediction_hmm, name = "n") %>%
  rename(prediction = prediction_hmm) %>%
  mutate(method = "HMM")

bgf_disagree_counts <- hmm_bgf_disagree %>%
  count(prediction_bgf, name = "n") %>%
  rename(prediction = prediction_bgf) %>%
  mutate(method = "BGF")

hmm_bgf_disagree_counts <- bind_rows(hmm_disagree_counts, bgf_disagree_counts) %>%
  left_join(pathwaymap, by = c("prediction" = "short_pathways")) %>%
  mutate(pathway = ifelse(is.na(long_pathways), prediction, long_pathways)) %>%
  select(pathway, method, n) %>%
  complete(pathway = canonical_pathways, method = c("HMM", "BGF"), fill = list(n = 0)) %>%
  mutate(
    pathway = factor(pathway, levels = canonical_pathways),
    method  = factor(method, levels = c("HMM", "BGF"))
  )

hmm_bgf_disagree_plot <- ggplot(hmm_bgf_disagree_counts, aes(x = pathway, y = n, fill = method)) +
  geom_col(position = "dodge") +
  coord_flip() +
  scale_x_discrete(drop = FALSE) +
  scale_y_continuous(labels = sci_labeller) +
  labs(x = NULL, y = "Disagreement Count", fill = "Called by",
       title = "Pathways Where HMM and BGF Disagree") +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.y = element_text(size = 10),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right"
  )

ggsave("../../results/figures/misc/hmm_bgf_disagreement_barplot.png", hmm_bgf_disagree_plot,
       width = 9, height = 10, dpi = 600)



# ---------------------------------------------------------------------------
# Venn diagrams: proteins annotated by BGF vs each other method. This is set
# overlap on "did this method make any call at all" for the protein -- it
# doesn't require the two methods to agree on which pathway (see the
# HMM/BGF agreement & disagreement plots above for that finer-grained
# comparison).
#
# Uses eulerr (area-proportional Euler diagrams) instead of ggVennDiagram's
# fixed schematic layout, so circle/overlap area actually scales with the
# magnitude of each intersection rather than every region being drawn the
# same size regardless of count. Requires eulerr
# (install.packages("eulerr") if missing).
# ---------------------------------------------------------------------------
library(eulerr)

diamond_ids <- unique(df$query_id[!is.na(df$prediction_diamond)])
bgf_ids     <- unique(df$query_id[!is.na(df$prediction_bgf)])
hmm_ids     <- unique(df$query_id[!is.na(df$prediction_hmm)])
kegg_ids    <- unique(df$query_id[!is.na(df$prediction_kegg)])

# Saves via png()/dev.off() rather than ggsave() -- eulerr's plot() returns a
# lattice/grid object, not a ggplot, but it prints fine to a graphics device.
save_euler <- function(set_list, title, filename, width = 7, height = 7) {
  fit <- eulerr::euler(set_list)
  p <- plot(
    fit,
    quantities = TRUE,
    fills = list(fill = c("steelblue", "grey70"), alpha = 0.7),
    labels = list(fontsize = 14),
    main = title
  )
  png(filename, width = width, height = height, units = "in", res = 600)
  print(p)
  dev.off()
}

save_euler(
  list(BGF = bgf_ids, DIAMOND = diamond_ids),
  "Shared Annotations: BGF vs DIAMOND",
  "../../results/figures/misc/euler_bgf_diamond.png"
)

save_euler(
  list(BGF = bgf_ids, HMM = hmm_ids),
  "Shared Annotations: BGF vs HMM",
  "../../results/figures/misc/euler_bgf_hmm.png"
)

save_euler(
  list(BGF = bgf_ids, `DIAMOND-KEGG` = kegg_ids),
  "Shared Annotations: BGF vs DIAMOND-KEGG",
  "../../results/figures/misc/euler_bgf_kegg.png"
)

# 4-set version -- all methods at once, still area-proportional.
fit_all <- eulerr::euler(list(
  BGF = bgf_ids, DIAMOND = diamond_ids, HMM = hmm_ids, `DIAMOND-KEGG` = kegg_ids
))
p_all <- plot(
  fit_all,
  quantities = TRUE,
  fills = list(fill = c("steelblue", "grey70", "darkorange", "seagreen"), alpha = 0.7),
  labels = list(fontsize = 14),
  main = "Shared Annotations: BGF vs All Methods"
)
png("../../results/figures/misc/euler_bgf_all_methods.png", width = 9, height = 9, units = "in", res = 600)
print(p_all)
dev.off()


