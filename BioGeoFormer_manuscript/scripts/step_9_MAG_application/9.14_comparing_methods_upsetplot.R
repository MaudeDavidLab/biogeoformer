library(dplyr)
library(stringr)


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

### diamond analysis

diamond <- read.csv("../../cold_seep_MAG_application/DIAMOND_BGF_processed/DIAMOND_BGF_mags_processed.csv")

diamond_selected <- select(diamond, query_id, Cycle)

colnames(diamond_selected) <- c("query_id", "prediction_diamond")


hmms <- read.csv("../../cold_seep_MAG_application/HMM_BGF_processed/HMM_BGF_processed.csv")

hmm_selected <- select(hmms, target_name, pathway)
hmm_selected$pathway <- gsub("cenmetpat_aom", "cenmetpat,aom", hmm_selected$pathway)


colnames(hmm_selected) <- c("query_id", "prediction_hmm")

kegg <- read.csv("../../cold_seep_MAG_application/KEGG_processed/kegg_annotated_withpathways.csv")

kegg_selected <- select(kegg, Sequence_ID, Pathway)

colnames(kegg_selected) <- c("query_id", "prediction_kegg")

# BioGeoFormer (formerly "cycformer") model predictions, bgf_train_50 split --
# matches the train-split percentage used for the HMM run above. Raw output
# covers every MAG ORF; the (large, ~2GB) sequence column is skipped on read
# since only the label/confidence are needed here.
bgf <- readr::read_csv(
  "../../cold_seep_MAG_application/BGF_annotations/cleaned_MAGS_forbgf_output_50.csv",
  col_select = c(seq_id, predicted_label, confidence),
  show_col_types = FALSE
)

bgf_filtered <- filter(bgf, confidence >= 0.733)

bgf_selected <- select(bgf_filtered, seq_id, predicted_label)

colnames(bgf_selected) <- c("query_id", "prediction_bgf")

#train_overlap <- read.csv("train80_overlap_toremove.csv")



library(dplyr)

# Start with one and progressively join the others
joined_predictions <- diamond_selected %>%
  full_join(hmm_selected, by = "query_id") %>%
  full_join(kegg_selected, by = "query_id") %>%
  full_join(bgf_selected, by = "query_id")



#joined_predictions <- joined_predictions %>%
#  anti_join(train_overlap, by = c("query_id" = "query_id"))


write.csv(joined_predictions, "../../cold_seep_MAG_application/joined_predictions_coldseep_mags.csv", row.names = FALSE)

# Load libraries
library(tidyverse)
library(entropy)

# Example: read your dataframe (adjust if you're loading from a CSV or RDS)
# df <- read.csv("your_predictions.csv", stringsAsFactors = FALSE)

# Replace NA strings ("<NA>") if necessary
df <- joined_predictions %>%
  mutate(across(starts_with("prediction_"), ~na_if(.x, "<NA>")))

# Compute number of unique predictions per row (ignoring NAs)
df$unique_predictions <- df %>%
  select(starts_with("prediction_")) %>%
  apply(1, function(x) length(unique(na.omit(x))))

# Count number of non-NA predictions per row
df$num_predictions <- df %>%
  select(starts_with("prediction_")) %>%
  apply(1, function(x) sum(!is.na(x)))

# Compute entropy per row (higher entropy = more disagreement)
df$entropy <- df %>%
  select(starts_with("prediction_")) %>%
  apply(1, function(x) {
    preds <- na.omit(x)
    freqs <- table(preds)
    if (length(freqs) <= 1) return(0)
    entropy.empirical(freqs / sum(freqs), unit = "log2")
  })

# Classify agreement level
df$agreement_category <- case_when(
  df$unique_predictions == 1 & df$num_predictions == 4 ~ "4 Agree",
  df$unique_predictions == 1 & df$num_predictions == 3 ~ "3 Agree",
  df$unique_predictions == 1 & df$num_predictions == 2 ~ "2 Agree",
  df$unique_predictions == df$num_predictions ~ "All Disagree",
  TRUE ~ "Partial Agreement"
)





# Summarize and plot
agreement_summary <- df %>%
  count(agreement_category) %>%
  arrange(desc(n))

print(agreement_summary)

# Optional: barplot
ggplot(agreement_summary, aes(x = reorder(agreement_category, -n), y = n)) +
  geom_col(fill = "steelblue") +
  labs(title = "Model Agreement Summary", x = "Agreement Category", y = "Number of Queries") +
  theme_minimal()




library(ComplexUpset)

df <- joined_predictions %>%
  mutate(across(starts_with("prediction_"), ~na_if(.x, "<NA>")))

# Filter out rows where fewer than 2 models made predictions
# (vectorized rowSums instead of rowwise()+c_across(), which is unusably
# slow at >1M rows -- rowwise() processes one row at a time with real
# per-row dplyr overhead)
df$non_na <- rowSums(!is.na(as.matrix(select(df, starts_with("prediction_")))))
df_filtered <- df %>% filter(non_na >= 2)

# Function: check which models agreed with each other per row.
# Majority vote genuinely needs a per-row computation, but base apply() on
# a plain matrix avoids dplyr::rowwise()'s grouping overhead and is far
# faster in practice than rowwise()+c_across() at this scale.
pred_mat <- as.matrix(select(df_filtered, starts_with("prediction_")))
majority_vote <- apply(pred_mat, 1, function(row) {
  row <- row[!is.na(row)]
  if (length(row) == 0) return(NA_character_)
  names(sort(table(row), decreasing = TRUE))[1]
})

df_upset <- df_filtered %>%
  mutate(
    majority_vote = majority_vote,
    diamond_agree = prediction_diamond == majority_vote,
    hmm_agree     = prediction_hmm     == majority_vote,
    kegg_agree    = prediction_kegg    == majority_vote,
    bgf_agree     = prediction_bgf     == majority_vote
  )

upset_data <- df_upset %>%
  select(query_id, diamond_agree, hmm_agree, kegg_agree, bgf_agree) %>%
  dplyr::rename(
    diamond    = diamond_agree,
    hmm        = hmm_agree,
    kegg       = kegg_agree,
    bgf        = bgf_agree
  ) %>%
  mutate(across(c(diamond, hmm, kegg, bgf), ~ replace_na(.x, FALSE))) %>%
  mutate(across(c(diamond, hmm, kegg, bgf), ~ as.integer(.))) %>%
  mutate(n = 1)

options(warn = 0)  # warnings print, but won’t stop execution

upset(upset_data, intersect = c("diamond", "hmm", "kegg", "bgf"),
      base_annotations = list('Intersection size' = intersection_size())) 




upset_data_pretty <- upset_data %>%
  rename(
    `Diamond-BGFdb` = diamond,
    HMM = hmm,
    `Diamond-KEGG` = kegg,
    BGF = bgf
  )

upset(
  upset_data_pretty,
  intersect = c("Diamond-BGFdb", "HMM", "Diamond-KEGG", "BGF"),
  name = "Model Agreement",
  sort_intersections_by = "cardinality",
  sort_sets = "descending"
)


intersection_no_labels <- intersection_size(text = list(element_blank()))

# Create the plot
intersection_no_labels <- intersection_size()
intersection_no_labels$layers <- intersection_no_labels$layers[1]  # remove geom_text layer

upset_plot <- upset(
  upset_data_pretty,
  intersect = c("Diamond-BGFdb", "HMM", "Diamond-KEGG", "BGF"),
  base_annotations = list(
    "Intersection size" = intersection_size() +
      theme(
        # Removes only the bar-top numbers
        panel.grid = element_blank(),
        axis.text.y = element_text(color = "black"),
        axis.title.y = element_text(color = "black"),
        plot.title = element_text(size = 14),
        plot.subtitle = element_blank()  # Suppress embedded text in recent ComplexUpset
      )
  ),
  sort_intersections_by = "cardinality",
  sort_sets = "descending"
)



library(dplyr)
library(ComplexUpset)
library(ggplot2)

# add a column with number of agreeing models
upset_data_colored <- upset_data_pretty %>%
  mutate(
    num_models = rowSums(select(., `Diamond-BGFdb`, HMM, `Diamond-KEGG`, BGF))
  )

upset_plot <- upset(
  upset_data_colored,
  intersect = c("Diamond-BGFdb", "HMM", "Diamond-KEGG", "BGF"),
  base_annotations = list(
    "Intersection size" = (
      intersection_size(
        mapping = aes(fill = num_models),
        text = list(size = 0)  # suppress text annotations on bars
      ) +
        scale_fill_gradient(low = "lightblue", high = "darkblue") +
        theme(
          panel.grid   = element_blank(),
          axis.text.y  = element_text(color = "black", size = 14),
          axis.title.y = element_text(color = "black", size = 16),
          plot.title   = element_text(size = 18, face = "bold"),
          plot.subtitle = element_blank(),
          axis.text.x  = element_blank(),     # 🔥 removes 3-1, 2-3-4-1 labels
          axis.ticks.x = element_blank(),     # also removes tick marks
          axis.title.x = element_text(size = 16)
        )
    )
  ),
  sort_intersections_by = "cardinality",
  sort_sets = "descending"
)




ggsave("../../results/figures/misc/upset_plot.png", plot = upset_plot, width = 10.5, height = 9, dpi = 300)
ggsave("../../results/figures/misc/upset_plot.svg", plot = upset_plot, width = 10.5, height = 9, dpi = 300)


colSums(upset_data_pretty[c("Diamond-BGFdb", "HMM", "Diamond-KEGG", "BGF")])







df <- joined_predictions %>%
  mutate(across(starts_with("prediction_"), ~na_if(.x, "<NA>")))

# Compute number of unique predictions per row (ignoring NAs)
df$unique_predictions <- df %>%
  select(starts_with("prediction_")) %>%
  apply(1, function(x) length(unique(na.omit(x))))

# Count number of non-NA predictions per row
df$num_predictions <- df %>%
  select(starts_with("prediction_")) %>%
  apply(1, function(x) sum(!is.na(x)))


# Classify agreement level
df$agreement_category <- case_when(
  df$unique_predictions == 1 & df$num_predictions == 4 ~ "4 Agree",
  df$unique_predictions == 1 & df$num_predictions == 3 ~ "3 Agree",
  df$unique_predictions == 1 & df$num_predictions == 2 ~ "2 Agree",
  df$unique_predictions == df$num_predictions ~ "All Disagree",
  TRUE ~ "Partial Agreement"
)



write.csv(df, "../../cold_seep_MAG_application/model_predictions_mags_allmodels_df.csv", row.names = FALSE)



df_long <- df %>%
  pivot_longer(
    cols = starts_with("prediction_"),
    names_to = "method",
    values_to = "annotation"
  ) %>%
  filter(!is.na(annotation)) %>%   # drop NAs
  mutate(
    method = gsub("prediction_", "", method),
    method = factor(method, levels = c("diamond", "hmm", "kegg", "bgf"))
  )

# Count per method × num_predictions
annotation_counts <- df_long %>%
  group_by(method, num_predictions) %>%
  summarise(n = n(), .groups = "drop")


sci_labeller <- function(x) {
  parse(text = gsub("e\\+?", " %*% 10^", scientific_format()(x)))
}


# Stacked barplot styled like your barplot_total
barplot_agreement <- ggplot(annotation_counts, 
                            aes(x = method, y = n, fill = factor(num_predictions))) +
  geom_col(alpha = 0.9) +
  coord_flip() +
  scale_x_discrete(labels = c("DIAMOND-BGFdb", "HMM", "DIAMOND-KEGG", "BGF")) +
  scale_y_continuous() +   # your scientific notation formatter
  scale_fill_manual(
    values = c("1" = "lightblue",
               "2" = "#6CA6CD",
               "3" = "#1E5AA8",
               "4" = "darkblue"),
    name = "Methods agreed"
  ) +
  labs(
    x = "", 
    y = "Total Gene Annotations",
    title = " "
  ) +
  theme_minimal() +
  theme(
    axis.text   = element_text(size = 12, color = "black"),
    axis.title  = element_text(size = 14, color = "black"),
    plot.title  = element_text(size = 16, hjust = 0.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.title = element_text(size = 12),
    legend.text  = element_text(size = 11)
  )



barplot_agreement_quals <- ggplot(annotation_counts, 
                            aes(x = method, y = n, fill = factor(num_predictions))) +
  geom_col(alpha = 0.9) +
  scale_x_discrete(labels = c("DIAMOND-BGFdb", "HMM", "DIAMOND-KEGG", "BGF")) +
  scale_y_continuous(labels = function(x) {
    ifelse(x == 0, "0", 
           parse(text = paste0(format(x / 10^floor(log10(abs(x))), digits = 1), 
                               " %*% 10^", floor(log10(abs(x))))))
  }) +
  scale_fill_manual(
    values = c("1" = "lightblue",
               "2" = "#6CA6CD",
               "3" = "#1E5AA8",
               "4" = "darkblue"),
    name = "Methods agreed"
  ) +
  labs(
    x = "", 
    y = "Total Gene Annotations",
    title = " "
  ) +
  theme_minimal() +
  theme(
    axis.text   = element_text(size = 12, color = "black"),
    axis.title  = element_text(size = 14, color = "black"),
    plot.title  = element_text(size = 16, hjust = 0.5),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.title = element_text(size = 12),
    legend.text  = element_text(size = 11)
  )






library(patchwork)
library(ggplotify)

library(cowplot)



combined <- upset_plot | barplot_agreement

# Adjust relative widths so barplot is narrower
combined <- combined + plot_layout(widths = c(3, 1))


ggsave("../../results/figures/misc/upset_with_barplot.svg",
       combined,
       width = 16, height = 9, dpi = 600)


ggsave("../../results/figures/misc/upset_with_barplot.png",
       combined,
       width = 16, height = 9, dpi = 600)








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

# 3. Barplot
ggplot(cycle_counts, aes(x = prediction_bgf, y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +   # 🔥 horizontal orientation
  labs(
    x = "Cycle predicted by BGF only",
    y = "Number of genes",
    title = "Unique BGF Predictions"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.y = element_text(size = 12, color = "black"),
    axis.title  = element_text(size = 14, color = "black"),
    plot.title  = element_text(size = 16, hjust = 0.5)
  )














