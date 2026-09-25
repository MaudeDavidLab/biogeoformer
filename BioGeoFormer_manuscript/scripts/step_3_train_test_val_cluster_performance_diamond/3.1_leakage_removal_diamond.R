library(tidyverse)
library(patchwork)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# 1. SETUP PATHS AND PARAMETERS
benchmark_dir <- "../../BGF_clustering/comparing_train_test_val_diamond/"
out_dir <- "../../BGF_clustering/graphpart_raw/"
thresholds <- seq(10, 90, by = 10)
cols <- c("qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue")

leakage_list <- list()
full_dist_list <- list()
coverage_list <- list()

# 2. DATA LOADING LOOP
for (T in thresholds) {
  for (split in c("test", "val")) {
    
    # Reset temp variables for each iteration
    temp_leak_raw <- NULL
    
    leak_file <- paste0(benchmark_dir, "leak_", split, "_T", T, ".tsv")
    bgf_file  <- paste0(out_dir, "bgf_", split, "_", T, ".csv")
    
    # Process DIAMOND Hits
    if (file.exists(leak_file)) {
      temp_leak_raw <- read_tsv(leak_file, col_names = cols, show_col_types = FALSE)
      
      if(nrow(temp_leak_raw) > 0) {
        # A. Store the FULL distribution for density/boxplots
        full_dist_list[[paste0(split, T)]] <- temp_leak_raw %>%
          mutate(Threshold = factor(T, levels = thresholds), 
                 Split = str_to_title(split))
        
        # B. Filter for ACTUAL leaks (pident > T) for the barplot
        temp_leak <- temp_leak_raw %>% filter(pident > T)
        if(nrow(temp_leak) > 0) {
          leakage_list[[paste0(split, T)]] <- temp_leak %>%
            mutate(Threshold = factor(T, levels = thresholds), 
                   Split = str_to_title(split))
        }
      }
    }
    
    # Process BGF Totals for Coverage
    if (file.exists(bgf_file)) {
      bgf_count <- nrow(read.csv(bgf_file))
      
      # Use the count of ACTUAL leaks (pident > T)
      leak_count <- if (!is.null(leakage_list[[paste0(split, T)]])) {
        nrow(leakage_list[[paste0(split, T)]])
      } else {
        0
      }
      
      coverage_list[[paste0(split, T)]] <- data.frame(
        Threshold = factor(T, levels = thresholds),
        Split = str_to_title(split),
        Status = c("Leakage", "Clean"),
        Count = c(leak_count, bgf_count - leak_count)
      )
    }
  }
}

# Combine into master dataframes
df_full_dist <- bind_rows(full_dist_list)
df_all_leakage <- bind_rows(leakage_list)
df_all_coverage <- bind_rows(coverage_list) %>%
  group_by(Threshold, Split) %>%
  mutate(Percent = (Count / sum(Count)) * 100)

# 3. PLOTTING

# Plot A: Full Distribution with Threshold Indicators
p1_all <- ggplot(df_full_dist, aes(x = pident, fill = Split)) +
  geom_density(alpha = 0.5) +
  # Adds a red dashed line at the threshold value for each facet
  geom_vline(aes(xintercept = as.numeric(as.character(Threshold))), 
             linetype = "dashed", color = "red", alpha = 0.8) +
  facet_wrap(~Threshold, ncol = 3, labeller = label_both) + 
  theme_minimal() +
  scale_fill_manual(values = c("Test" = "#F8766D", "Val" = "#00BFC4")) +
  labs(title = "Full Top-Hit Distribution (Red Line = GraphPart T)", 
       x = "% Identity", y = "Density")

# Plot B: Global Boxplot Trends
p2_all <- ggplot(df_full_dist, aes(x = Threshold, y = pident, fill = Split)) +
  geom_boxplot(outlier.size = 0.2, alpha = 0.7) +
  theme_minimal() +
  scale_fill_manual(values = c("Test" = "#F8766D", "Val" = "#00BFC4")) +
  labs(title = "Global Identity Trends", x = "GraphPart T", y = "% Identity")

# Plot C: Coverage Rates (Red for Leakage)
p3_all <- ggplot(df_all_coverage, aes(x = Threshold, y = Percent, fill = Status)) +
  geom_bar(stat = "identity", position = position_fill()) +
  scale_y_continuous(labels = scales::percent_format()) +
  facet_wrap(~Split) +
  theme_minimal() +
  scale_fill_manual(values = c("Leakage" = "red", "Clean" = "grey85")) +
  labs(title = "Leakage Rate (Hits > Threshold %)", x = "GraphPart T", y = "% of Dataset")

# Final Combined Layout
combined_plot <- wrap_plots(p1_all, wrap_plots(p2_all, p3_all), ncol = 1, heights = c(2, 1))

# Render plot
combined_plot




################################### LEAKAGE FILTERING ########################################






library(tidyverse)

# Define paths
out_dir <- "../../BGF_clustering/graphpart_raw/"
thresholds <- seq(10, 90, by = 10)


# 1. Clean the master leakage list IDs
# This removes everything from the first '|' to the end of the string
df_all_leakage_clean <- df_all_leakage %>%
  mutate(qseqid_clean = str_remove(qseqid, "\\|.*$"))

# Create a 'cleaned' subdirectory (sibling of graphpart_raw/, not nested in it)
clean_out_dir <- "../../BGF_clustering/cleaned/"
dir.create(clean_out_dir, showWarnings = FALSE)

for (T_val in thresholds) {
  for (split_val in c("Test", "Val")) {
    
    file_name <- paste0("bgf_", tolower(split_val), "_", T_val, ".csv")
    input_path <- paste0(out_dir, file_name)
    output_path <- paste0(clean_out_dir, file_name)
    
    if (file.exists(input_path)) {
      current_df <- read.csv(input_path)
      
      # 2. Identify IDs to remove using the CLEANED version
      ids_to_remove <- df_all_leakage_clean %>%
        filter(Threshold == T_val, Split == split_val) %>%
        pull(qseqid_clean)
      
      # 3. Filter the CSV (assuming the column name in your CSV is 'id')
      clean_df <- current_df %>%
        filter(!(id %in% ids_to_remove))
      
      write.csv(clean_df, output_path, row.names = FALSE)
      
      removed_count <- nrow(current_df) - nrow(clean_df)
      message(paste0("Processed ", split_val, " T", T_val, ": Removed ", removed_count, " sequences."))
    }
  }
}



library(tidyverse)

# 1. Parse the Gene and Category from the ID string
# Format: Accession|GeneName|Category
df_leakage_details <- df_all_leakage %>%
  separate(qseqid, 
           into = c("Accession", "Gene", "Category"), 
           sep = "\\|", 
           remove = FALSE, 
           extra = "merge")

# 2. Create the Summary Table
# We group by Split and Gene to see which metabolic pathways were "leakiest"
summary_removals <- df_leakage_details %>%
  group_by(Threshold, Split, Gene, Category) %>%
  summarise(Count = n(), .groups = "drop") %>%
  # Pivot wider to make it easy to read Thresholds as columns
  pivot_wider(names_from = Threshold, 
              values_from = Count, 
              values_fill = 0) %>%
  arrange(Split, desc(`40`)) # Sorting by the T=40 spike we saw earlier

# 3. View the top of the table
print(head(summary_removals, 20))

# 4. Export for your Supplementary Data
write.csv(summary_removals, "../../BGF_clustering/cleaned/leakage_summary_by_gene.csv", row.names = FALSE)






library(tidyverse)

# 1. SETUP PATHS
# Use the 'cleaned' directory we created in the previous step
clean_out_dir <- "../../BGF_clustering/cleaned/"
# The original DIAMOND hits are still needed for the check
benchmark_dir <- "../../BGF_clustering/comparing_train_test_val_diamond/"

thresholds <- seq(10, 90, by = 10)
cols <- c("qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue")

# Initialize a results table
verification_summary <- data.frame()

# 2. VERIFICATION LOOP
for (T_val in thresholds) {
  for (split_val in c("test", "val")) {
    
    clean_file <- paste0(clean_out_dir, "bgf_", split_val, "_", T_val, ".csv")
    leak_file  <- paste0(benchmark_dir, "leak_", split_val, "_T", T_val, ".tsv")
    
    if (file.exists(clean_file) && file.exists(leak_file)) {
      # Load cleaned metadata
      df_clean <- read.csv(clean_file)
      
      # Load original hits and clean the IDs (removing | and everything after)
      df_hits <- read_tsv(leak_file, col_names = cols, show_col_types = FALSE) %>%
        mutate(qseqid_clean = str_remove(qseqid, "\\|.*$"))
      
      # Logic: Check if any ID currently in the 'clean' CSV still exists 
      # in the hit list where the identity is above the threshold
      remaining_leaks <- df_hits %>%
        filter(qseqid_clean %in% df_clean$id, pident > T_val)
      
      # Record results
      verification_summary <- bind_rows(verification_summary, data.frame(
        Threshold = T_val,
        Split = split_val,
        Initial_Count = nrow(df_clean),
        Leaks_Found = nrow(remaining_leaks)
      ))
      
    } else {
      message(paste("Missing files for T", T_val, split_val))
    }
  }
}

# 3. PRINT THE VERDICT
print("--- Leakage Verification Results ---")
print(verification_summary)

if (sum(verification_summary$Leaks_Found) == 0) {
  message("SUCCESS: All datasets are now 100% leak-free based on the specified thresholds!")
} else {
  warning("FAILURE: Some leaks still persist. Check the verification_summary table.")
}










