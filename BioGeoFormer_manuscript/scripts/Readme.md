# Readme for scripts in BGF

## Step_1_BioGeoFormer_db

* in order to run this step you must have the databases downloaded: MCycDB (https://github.com/qichao1984/MCycDB), NCycDB (https://github.com/qichao1984/NCyc), SCycDB (https://github.com/qichao1984/SCycDB), and PCycDB (https://github.com/ZengJiaxiong/Phosphorus-cycling-database). Once downloaded, place each in the BioGeoFormer folder within this repository. 

* 1.1-1.4: scripts that create the .csv files of the metadata and database, for each respective database -- MCycDB (1.1), NCycDB (1.2), PCycDB (1.3), and SCycDB (1.4). 
* 1.5-1.6: scripts that group genes based on their respective biogeochemical pathways. A given ID may be assigned to one or more biogeochemical pathways at this point. 
* 1.9: Creation of the BioGeoFormer_db (BGF_db). This entails filtering out genes that fall into two or more pathways, with the exception of anaerobic oxidation of methane and the central methanogenic pathway. Combining all databases together, and mapping genes to pathways. 
* 1.10: Creating a fasta file of BioGeoFormer_db


## Step_2_clustering_train_test_val_BGF

* 2.1: GraphPart command used to split BGF_db into a training partition and a combined validation/test partition, stratified by pathway, at each identity split (10-90%). Shown for the 30% split. These commands were run on an HPC.
* 2.2: Script to split the validation/test partition 50/50 by pathway, giving a roughly 60/20/20 training/validation/test split.

## Step_3_train_test_val_cluster_performance_diamond

* 3.1: Script using DIAMOND output (training set against validation and test sets, >= 80% query and subject coverage) to remove validation and test sequences above each split's identity threshold. The DIAMOND commands were run on an HPC.
* 3.2: Removing pathways with too few sequences from all three sets (training > 100, validation > 30, test > 30), giving the final training/validation/test sets.
* 3.3: Plotting each validation and test sequence's identity to the training set, colored by whether it was kept, removed by 3.2, or removed by 3.1 (Supplementary Figs 1 and 2).
* 3.4: Writing fasta files for the training, validation, and test datasets.
* 3.5: Writing fasta files of the training sets split by pathway and gene, for building HMMs.

## Step_4_hmm_runs
* Building HMM profiles for each gene from the training sets, and running them against the validation and test sets, were done on an HPC.
* 4.4: Script to process the HMM runs on the validation and test datasets into predictions and performance metrics for each identity split.

## Step_5_diamond_runs
* DIAMOND databases were built for each training split and run against the validation and test sets on an HPC.
* 5.4: Script to process the DIAMOND output into predictions and performance metrics for each identity split.

## Step_6_training_scaling_running_BGF
* 6.1: Scripts to train BioGeoFormer on each identity split's training set.
* 6.2: Scripts to carry out temperature scaling on all model splits post-training. These scripts were run on an A100 GPU on Google Colab.
* 6.3: Scripts used to run BioGeoFormer against test sets. test_inference writes the top prediction for each sequence, and test_inference_full_preds also writes the probability of every class. These scripts were run on an A100 GPU on Google Colab.

## Step_7_machine_learning
* ml_bench.py, ml_bench.sh: Machine learning baselines (k-nearest neighbors, logistic regression, random forest) trained on ESM-2 (8M, 35M, 150M) embeddings for each identity split.

## Step_8_evaluating_models
* 8.1: Scripts to score every sequence against every pathway for DIAMOND (per-pathway databases, run on an HPC), HMMs, and BGF, and build precision-recall curves.
* 8.2: Performance metric evaluations and comparison of methods (BGF, HMMs, DIAMOND) using Matthews Correlation Coefficient, and weighted Accuracy, Precision, Recall, and F1, with and without confidence thresholds. Also calculates the share of predictions going to each pathway.
* 8.3: Plotting precision-recall curves, area under the precision-recall curve, and precision of high-confidence predictions for each method (Fig 2, Supplementary Figs 4 and 6-8).
* 8.4: Heatmap of BGF MCC by each test sequence's MMseqs2 identity to the training set (Fig 4c).
* 8.5: Heatmap of BGF precision for each pathway across identity splits (Supplementary Fig 9).

## Step_9_MAG_application
* In order to run this step, you must download the deduplicated metagenome-assembled genomes (MAGs) from Han et al., 2023 (https://identifiers.org/ncbi/bioproject:PRJNA950938)
  
* 9.1: calling open reading frames (ORFs) for MAGs using Prodigal. These commands were run on an HPC.
* 9.2: DIAMOND run annotating MAG ORFs using the KEGG database. Requires a local KEGG database to run these commands. These commands were run on an HPC. 
* 9.3: DIAMOND run annotating MAG ORFs using the BGFdb database. These commands were run on an HPC.
* 9.4: HMM run annotating MAG ORFs against the HMMs constructed from the 50% identity split training set (9.4a, run on an HPC), and keeping the best hit for each ORF (9.4b).
* 9.5: Script processing the KEGG DIAMOND output into a .csv file.
* 9.6: Script to parse the KEGG orthology database from KEGG BRITE. Prior to running this script, download the ko0001 file from (https://www.kegg.jp/kegg-bin/get_htext?ko00001).
* 9.7: Script assigning gene names to KO IDs.
* 9.8: Mapping biogeochemical pathways to KEGG file. Primarily using gene names assigned in script 9.7.
* 9.9: Processing DIAMOND output for BGFdb, into a .csv file.
* 9.10: Processing HMM output for BGFdb, into a .csv file.
* 9.11: Formatting MAG data to be run by BioGeoFormer. Shortening sequences to more manageably work with the file, however not so short that they are included as complete proteins by BGF (context window 1024 tokens).
* 9.13: BGF inference of MAG data with the 50% identity split model. This script was run on Google Colab with an A100 GPU.
* 9.14: Script to compare annotation methods with one another, and plotting an Upset plot to visualize the comparison.
* 9.15: Script making a bubble plot to compare annotation counts for each pathway by each method. Additionally plotting a bar plot of only BGF annotations and appending it to the bubble plot.
* 9.16: Histogram of the identity of each MAG ORF's top DIAMOND hit against BGFdb (Supplementary Fig 11).
