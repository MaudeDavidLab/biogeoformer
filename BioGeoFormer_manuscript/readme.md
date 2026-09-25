# File structure for the BioGeoFormer manuscript

### BGF_clustering: Folder for clustering and evaluating the BGF database
* graphpart_raw: raw GraphPart train and test+validation partitions for each identity split (10-90%), in .csv and .fasta format.
* comparing_train_test_val_diamond: DIAMOND output comparing validation and test sequences against the training set for each identity split, used to find sequences above the split's identity threshold.
* cleaned: validation and test sets after removing sequences above each split's identity threshold with the training set. Also includes each sequence's identity to the training set and whether it was kept or removed (similarity_status_3way.csv, similarity_status_3way_val.csv).
* train_test_val_final: final training/validation/test sets after removing pathways with too few sequences (training > 100, validation > 30, test > 30). These are in .csv and .fasta format.
* train_test_val_fasta_bypathway: training sets split by pathway and gene for building HMMs. manifest_all_splits.csv maps each gene to its pathway.

### BGF_run_val_test: output of BGF on the test sets
* test: BGF inference on the test set of each identity split (sim_10-sim_90), with predictions (test_predictions.csv) and the probability of every class (test_probs_long.parquet).

### BioGeoFormer_db: constructed BioGeoFormer_db database (BioGeoFormer_db.csv, BioGeoFormer_db.fasta), and overview of the genes and pathways in the database (table_S1_BioGeoFormer_db_overview.csv, BioGeoFormer_db_overview.docx)

### cold_seep_MAG_application: Folder for annotating metagenome-assembled genomes and comparing method performance
* BGF_annotations: BGF output after inference with the 50% identity split model.
* BGF_input_data: MAG data processed into a fasta file that BGF can run inference on.
* DIAMOND_BGF_output: Output from running BGFdb DIAMOND alignment against MAGs.
* DIAMOND_BGF_processed: Processing BGFdb DIAMOND output into a .csv file.
* DIAMOND_KEGG_output: Output from running KEGG database DIAMOND alignment against MAGs.
* HMM_BGF_output: Output from running HMMs constructed with the BGFdb, and the best hit for each sequence after filtering. The 50% identity split was selected for this application.
* HMM_BGF_processed: Processing HMM output into a .csv file.
* joined_predictions_coldseep_mags.csv: File that contains predictions from each method on the MAG dataset.
* model_predictions_mags_allmodels_df.csv: File that contains predictions from each method alongside the majority vote across methods.
* KEGG_mapping: Folder containing files for mapping KO to gene ID and then biogeochemical pathway.
* KEGG_processed: Processed KEGG output into a .csv file.
* mags_processed: MAG data processed into a .faa and .csv file as opposed to unique .fa files for each MAG.

## combined_cycdb: Folder containing 'combined100.faa', a a combination of MCycDB, NCycDB, SCycDB, and PCycDB databases. 

## cycdb_csv: Folder containing processing steps of the CycDB databases. 
* Metadata: Processed metadata from each of the CycDB databases
* Metadata_with_pathways: Metadata with biogeochemical pathways mapped onto each file.
* sequences: CycDB databases with metadata next to ID's removed for downstream processing.

## cycle_dicts: Dictionaries (.json and .txt files) of ID's mapped to their corresponding biogeochemical pathways. 

## graphpart_diamondrun: DIAMOND output of the validation and test sets against the training set for each identity split. The best hit is used as DIAMOND's prediction.
* perclass_bench: the same search run against one training database per pathway, used for DIAMOND precision-recall curves.

## hmm_runs: HMM output and processing for the validation and test sets.
* scan_results: hmmscan output on the validation and test sets.
* scan_results_max_besthit: hmmscan output without score thresholds (splits 10-40), reduced to the best hit for each sequence.
* pathway_prediction_summary.csv, pathway_prediction_per_class.csv, pathway_prediction_detail.csv: HMM performance for each split, each pathway, and each sequence.

## mmseqs_identity_to_train: MMseqs2 identity of each test sequence to its closest training sequence for each identity split, with no coverage or identity cutoff.

## results: Figures and tables generated for the BioGeoFormer manuscript
* figures/main_text, figures/supplement: figures named by their number in the manuscript (e.g., Fig2, SFig4).
* tables/main_text, tables/supplement: tables named by their number in the manuscript (e.g., Table1, STable2).
* All other folders hold the output of the scripts that generate the figures and tables.

## scripts: scripts ordered to reproduce the BioGeoFormer manuscript

## temperature_scaling_BGF: temperature scaling output for BGF
* bgf_10_init1.0-bgf_90_init1.0: reliability diagrams before and after temperature scaling, and the fitted temperature, for each model's identity split.

## test_set_annotations: BGF predictions on the test set of each identity split (test_predictions_10-90.csv), and the confidence threshold chosen for each split (confidence_thresholds.csv).
