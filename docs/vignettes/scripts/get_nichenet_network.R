library(nichenetr)
library(dplyr)
library(tools)
library(readr)
library(optparse)
options(download.file.method = "libcurl")




# Load matrices
# if path exist load them, otherwise download and save env
matrices_dir <- "db"
matrices_file <- "matrices.rda"
matrices_path <- file.path(matrices_dir, matrices_file)

if (file.exists(matrices_path)) {

  message("Loading cached matrices.rda")
  load(matrices_path)

} else {

  dir.create(matrices_dir, showWarnings = FALSE)

  download_if_needed <- function(url, dest) {
    if (!file.exists(dest)) {
      message(paste("Downloading:", url))
      for (i in 1:5) {
        tryCatch({
          download.file(url, dest, mode = "wb")
          return(TRUE)
        }, error = function(e) {
          message(paste("Attempt", i, "failed. Retrying..."))
          Sys.sleep(5 * i)
        })
      }
      stop(paste("Failed to download:", url))
    }
  }

  files <- list(
    weighted_networks = "https://zenodo.org/record/7074291/files/weighted_networks_nsga2r_final.rds",
    ligand_tf_matrix  = "https://zenodo.org/record/7074291/files/ligand_tf_matrix_nsga2r_final.rds",
    lr_network        = "https://zenodo.org/record/7074291/files/lr_network_human_21122021.rds",
    sig_network       = "https://zenodo.org/record/7074291/files/signaling_network_human_21122021.rds",
    gr_network        = "https://zenodo.org/record/7074291/files/gr_network_human_21122021.rds"
  )

  paths <- setNames(
    file.path(matrices_dir, paste0(names(files), ".rds")),
    names(files)
  )

  # Download only missing files
  mapply(download_if_needed, files, paths)

  # Load from disk (robust)
  weighted_networks <- readRDS(paths[["weighted_networks"]])
  ligand_tf_matrix  <- readRDS(paths[["ligand_tf_matrix"]])
  lr_network        <- readRDS(paths[["lr_network"]])
  sig_network       <- readRDS(paths[["sig_network"]])
  gr_network        <- readRDS(paths[["gr_network"]])

  # Save everything in one file
  save(
    weighted_networks,
    ligand_tf_matrix,
    lr_network,
    sig_network,
    gr_network,
    file = matrices_path
  )
}
# Argument parsing
option_list <- list(
  make_option(c("-i", "--input"), type="character", default=NULL,
             help="Input CSV file path", metavar="file"),
  make_option(c("-t", "--top_n_regulators"), type="integer", default=4,
             help="Top N regulators for signaling network", metavar="integer"),
  make_option(c("-r", "--consider_receptors"), type="logical", default=FALSE,
             help="Consider receptors in network construction", metavar="logical"),
  # ADD THIS LINE:
  make_option(c("-o", "--output_dir"), type="character", default=NULL,
             help="Directory to save outputs", metavar="directory")
)

opt <- parse_args(OptionParser(option_list = option_list))

input_path <- opt$input
top_n_regulators <- opt$top_n_regulators
consider_receptors <- opt$consider_receptors

sample_name <- basename(file_path_sans_ext(opt$input))

if (!is.null(opt$output_dir)) {
  output_path <- opt$output_dir
} else {
  output_path <- file.path("results", sample_name)
}


if (!dir.exists(output_path)){
  dir.create(output_path, recursive = TRUE)
}

# Load CSV file and get vectors:
df <- read.csv(input_path,
               header = TRUE,
               fill = TRUE,
               na.strings = c("","NA"),
               stringsAsFactors = FALSE)

receptors <- df$receptors[!is.na(df$receptors)]
targets_oi <- df$targets[!is.na(df$targets)]

# Get ligands activating receptors of interest from LR network
ligands_oi = unique(lr_network[lr_network$to %in% receptors,]$from)


valid_targets <- unique(weighted_networks$gr$to)
invalid_targets <- setdiff(targets_oi, valid_targets)
if (length(invalid_targets) > 0) {
  warning(paste(
    "Removing target genes not found in NicheNet's gene regulatory network database:", 
    paste(invalid_targets, collapse = ", ")
  ))
  targets_oi <- intersect(targets_oi, valid_targets)
}

valid_ligands <- colnames(ligand_tf_matrix)
invalid_ligands <- setdiff(ligands_oi, valid_ligands)
if (length(invalid_ligands) > 0) {
  warning(paste(
    "Removing ligands not found in NicheNet's ligand-TF database:", 
    paste(invalid_ligands, collapse = ", ")
  ))
  ligands_oi <- intersect(ligands_oi, valid_ligands)
}

# Exit cleanly if filtering results in empty sets
if (length(targets_oi) == 0) {
  stop("Error: No valid target genes remaining after filtering against the regulatory network.")
}
if (length(ligands_oi) == 0) {
  stop("Error: No valid ligands remaining after filtering against the ligand-TF matrix.")
}
# ==========================================


# Compute signalling network
if (consider_receptors) {
  active_signaling_network <- get_ligand_signaling_path_with_receptor(
    ligands_all = ligands_oi,
    targets_all = targets_oi,
    weighted_networks = weighted_networks,
    ligand_tf_matrix = ligand_tf_matrix,
    top_n_regulators = top_n_regulators,
    recepts = receptors
  )
} else {
  active_signaling_network <- get_ligand_signaling_path(
    ligands_all = ligands_oi,
    targets_all = targets_oi,
    weighted_networks = weighted_networks,
    ligand_tf_matrix = ligand_tf_matrix,
    top_n_regulators = top_n_regulators
  )
} 
# Infer sources
data_source_network <- infer_supporting_datasources(signaling_graph_list = active_signaling_network,
                                                    lr_network = lr_network, sig_network = sig_network, gr_network = gr_network)


# Get annotation table
# Node annotation table
specific_annotation_tbl <- bind_rows(
  tibble(gene = ligands_oi, annotation = "ligand"),
  tibble(gene = targets_oi, annotation = "target"),
  tibble(gene = c(data_source_network$from, data_source_network$to) %>% unique() %>% setdiff(c(targets_oi,ligands_oi)) %>% intersect(lr_network$to %>% unique()), annotation = "receptor"),
  tibble(gene = c(data_source_network$from, data_source_network$to) %>% unique() %>% setdiff(c(targets_oi,ligands_oi)) %>% intersect(gr_network$from %>% unique()) %>% setdiff(c(data_source_network$from, data_source_network$to) %>% unique() %>% intersect(lr_network$to %>% unique())),annotation = "transcriptional regulator")
)
non_specific_annotation_tbl <- tibble(gene = c(data_source_network$from, data_source_network$to) %>% unique() %>% setdiff(specific_annotation_tbl$gene), annotation = "signaling mediator")


# Save outputs

# weighted networks 
bind_rows(
  active_signaling_network$sig %>% mutate(layer = "signaling"),
  active_signaling_network$gr %>% mutate(layer = "regulatory")) %>%
  write_tsv(
      file.path(output_path,paste0(sample_name,"_weighted_signaling_network.tsv"))
    )

# networks with information of supporting data sources 
data_source_network %>% 
  write_tsv(
    file.path(output_path,paste0(sample_name,"_data_source_network.tsv"))
  )

# Save annotation
bind_rows(specific_annotation_tbl, non_specific_annotation_tbl) %>%
  write_tsv(
    file.path(output_path,paste0(sample_name,"_annotation_table.tsv"))
  )