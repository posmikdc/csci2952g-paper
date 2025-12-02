################################################################################
### Title: Generate low-dimensional Protein MDS Representation
### Author: Daniel Posmik (daniel_posmik@brown.edu)
################################################################################

# Setup ========================================================================
# Environment
#install.packages(c("bio3d", "smacof", "rbioapi"))
library(bio3d) 
library(smacof) 
library(readr)
library(tidyverse)
library(rbioapi)
library(httr)
library(jsonlite)

#setwd("/Users/posmikdc/Documents/brown/classes/year2/fall25/csci2952g-dlgenomics/csci2952g-paper/code/proteins-mds")

# Get list of available PDB files
pdb_files <- list.files("../../data/prot-structure/prot-coords", 
                        pattern = "\\.pdb$", 
                        full.names = TRUE)
prot_names <- sub("\\.pdb$", "", basename(pdb_files))

# Initialize MDS stress log
mds_log <- data.frame(
  protein = character(),
  stress = numeric(),
  status = character(),
  stringsAsFactors = FALSE
)

# MDS parameters
k <- 2  # Target dimension

# Process each protein =========================================================
# Warning: This will take a while (~20min)
for(i in seq_along(pdb_files)) {
  prot_name <- prot_names[i]
  cat("Processing:", prot_name, "(", i, "/", length(pdb_files), ")\n")
  
  result <- tryCatch({
    # Read PDB file
    pdb <- read.pdb(pdb_files[i])
    
    # Extract C-alpha coordinates
    ca_coords <- matrix(pdb$xyz[pdb$calpha], ncol = 3, byrow = TRUE)
    
    # Calculate distance matrix
    dist_mat <- as.matrix(dist(ca_coords))
    
    # Perform MDS
    mds_result <- smacofSym(dist_mat, ndim = k, type = "ratio")
    
    # Extract low-dimensional coordinates
    low_dim_coords <- mds_result$conf
    
    # Save coordinates to file
    output_file <- paste0("../../output/proteins-mds/prots-mds/", prot_name, ".txt")
    write.table(low_dim_coords, output_file, row.names = FALSE, col.names = FALSE)
    
    # Log success
    data.frame(protein = prot_name, 
               stress = mds_result$stress, 
               status = "success")
    
  }, error = function(e) {
    cat("  Failed:", prot_name, "-", e$message, "\n")
    data.frame(protein = prot_name, 
               stress = NA, 
               status = paste0("failed: ", e$message))
  })
  
  mds_log <- rbind(mds_log, result)
}

# Save MDS stress log
write_csv(mds_log, "../../output/proteins-mds/prots-mds/_mds_stress_log.csv")

# Print summary
cat("\nMDS Summary:\n")
cat("Successful:", sum(mds_log$status == "success"), "\n")
cat("Failed:", sum(mds_log$status != "success"), "\n")
cat("Mean stress:", mean(mds_log$stress, na.rm = TRUE), "\n")
cat("Median stress:", median(mds_log$stress, na.rm = TRUE), "\n")
