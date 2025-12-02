################################################################################
### Title: Download AlphaFold Protein Structures
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

# Read in crosswalk
prot_xwalk <- read_csv("../../output/proteins-mds/prot_crosswalk.csv")

# Obtain 3D protein files ======================================================
# Get all non-NA accessions
all_prots <- prot_xwalk$uniprot_accession[!is.na(prot_xwalk$uniprot_accession)]

# Initialize download log
download_log <- data.frame(
  uniprot_accession = character(),
  status = character(),
  url = character(),
  stringsAsFactors = FALSE
)

# Download AlphaFold PDB files (v6) (Warning: This will take a while)
for(acc in all_prots) {
  url <- paste0("https://alphafold.ebi.ac.uk/files/AF-", acc, "-F1-model_v6.pdb")
  destfile <- paste0("../../data/prot-structure/prot-coords/", acc, ".pdb")
  
  result <- tryCatch({
    download.file(url, destfile, quiet = TRUE)
    cat("Downloaded:", acc, "\n")
    "success"
  }, error = function(e) {
    cat("Failed:", acc, "\n")
    "failed"
  })
  
  # Log result
  download_log <- rbind(download_log, 
                        data.frame(uniprot_accession = acc,
                                   status = result,
                                   url = url))
}

# Save download log
write_csv(download_log, "../../data/prot-structure/prot-coords/_pdb_download_log.csv")

# Print diagnostics ============================================================
# Print summary
cat("\nDownload Summary:\n")
cat("Successful:", sum(download_log$status == "success"), "\n")
cat("Failed:", sum(download_log$status == "failed"), "\n")