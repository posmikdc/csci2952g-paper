################################################################################
### Title: String to Uniprot Protein ID Crosswalk
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

# Please set your working directory to this folder
#setwd("/Users/posmikdc/Documents/brown/classes/year2/fall25/csci2952g-dlgenomics/csci2952g-paper/code/proteins-mds")

# Read in file names
prot_dict <- 
  read_tsv("../../data/gao_shs27k_data/protein.SHS27k.sequences.dictionary.pro3.tsv", 
           col_names = c("string_id", "seq"))

# STRING to Gene Name Mapping ==================================================
# Remove species prefix (9606.) for STRING API
proteins <- sub("^9606\\.", "", prot_dict$string_id)

# Query STRING API to get gene names and Ensembl protein IDs
url <- "https://string-db.org/api/json/get_string_ids"
response <- POST(url, 
                 body = list(identifiers = paste(proteins, collapse = "\n"),
                             species = 9606,
                             limit = 1))

# Parse STRING API results
prot_xwalk <- content(response, "text") %>% fromJSON()

# UniProt ID Mapping ===========================================================
# Download UniProt idmapping file for human proteins
url <- "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping_selected.tab.gz"
download.file(url, "../../data/prot-structure/HUMAN_9606_idmapping.tab.gz")

# Read idmapping file and add column names
id_mapping <- read.delim("../../data/prot-structure/HUMAN_9606_idmapping.tab.gz", 
                         header = FALSE)
colnames(id_mapping) <- c("UniProtKB_AC", "UniProtKB_ID", "GeneID", "RefSeq", 
                          "GI", "PDB", "GO", "UniRef100", "UniRef90", "UniRef50",
                          "UniParc", "PIR", "NCBI_taxon", "MIM", "UniGene", 
                          "PubMed", "EMBL", "EMBL_CDS", "Ensembl", "Ensembl_TRS",
                          "Ensembl_PRO", "Additional_PubMed")

# Clean idmapping: split multiple ENSP IDs and remove version numbers
id_mapping_clean <- id_mapping %>%
  select(UniProtKB_AC, Ensembl_PRO) %>%
  filter(Ensembl_PRO != "") %>%
  separate_rows(Ensembl_PRO, sep = "; ") %>%
  mutate(Ensembl_PRO = sub("\\..*", "", Ensembl_PRO))

# Join STRING results with UniProt accessions via Ensembl protein IDs
prot_xwalk <- prot_xwalk %>%
  left_join(
    id_mapping_clean,
    by = c("queryItem" = "Ensembl_PRO")
  ) %>%
  rename(uniprot_accession = UniProtKB_AC)

# Save Final Crosswalk =========================================================
write_csv(prot_xwalk, "../../output/proteins-mds/prot_crosswalk.csv")
