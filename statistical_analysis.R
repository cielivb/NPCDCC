# Statistical Analysis of Neurotransmitter Probabilities (Dirichlet Regression)

################################# SET UP #######################################

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv") 
}
if (!renv::is_activated()) {
  renv::activate()
  renv::restore()
}

library(DirichletReg)
library(arrow) # parquet dataset handling
library(parallel) # detectCores() for automatic CPU core count detection
library(simr) # powerSim() for simulation-based power analysis
library(future) # plan(multisession) for parallel execution
library(future.apply) # future_lapply() for parallel apply across models

# Detect number of cores (used for parallel power simulation). Try to obtain 
# number of physical cores, and fall back to logical core count if that fails.
physical_cores <- tryCatch(detectCores(logical = FALSE), error = function(e) NA)
if (is.na(physical_cores) || physical_cores < 1) {
  cores <- detectCores(logical = TRUE)
} else {
  cores <- physical_cores
}

# Process arguments
args <- commandArgs(trailingOnly = TRUE)
result_dir <- args[1]


############################ FIT MODELS + ANOVA ################################

# Load a sample of neurotransmitter probabilities from file
data_dir = "results/temp.parquet"
rdf <- as.data.frame(arrow::open_dataset(data_dir)) # Load
unlink(data_dir, recursive = TRUE) # Remove temp data

# Format response variable (required for Dirichlet regression)
rdf$resp <- DirichletReg::DR_data(rdf[, c("gaba", "ach", "other")])

# Fit Dirichlet models
null <- DirichletReg::DirichReg(resp ~ 1, data = rdf)
mod1 <- DirichletReg::DirichReg(resp ~ region, data = rdf)
mod2 <- DirichletReg::DirichReg(resp ~ neuropil, data = rdf)
mod3 <- DirichletReg::DirichReg(resp ~ neuropil_size, data = rdf)
mod4 <- DirichletReg::DirichReg(resp ~ region + neuropil_size, data = rdf)
null_info <- capture.output(summary(null))
mod1_info <- capture.output(summary(mod1))
mod2_info <- capture.output(summary(mod2))
mod3_info <- capture.output(summary(mod3))
mod4_info <- capture.output(summary(mod4))

# Do ANOVA and add adjust for multiple testing
anova_res <- anova(null, mod1, mod2, mod3, mod4)
pvals <- anova_res[,"Pr(>Chi)"]
pvals_adj <- p.adjust(pvals, method = "BH")
anova_table <- data.frame(
  Model = rownames(anova_res),
  ChiSq = anova_res[,"Chisq"],
  df    = anova_res[,"Df"],
  pval  = pvals,
  pval_adj = pvals_adj
)


######################### PARALLEL POWER SIMULATION ############################

# The Dirichlet distribution is non-parametric, hence must use simulations to
# estimate power. powerSim is single-threaded, so will use multiprocessing and
# batching to run multiple simulations in parallel while keeping RAM in check.

run_power_batches <- function(model, total_sims = 50, batch_size = 10) {
  sims_done <- 0
  detections <- 0
  while (sims_done < total_sims) {
    sims_to_run <- min(batch_size, total_sims - sims_done)
    result <- powerSim(model, nsim = sims_to_run, test = "anova", progress = FALSE)
    detections <- detections + result$x  # + num rejections
    sims_done <- sims_done + sims_to_run
    gc() # free memory
  }
  power_estimate <- detections / total_sims
  return(list(power = power_estimate, detections = detections, num_sims = total_sims))
}

plan(multisession, workers = min(cores, 4))

results <- future_lapply( # Run run_power_batches across all models in parallel
  list(as(mod1, "glm"), as(mod1, "glm"), as(mod1, "glm"), as(mod1, "glm")),
  run_power_batches, total_sims = 100, batch_size = 20
)
names(results) <- c("mod1", "mod2", "mod3", "mod4")

format_power_result <- function(name, res) {
  paste0(
    "\nPOWER SUMMARY - ", name, ":\n",
    "Total simulations : ", res$sims, "\n",
    "Detections : ", res$detections, "\n",
    "Estimated power : ", round(res$power, 3), "\n"
  )
}
mod1_pow_info <- format_power_result("mod1", results["mod1"])
mod2_pow_info <- format_power_result("mod2", results["mod2"])
mod3_pow_info <- format_power_result("mod3", results["mod3"])
mod4_pow_info <- format_power_result("mod4", results["mod4"])


############################# REPORT RESULTS ###################################

model_file <- file.path(result_dir, "models.txt")
anova_file <- file.path(result_dir, "anova.txt")
power_file <- file.path(result_dir, "power.txt")
writeLines(c("NULL MODEL SUMMARY", null_info,
             "\nMODEL 1 SUMMARY", mod1_info,
             "\nMODEL 2 SUMMARY", mod2_info,
             "\nMODEL 3 SUMMARY", mod3_info,
             "\nMODEL 4 SUMMARY", mod4_info),
           model_file)
writeLines(c("ANOVA SUMMARY", anova_info), anova_file)
writeLines(c(mod1_pow_info, mod2_pow_info, mod3_pow_info, mod4_pow_info), 
           power_file)

# Clear environment
rm(list = ls())
gc()
renv::snapshot()
