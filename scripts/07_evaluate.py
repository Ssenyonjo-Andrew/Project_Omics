"""Phase 4 scaffold: stability, fidelity, plausibility."""
# stability: 1000 bootstrap resamples, Spearman, 95% interval
# fidelity: deletion/insertion at 1/5/10/20% + retrain-without-top
# plausibility: hypergeometric enrichment vs annotation layer
# controls: shuffled labels, random window, randomised model
