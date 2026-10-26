"""Phase 3 scaffold: 4 attribution methods x 3 models = 12 per sequence.
Implement each method against the trained checkpoints."""
# occlusion: slide 50bp window, stride 25, replace with baseline, measure delta
# kernelshap: shap.KernelExplainer, 100-seq background, 500 samples
# permutation: permute 100bp windows, measure delta
# integrated_gradients: captum.attr.IntegratedGradients, 32 steps
# consensus: rank product at 25bp resolution
