# Release attribution

## Model — Apache License 2.0
LingBot-Depth by the LingBot-Depth authors (Robbyant).
Source: https://github.com/robbyant/lingbot-depth
Checkpoint: https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5
Revision: 79204ed6b837f4fdd192cf563e59481fecfa0295
Source commit: f3a237e434ae987bc38281476d6cfb5df3e4d739
Changes: fixed-shape Core ML graph, invalid-depth masking, convolution-only FP16.
No retraining. Apache-2.0 license text accompanies the model asset.

## Evaluation data — CC BY 4.0
iBims-1 / iBims-v1 (2019), Tobias Koch, Lukas Liebel, Friedrich Fraundorfer,
Marco Koerner, Technical University of Munich.
Source: https://mediatum.ub.tum.de/1455541
License: https://creativecommons.org/licenses/by/4.0/
Changes: RGB resized to 560x420; GT-derived sparse-500 / lowres-8x input depth;
original-resolution GT and validity masks preserved. Model predictions added.
Three deliberately selected worst cases, not a representative random subset.
No author device capture, serial, UDID, installed-app list, or private log is bundled.
No endorsement by the original model or dataset authors is implied.
