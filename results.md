# Results

Attention U-Net on the DRIVE test split (20 images). Protocol: full-image reflect-pad tile-and-stitch, patch 128, stride 64, threshold 0.5; metrics over all pixels (no FOV masking).

Weights: `attention_unet.weights.pt`.

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.9617 | 0.0032 |
| IoU | 0.6337 | 0.0286 |
| F1 (Dice) | 0.7754 | 0.0215 |

Per-image numbers: [Figures/metrics_per_image.csv](Figures/metrics_per_image.csv).

_Original TF/Keras run reported 0.9639 / 0.6383 / 0.7787 (acc/IoU/F1). The original was unseeded and its checkpoint was lost; these retrained numbers are the committed truth and may drift from the originals._

## Thin-vessel analysis

Recall / precision / F1 pooled over the 20 test images, stratified by ground-truth vessel width. False positives are attributed to the nearest vessel's width bin.

| Width (px) | GT px | Recall | Precision | F1 |
|---|---|---|---|---|
| [1,2) | 172710 | 0.5900 | 0.5804 | 0.5851 |
| [2,4) | 186897 | 0.8206 | 0.8480 | 0.8341 |
| [4,6) | 148072 | 0.8436 | 0.9227 | 0.8814 |
| [6,inf) | 70266 | 0.8359 | 0.9648 | 0.8957 |

![Recall vs vessel width](Figures/thin_vessel_recall.png)

Recall rises steeply from the thinnest vessels (0.590) to the mid widths (0.844) and then plateaus; precision and F1 increase monotonically with width. On the widest vessels the remaining false negatives are concentrated at vessel *boundaries*, not centrelines (55.8% of false negatives lie in the outer third of the vessel versus 12.6% of true positives; mean centre-ness 0.480 vs 0.733). The residual error on thick vessels is therefore ordinary boundary under-segmentation rather than the central-reflex effect that wide retinal vessels might suggest.

