# Multimodal Identification of Informative Social Media Posts for Disaster Response

Identifies disaster tweets (text + image) as being one of the two classes: informative or not informative
Compares text only, image only and fusion models, using the CrisisMMD v2.0 dataset.

**Research question:** Does multimodal fusion improve informativeness classification
compared with text-only and image-only models?

## Results (test macro-F1, mean ± std over 3 seeds)
| Model | Macro-F1 |
|---|---|
| Text-only (DistilBERT) | 0.718 ± 0.008 |
| Image-only (EfficientNet-B0) | 0.727 ± 0.005 |
| Early fusion | 0.763 ± 0.002 |
| Gated fusion | 0.765 ± 0.002 |
| **Late fusion** | **0.769 ± 0.003** |

Late fusion vs text-only: McNemar p = 0.0007.

## How to reproduce
1. Open `CrisisMMD_submission.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Select **Runtime → Run all** (about 30 minutes; the dataset downloads automatically).
4. Results are saved to `outputs/figures` and `outputs/tables`.

## Files
- `CrisisMMD_submission.ipynb`: full pipeline (data quality, preprocessing, features, models, evaluation)
- `outputs/figures`: all figures used in the report
- `outputs/tables`: all result tables (CSV)

## Requirements
Python 3, PyTorch, torchvision, transformers, datasets, scikit-learn, pandas, matplotlib
(all available in Google Colab). Random seeds: 42, 43, 44.

## Data and ethics
CrisisMMD v2.0 (QCRI), licence CC BY-NC-SA 4.0, used for non-commercial research only.
The dataset is not included in this repository. Usernames are anonymised and no images
of people are displayed.

## References
- Alam, F., Ofli, F., & Imran, M. (2018). CrisisMMD: Multimodal Twitter datasets from natural disasters. ICWSM.
- Sanh, V. et al. (2019). DistilBERT. arXiv:1910.01108.
- Tan, M., & Le, Q. (2019). EfficientNet. ICML.
- Arevalo, J. et al. (2017). Gated multimodal units for information fusion. ICLR Workshop.

Author: Shijin Mathew