# v5: leak-free re-evaluation

Run from the repository root (needs `data/` and `ResNet_v4_large_15_0.906.h5`, whose frozen ResNet50 base is reused):

    python v5_retrain/dupcheck.py .                                        # -> dups.json (train/test near-duplicates)
    python v5_retrain/features.py . feat.npz 4                             # frozen ResNet50 features, original + 4 augmented views
    python v5_retrain/train_eval.py . feat.npz dups.json out/              # 5-fold group CV selection, single test evaluation
    python v5_retrain/verify_v5.py . out/ResNet_v5_cv_26_0.898.keras out/results_v5.json   # re-check on raw images
    python v5_retrain/cv_curve.py . feat.npz dups.json out/                # per-fold CV curve for the chosen settings
    python v5_retrain/plot_cv.py out/cv_curve_lr0.001_d0.2.json cv_curve.png

## Findings

- 24 of 660 test images repeat a training lesion (perceptual hash distance ≤ 4, same label, each pair checked by eye;
  benign/1, 1074, 1473 and 216 were look-alikes of different lesions and were kept). They are excluded from the
  clean test set (n = 636: 354 benign, 282 malignant).
- 5-fold StratifiedGroupKFold on training data only, with near-duplicate lesions kept in one fold, chose
  learning rate 0.001, dropout 0.2, 26 epochs. CV accuracy 89.0% ± 1.5.
- Test set (used once), n = 636: accuracy 89.8% (95% CI 87.4–92.0), sensitivity 92.6%, specificity 87.6%,
  PPV 85.6%, ROC AUC 0.957. Confusion matrix [[310, 44], [21, 261]].
- The original README's v4 settings were wrong: v4 used dropout 0.2 and shear / zoom / horizontal-flip augmentation.

## Files

| File | Purpose |
|---|---|
| `dupcheck.py` | Exact and near-duplicate search across train and test |
| `features.py` | ResNet50 feature extraction with augmented training views |
| `train_eval.py` | Cross-validated model selection, final training, single test evaluation |
| `verify_v5.py` | Re-scores the saved model on raw test images |
| `cv_curve.py`, `plot_cv.py` | Cross-validation accuracy figure (`cv_curve.png`) |
| `results_v5.json` | All reported metrics, the chosen settings and the excluded test images |
| `cv_curve_lr0.001_d0.2.json` | Per-fold validation accuracy by epoch |
