# v5: leak-free re-evaluation

Run from the repo root (data/ and ResNet_v4_large_15_0.906.h5 present):

    python dupcheck.py .                                   # -> dups.json (train/test near-duplicates)
    python features.py . feat.npz 4                        # frozen ResNet50 features, original + 4 augmented views
    python train_eval.py . feat.npz dups.json out/         # 5-fold group CV selection, single test evaluation
    python verify_v5.py . out/ResNet_v5_cv_26_0.898.keras out/results_v5.json

Findings
- 24 of 660 test images repeat a training lesion (perceptual hash <= 4, same label, visually confirmed;
  benign/1, 1074, 1473, 216 were look-alikes and kept). They are excluded from the clean test set (n = 636).
- 5-fold StratifiedGroupKFold on training data only (near-duplicate lesions kept in one fold) chose
  lr 0.001, dropout 0.2, 26 epochs. CV accuracy 89.0% +/- 1.5.
- Test (used once), n = 636: accuracy 89.8% (95% CI 87.4-92.0), sensitivity 92.6%, specificity 87.6%,
  PPV 85.6%, ROC AUC 0.957. Confusion matrix [[310, 44], [21, 261]].
- The README's v4 settings were wrong: v4 used dropout 0.2 and shear/zoom/horizontal-flip augmentation.
