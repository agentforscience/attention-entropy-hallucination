## Data

| model | kept | abstain | domain: n / halluc rate |
|---|---|---|---|
| Qwen2.5-7B | 7470 | 30 | biography 1490 / 0.77; geography 1494 / 0.38; creative_works 1488 / 0.94; science 1498 / 0.23; trivia 1500 / 0.42 |
<!-- judge-strmatch agreement 0.925 {'biography': 0.9550335570469799, 'geography': 0.965194109772423, 'creative_works': 0.9952956989247311, 'science': 0.7563417890520694, 'trivia': 0.9546666666666667} ; prompt/answer len {'biography': (54.9, 3.0), 'geography': (53.6, 2.5), 'creative_works': (53.7, 4.0), 'science': (60.5, 2.8), 'trivia': (64.2, 3.1)} -->
| Llama-3.1-8B | 7450 | 50 | biography 1477 / 0.64; geography 1491 / 0.33; creative_works 1483 / 0.80; science 1499 / 0.27; trivia 1500 / 0.25 |
<!-- judge-strmatch agreement 0.921 {'biography': 0.950575490859851, 'geography': 0.9550637156270959, 'creative_works': 0.983142279163857, 'science': 0.7758505670446965, 'trivia': 0.94} ; prompt/answer len {'biography': (76.9, 4.2), 'geography': (75.5, 3.4), 'creative_works': (75.6, 4.4), 'science': (82.5, 4.0), 'trivia': (85.5, 4.1)} -->

## E1 final-3-layer entropy, hallucinated vs factual (relative diff % [95% CI]; Cohen's d)


**mean over answer tokens**

| model | pooled | biography | geography | creative_works | science | trivia | macro |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | -18.4% [-19.4, -17.3]; d=-0.70 | -0.8% [-3.5, +1.8]; d=-0.04 | +2.1% [-0.7, +4.9]; d=+0.08 | -6.0% [-11.5, -0.6]; d=-0.30 | -2.9% [-5.4, -0.4]; d=-0.14 | -9.1% [-11.7, -6.7]; d=-0.36 | -3.3% |
| Llama-3.1-8B | -14.7% [-15.4, -14.0]; d=-0.91 | -3.7% [-5.0, -2.4]; d=-0.30 | -4.5% [-5.9, -3.0]; d=-0.33 | -14.1% [-16.3, -11.9]; d=-0.77 | -4.5% [-5.6, -3.4]; d=-0.44 | -9.8% [-11.5, -8.0]; d=-0.71 | -7.3% |

**first answer token**

| model | pooled | biography | geography | creative_works | science | trivia | macro |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | -11.7% [-12.6, -10.8]; d=-0.56 | -0.5% [-2.6, +1.6]; d=-0.03 | +4.5% [+2.5, +6.6]; d=+0.22 | +7.9% [+2.7, +13.1]; d=+0.41 | -0.8% [-2.7, +1.1]; d=-0.05 | -5.2% [-7.4, -3.1]; d=-0.24 | +1.2% |
| Llama-3.1-8B | -7.1% [-7.6, -6.6]; d=-0.59 | -1.8% [-2.7, -0.8]; d=-0.19 | -0.3% [-1.4, +0.8]; d=-0.03 | +2.7% [+1.3, +4.1]; d=+0.24 | -2.3% [-3.4, -1.2]; d=-0.23 | -6.2% [-7.7, -4.7]; d=-0.49 | -1.6% |

**length-normalised, answer mean**

| model | pooled | biography | geography | creative_works | science | trivia | macro |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | -17.6% [-18.7, -16.5]; d=-0.66 | -0.7% [-3.3, +2.1]; d=-0.03 | +2.6% [-0.2, +5.5]; d=+0.10 | -5.7% [-11.5, -0.2]; d=-0.29 | -2.8% [-5.2, -0.4]; d=-0.13 | -9.5% [-12.0, -6.9]; d=-0.37 | -3.2% |
| Llama-3.1-8B | -14.1% [-14.8, -13.4]; d=-0.88 | -3.7% [-5.0, -2.5]; d=-0.30 | -4.2% [-5.6, -2.7]; d=-0.30 | -13.8% [-15.9, -11.7]; d=-0.76 | -4.3% [-5.4, -3.2]; d=-0.42 | -9.8% [-11.6, -8.1]; d=-0.72 | -7.2% |

| model | pooled means (halluc / factual, nats) | OLS domain FE | OLS domain FE + length | layers sig. (FDR) |
|---|---|---|---|---|
| Qwen2.5-7B | 0.851 / 1.043 | -2.8% [-4.1, -1.5], p=1.9e-05 | -2.1% [-3.2, -0.9], p=5.4e-04 | 28/28 |
| Llama-3.1-8B | 0.995 / 1.166 | -6.5% [-7.1, -5.8], p=8.9e-76 | -6.4% [-7.1, -5.7], p=4.7e-74 | 30/32 |

Label robustness (ent_ans, pooled / macro):
- Qwen2.5-7B: hybrid: -18.4% [-19.4, -17.3] / -3.3%; judge: -17.9% [-19.0, -16.9] / -3.0%; strmatch: -15.3% [-16.4, -14.2] / -2.4%
- Llama-3.1-8B: hybrid: -14.7% [-15.4, -14.0] / -7.3%; judge: -14.2% [-14.9, -13.5] / -6.7%; strmatch: -12.7% [-13.4, -12.0] / -6.9%

## E2/E4 pooled classifiers (AUROC [95% CI] / within-domain macro AUROC of the same scores / AUPRC)

| features | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| domain_only | 0.785 [0.774, 0.796] / 0.429 / 0.817 | 0.723 [0.711, 0.736] / 0.423 / 0.682 |
| length_only | 0.669 [0.657, 0.681] / 0.527 / 0.685 | 0.652 [0.639, 0.664] / 0.535 / 0.585 |
| entropy_layers + domain | 0.843 [0.834, 0.852] / 0.725 / 0.882 | 0.827 [0.818, 0.836] / 0.745 / 0.805 |
| entropy_last3_mean (1 feat) | 0.700 [0.688, 0.711] / 0.539 / 0.754 | 0.731 [0.720, 0.742] / 0.638 / 0.717 |
| entropy_layers (ans mean) | 0.827 [0.817, 0.836] / 0.719 / 0.870 | 0.817 [0.807, 0.826] / 0.740 / 0.792 |
| entropy_layers (first token) | 0.826 [0.816, 0.835] / 0.683 / 0.868 | 0.821 [0.812, 0.830] / 0.743 / 0.812 |
| norm_entropy_layers (ans mean) | 0.826 [0.817, 0.835] / 0.718 / 0.868 | 0.815 [0.806, 0.825] / 0.740 / 0.790 |
| entropy_layers + length | 0.833 [0.824, 0.842] / 0.727 / 0.874 | 0.822 [0.813, 0.831] / 0.747 / 0.801 |
| logit_baseline | 0.876 [0.868, 0.884] / 0.812 / 0.902 | 0.878 [0.870, 0.886] / 0.847 / 0.864 |
| nll_mean only | 0.859 [0.850, 0.867] / 0.806 / 0.884 | 0.853 [0.845, 0.862] / 0.821 / 0.835 |
| entropy_layers + logit | 0.891 [0.884, 0.899] / 0.823 / 0.916 | 0.890 [0.883, 0.898] / 0.854 / 0.881 |
| entropy_heads (L*H) | 0.890 [0.883, 0.898] / 0.821 / 0.914 | 0.874 [0.867, 0.882] / 0.830 / 0.856 |

Seed sensitivity (SD of AUROC over 5 CV repeats), entropy_layers: {'qwen25_7b': 0.0009, 'llama31_8b': 0.0006}

Paired differences:
- Qwen2.5-7B: entropy_layers - length_only: +0.158 [0.145, 0.170], p=0.0
- Qwen2.5-7B: entropy_layers - domain_only: +0.041 [0.032, 0.051], p=0.0
- Qwen2.5-7B: entropy_layers - logit_baseline: -0.049 [-0.058, -0.040], p=0.0
- Qwen2.5-7B: entropy+logit - logit_baseline: +0.015 [0.010, 0.020], p=0.0
- Llama-3.1-8B: entropy_layers - length_only: +0.165 [0.152, 0.178], p=0.0
- Llama-3.1-8B: entropy_layers - domain_only: +0.093 [0.082, 0.105], p=0.0
- Llama-3.1-8B: entropy_layers - logit_baseline: -0.062 [-0.070, -0.053], p=0.0
- Llama-3.1-8B: entropy+logit - logit_baseline: +0.012 [0.009, 0.016], p=0.0

Label robustness, entropy_layers (ans mean) pooled / within-domain macro:
- Qwen2.5-7B: hybrid: 0.827 / 0.719; judge: 0.822 / 0.715; strmatch: 0.793 / 0.716
- Llama-3.1-8B: hybrid: 0.817 / 0.740; judge: 0.809 / 0.728; strmatch: 0.792 / 0.736

## E3 within-domain AUROC (trained and cross-validated inside each domain)


**Qwen2.5-7B**

| features | biography | geography | creative_works | science | trivia | mean |
|---|---|---|---|---|---|---|
| entropy_layers (ans mean) | 0.747 [0.716, 0.775] | 0.802 [0.777, 0.824] | 0.892 [0.857, 0.922] | 0.616 [0.581, 0.648] | 0.711 [0.684, 0.736] | 0.753 |
| entropy_layers (first token) | 0.739 [0.709, 0.769] | 0.777 [0.752, 0.801] | 0.914 [0.888, 0.937] | 0.564 [0.530, 0.599] | 0.725 [0.700, 0.750] | 0.744 |
| norm_entropy_layers (ans mean) | 0.745 [0.715, 0.775] | 0.808 [0.785, 0.832] | 0.899 [0.866, 0.926] | 0.620 [0.586, 0.653] | 0.713 [0.686, 0.739] | 0.757 |
| length_only | 0.507 [0.473, 0.541] | 0.599 [0.568, 0.629] | 0.552 [0.490, 0.613] | 0.546 [0.513, 0.580] | 0.593 [0.565, 0.621] | 0.559 |
| logit_baseline | 0.781 [0.754, 0.809] | 0.800 [0.776, 0.825] | 0.911 [0.879, 0.939] | 0.728 [0.697, 0.757] | 0.853 [0.834, 0.871] | 0.815 |
| entropy_layers + logit | 0.810 [0.784, 0.834] | 0.860 [0.842, 0.879] | 0.932 [0.909, 0.954] | 0.730 [0.699, 0.759] | 0.839 [0.819, 0.859] | 0.834 |

ranking: creative_works > geography > biography > trivia > science; bio - science = +0.131 [0.083, 0.176], p=0.0; P(bio highest)=0.000, P(science lowest)=1.000; transfer off-diagonal mean 0.651
  [judge] ranking: creative_works > geography > biography > trivia > science; biography 0.739, geography 0.799, creative_works 0.893, science 0.627, trivia 0.705
  [strmatch] ranking: creative_works > geography > biography > trivia > science; biography 0.729, geography 0.792, creative_works 0.891, science 0.650, trivia 0.695

transfer (rows train, cols test):

|                |   biography |   geography |   creative_works |   science |   trivia |
|:---------------|------------:|------------:|-----------------:|----------:|---------:|
| biography      |       0.747 |       0.568 |            0.883 |     0.566 |    0.644 |
| geography      |       0.583 |       0.802 |            0.851 |     0.593 |    0.613 |
| creative_works |       0.731 |       0.629 |            0.892 |     0.515 |    0.632 |
| science        |       0.569 |       0.587 |            0.727 |     0.616 |    0.683 |
| trivia         |       0.644 |       0.564 |            0.838 |     0.596 |    0.711 |

by PopQA relation:

| relation       | domain         |   n |   halluc_rate |   auroc_entropy |   auroc_logit |
|:---------------|:---------------|----:|--------------:|----------------:|--------------:|
| author         | creative_works | 286 |         0.881 |           0.905 |         0.892 |
| capital        | geography      | 514 |         0.239 |           0.851 |         0.893 |
| capital of     | geography      | 290 |         0.562 |           0.684 |         0.792 |
| country        | geography      | 690 |         0.4   |           0.671 |         0.774 |
| father         | biography      | 435 |         0.724 |           0.656 |         0.821 |
| mother         | biography      | 145 |         0.745 |           0.726 |         0.791 |
| occupation     | biography      | 436 |         0.718 |           0.746 |         0.764 |
| place of birth | biography      | 474 |         0.869 |           0.666 |         0.793 |
| screenwriter   | creative_works | 361 |         0.945 |           0.876 |         0.868 |

**Llama-3.1-8B**

| features | biography | geography | creative_works | science | trivia | mean |
|---|---|---|---|---|---|---|
| entropy_layers (ans mean) | 0.742 [0.713, 0.768] | 0.831 [0.808, 0.851] | 0.820 [0.792, 0.846] | 0.694 [0.664, 0.722] | 0.741 [0.711, 0.769] | 0.765 |
| entropy_layers (first token) | 0.750 [0.724, 0.777] | 0.845 [0.825, 0.865] | 0.854 [0.831, 0.877] | 0.640 [0.609, 0.669] | 0.741 [0.712, 0.770] | 0.766 |
| norm_entropy_layers (ans mean) | 0.742 [0.715, 0.769] | 0.832 [0.809, 0.853] | 0.819 [0.792, 0.846] | 0.693 [0.663, 0.723] | 0.741 [0.709, 0.772] | 0.765 |
| length_only | 0.423 [0.392, 0.452] | 0.655 [0.626, 0.686] | 0.633 [0.597, 0.669] | 0.584 [0.553, 0.618] | 0.491 [0.457, 0.524] | 0.557 |
| logit_baseline | 0.827 [0.804, 0.849] | 0.853 [0.832, 0.873] | 0.885 [0.865, 0.904] | 0.792 [0.767, 0.816] | 0.879 [0.858, 0.898] | 0.847 |
| entropy_layers + logit | 0.845 [0.824, 0.864] | 0.877 [0.857, 0.895] | 0.896 [0.875, 0.914] | 0.813 [0.789, 0.837] | 0.883 [0.862, 0.902] | 0.863 |

ranking: geography > creative_works > biography > trivia > science; bio - science = +0.048 [0.009, 0.088], p=0.01; P(bio highest)=0.000, P(science lowest)=0.984; transfer off-diagonal mean 0.682
  [judge] ranking: geography > creative_works > trivia > biography > science; biography 0.730, geography 0.820, creative_works 0.812, science 0.691, trivia 0.730
  [strmatch] ranking: geography > creative_works > trivia > biography > science; biography 0.727, geography 0.832, creative_works 0.822, science 0.698, trivia 0.735

transfer (rows train, cols test):

|                |   biography |   geography |   creative_works |   science |   trivia |
|:---------------|------------:|------------:|-----------------:|----------:|---------:|
| biography      |       0.742 |       0.782 |            0.778 |     0.596 |    0.705 |
| geography      |       0.655 |       0.831 |            0.752 |     0.636 |    0.674 |
| creative_works |       0.58  |       0.716 |            0.82  |     0.541 |    0.667 |
| science        |       0.537 |       0.746 |            0.734 |     0.694 |    0.715 |
| trivia         |       0.668 |       0.75  |            0.76  |     0.654 |    0.741 |

by PopQA relation:

| relation       | domain         |   n |   halluc_rate |   auroc_entropy |   auroc_logit |
|:---------------|:---------------|----:|--------------:|----------------:|--------------:|
| author         | creative_works | 288 |         0.736 |           0.795 |         0.89  |
| capital        | geography      | 509 |         0.163 |           0.843 |         0.912 |
| capital of     | geography      | 291 |         0.471 |           0.667 |         0.757 |
| composer       | creative_works | 185 |         0.805 |           0.749 |         0.904 |
| country        | geography      | 691 |         0.392 |           0.812 |         0.836 |
| director       | creative_works | 363 |         0.774 |           0.781 |         0.858 |
| father         | biography      | 437 |         0.558 |           0.761 |         0.863 |
| mother         | biography      | 145 |         0.641 |           0.722 |         0.903 |
| occupation     | biography      | 435 |         0.556 |           0.675 |         0.772 |
| place of birth | biography      | 460 |         0.793 |           0.684 |         0.798 |
| producer       | creative_works | 289 |         0.903 |           0.765 |         0.896 |
| screenwriter   | creative_works | 358 |         0.799 |           0.785 |         0.885 |
