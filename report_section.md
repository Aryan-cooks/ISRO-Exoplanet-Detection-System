# Technical Report: Exoplanet Detection Pipeline

## Blend Detection

To address false positives caused by background eclipsing binaries or nearby contaminating stars, we implemented a robust blend detection mechanism. This relies on querying the TESS Input Catalog (TIC) for neighboring sources within a 60 arcsecond search radius. 

The blend probability is estimated using a Gaussian Point Spread Function (PSF) contamination model. Each neighboring star's flux is weighted based on its distance from the target and a standard deviation roughly matching the TESS pixel scale (21 arcseconds/pixel). Specifically, sources within 1 pixel (21") carry high risk, those within 2 pixels carry medium risk, and beyond that carry low risk. A baseline contamination ratio is calculated, and additional penalties are applied for nearby bright neighbors (brighter than the target), ultimately yielding a normalized `blend_probability` score from 0 to 1.

## Classification

The core classification engine categorizes the processed light curves into four distinct classes:
1. **Transit**: Genuine exoplanet candidate exhibiting typical U-shaped transits and flat out-of-transit baselines.
2. **Eclipsing Binary**: Stellar binaries often characterized by deep, V-shaped primary and secondary eclipses.
3. **Blend**: Light curves heavily contaminated by neighboring stars, causing false positive transit signals.
4. **Other**: Noise, systematics, or stellar variability that cannot be classified as a transit or eclipse.

The classifier utilizes 14 engineered features extracted during the Box Least Squares (BLS) search and phase-folding processes, capturing transit symmetry, ingress/egress duration, depth, SNR, and odd/even eclipse depth disparities.

## Training Methodology

To accommodate the complex decision boundaries and inherent class imbalances in the dataset, we evaluated both Random Forest and XGBoost ensemble classifiers. 

Hyperparameter optimization was conducted via Randomized Search. Recognizing the class imbalance (transits are significantly rarer than background noise or EBs), we integrated the Synthetic Minority Over-sampling Technique (SMOTE) into the training pipeline. SMOTE synthesizes new examples for minority classes in the feature space, preventing the model from becoming biased towards the majority classes. The best-performing model (Random Forest or XGBoost) is automatically selected based on its macro F1-score.

## Evaluation

Model performance is rigorously evaluated on a held-out test set (20% of the dataset) using several key metrics:
- **Precision**: Measuring the proportion of true transit predictions against all transit predictions (minimizing false positives).
- **Recall**: Measuring the proportion of actual transits successfully identified by the pipeline (minimizing false negatives).
- **F1 Score**: The harmonic mean of Precision and Recall, used as the primary optimization metric.
- **Confusion Matrix**: Used to visually identify where the model struggles (e.g., misclassifying Blends as Transits). 

*Metrics logs and the classifier evaluation script (`evaluate_classifier.py`) continuously generate these results after training.*

## Dataset Discussion

The pipeline supports and explicitly distinguishes between two modes of operation:
- **Synthetic Dataset**: A mock dataset (`curated_dataset.csv`) generated internally via `generate_mock_dataset.py` with parameterized feature distributions. This is primarily used for pipeline testing, validation, and continuous integration.
- **Real Curated Dataset**: The officially provided dataset containing extracted features from actual TESS light curves.

The pipeline automatically identifies synthetic data by parsing the `TIC_ID` values for the "MOCK" keyword. It appropriately logs warnings to the console when mock data is used, and it explicitly outputs the `dataset_type` metadata to `metrics.json` to prevent accidental inclusion of mock-trained models in the final submission.
