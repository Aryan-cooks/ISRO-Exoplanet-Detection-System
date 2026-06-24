# ISRO Exoplanet Detection System

A Python-based pipeline for discovering and validating exoplanets from TESS (Transiting Exoplanet Survey Satellite) light curve data.

## Features

- **TESS Data Downloading:** Automated retrieval of detrended light curves using `lightkurve`.
- **Detrending Pipeline:** Removes systematic noise and stellar variability.
- **Transit Search:** Uses Box Least Squares (BLS) periodograms to identify periodic transit signals.
- **Transit Fitting:** Uses `batman` to model the transit and estimate planetary radius.
- **Odd-Even Eclipse Test:** Helps discriminate true transits from eclipsing binaries by comparing odd and even eclipses.
- **Blend Detection:** Queries the MAST catalog using `astroquery` to detect nearby stars within a 60 arcsecond radius and estimates contamination ratio to identify false positives.
- **Classifier:** An XGBoost-based classifier to predict whether a signal is a Transit, Eclipsing Binary, or Undetected Long Period, using physical features and blend probabilities.

## Modules

- `download_tess_data.py`: Downloads lightcurves.
- `detrend_data.py`: Flattens light curves.
- `run_bls.py`: Runs BLS on detrended curves to find periods, depths, durations.
- `odd_even_test.py`: Calculates odd/even eclipse differences.
- `fit_batman.py`: Generates the best-fit transit model.
- `blend_detector.py`: Evaluates crowding and contamination ratio for targets.
- `blend_visualization.py`: Plots sky maps centered on target with neighbor stars.
- `summary_plot.py`: Generates composite plots showing the whole pipeline.
- `train_classifier.py`: Trains the final XGBoost classifier.

## Outputs
- Data files in `/data`
- Visualizations in `/plots` and `/summary_plots`
- `feature_table.csv` and `blend_features.csv`
