# Blend Validation Report (v1 vs v2)

This report validates the new distance-weighted contamination framework and TESS pixel-scale metrics against the previous v1 model.

## Contamination Comparison

The table below contrasts the raw contamination ratio (v1) against the new distance-weighted contamination ratio (v2) which utilizes a Gaussian PSF model ($\sigma=15"$) to downweight stars based on their angular separation from the target.

| TIC_ID        | V1 Contamination Ratio | V2 Weighted Contamination | Neighbors < 21" (1px) |
|:--------------|----------------------:|-------------------------------:|-----------------------:|
| TIC_100100827 |           0.00688666  |                    2.72514e-05 |                      0 |
| TIC_150361911 |           0.111866    |                    0.00907529  |                      2 |
| TIC_283722336 |           0.00095075  |                    6.62482e-05 |                      4 |
| TIC_38699825  |           0.000275758 |                    3.2731e-05  |                      1 |
| TIC_55525572  |           0.01554     |                    0.00591751  |                      2 |

## Analysis of Differences

1. **Massive Reduction in Distant Contamination:** 
   For `TIC_150361911`, the raw contamination was $11.1\%$. The new weighted model reduced this to $0.9\%$. This aligns with physical expectations because many of its 19 neighbors were far from the target and did not significantly bleed into the TESS aperture.

2. **Pixel Scale Occupancy:**
   Although the weighted flux dropped, we now have explicit pixel occupancy features. `TIC_283722336` has 4 neighbors inside the 1-pixel (21") radius. Even if they are faint, this crowding adds risk that the v2 model correctly captures via the continuous probability score.

The new features (`neighbors_within_1px`, `brightest_neighbor_delta_mag`, etc.) have been added alongside the distance-weighted contamination ratio. This allows the classifier to make multi-dimensional decisions rather than relying on a single sum.
