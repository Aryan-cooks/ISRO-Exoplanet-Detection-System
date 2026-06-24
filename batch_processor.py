import os
import numpy as np
import pandas as pd
import lightkurve as lk
import joblib
import traceback
from astroquery.mast import Catalogs
import astropy.units as u

# Load ML models globally for workers
try:
    classifier_model = joblib.load('model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    classifier_model = None
    label_encoder = None
    print(f"Warning: Could not load ML models: {e}")

try:
    import batman
    from scipy.optimize import curve_fit
    BATMAN_AVAILABLE = True
except ImportError:
    BATMAN_AVAILABLE = False

def calculate_flux(mag):
    return 10 ** (-0.4 * mag)

def calculate_gaussian_weight(distance_arcsec, sigma=15.0):
    return np.exp(-(distance_arcsec**2) / (2 * sigma**2))

def compute_snr(lc, period, t0, duration_hours, depth):
    phase = (lc.time.value - t0 + 0.5 * period) % period - 0.5 * period
    duration_days = duration_hours / 24.0
    in_transit = np.abs(phase) < (duration_days / 2.0)
    out_transit = ~in_transit
    n_in = np.sum(in_transit)
    if n_in == 0:
        return 0.0
    out_std = np.nanstd(lc.flux.value[out_transit])
    if out_std == 0:
        return 0.0
    return (depth / out_std) * np.sqrt(n_in)

def compute_odd_even(lc, period, t0, duration_hours):
    cycles = np.round((lc.time.value - t0) / period)
    even_mask = (cycles % 2 == 0)
    odd_mask = (cycles % 2 != 0)
    
    lc_even = lc[even_mask]
    lc_odd = lc[odd_mask]
    
    if len(lc_even) == 0 or len(lc_odd) == 0:
        return 0.0
        
    lc_even_folded = lc_even.fold(period=period, epoch_time=t0)
    lc_odd_folded = lc_odd.fold(period=period, epoch_time=t0)
    
    def get_depth_and_error(folded_lc):
        phase = folded_lc.time.value
        duration_days = duration_hours / 24.0
        in_transit = np.abs(phase) < (duration_days / 2.0)
        out_transit = ~in_transit
        n_in = np.sum(in_transit)
        if n_in == 0 or np.sum(out_transit) == 0:
            return 0.0, 0.0
        in_flux = np.nanmedian(folded_lc.flux.value[in_transit])
        out_flux = np.nanmedian(folded_lc.flux.value[out_transit])
        depth = out_flux - in_flux
        in_std = np.nanstd(folded_lc.flux.value[in_transit])
        error = in_std / np.sqrt(n_in)
        return depth, error

    depth_even, err_even = get_depth_and_error(lc_even_folded)
    depth_odd, err_odd = get_depth_and_error(lc_odd_folded)
    
    if err_even > 0 and err_odd > 0:
        combined_err = np.sqrt(err_even**2 + err_odd**2)
        sigma = abs(depth_even - depth_odd) / combined_err
    else:
        sigma = 0.0
        
    return sigma

def compute_transit_shape_features(lc, period, t0, duration_hours):
    """
    Computes shape-based features of the transit:
    - symmetry_score: Compare area of ingress vs egress.
    - ingress/egress_duration: Measured from 10% to 90% depth.
    - v_shape/u_shape scores: Proxies for V-shaped (EB) or U-shaped (Transit) transits.
    """
    # Phase fold
    folded_lc = lc.fold(period=period, epoch_time=t0)
    
    # Isolate the transit and immediate out-of-transit baseline
    phase = folded_lc.time.value
    duration_days = duration_hours / 24.0
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    phase = phase[sort_idx]
    flux = folded_lc.flux.value[sort_idx]
    
    # We define the transit window as +/- duration_days
    window_mask = np.abs(phase) <= duration_days
    if np.sum(window_mask) < 10:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    p_win = phase[window_mask]
    f_win = flux[window_mask]
    
    # Find exact minimum flux (transit bottom)
    min_idx = np.argmin(f_win)
    t_min = p_win[min_idx]
    
    # Separate into left half (ingress side) and right half (egress side)
    left_mask = p_win <= 0.0
    right_mask = p_win > 0.0
    
    f_left = f_win[left_mask]
    f_right = f_win[right_mask]
    
    # Interpolate to compute symmetry (compare area)
    area_left = np.trapz(1.0 - f_left, p_win[left_mask]) if len(f_left) > 1 else 0
    area_right = np.trapz(1.0 - f_right, p_win[right_mask]) if len(f_right) > 1 else 0
    
    if (area_left + area_right) > 0:
        symmetry_score = 1.0 - abs(area_left - area_right) / (area_left + area_right)
    else:
        symmetry_score = 0.0
        
    # Ingress / Egress duration proxy based on percentiles of flux drop
    # 10% to 90% flux drop
    f_baseline = 1.0
    f_bottom = f_win[min_idx]
    depth = f_baseline - f_bottom
    
    if depth <= 0:
        return 0.0, 0.0, symmetry_score, 0.0, 0.0
        
    level_10 = f_baseline - 0.1 * depth
    level_90 = f_baseline - 0.9 * depth
    
    try:
        t1 = p_win[np.where(f_win < level_10)[0][0]]   # First time below 10% depth
        t2 = p_win[np.where(f_win < level_90)[0][0]]   # First time below 90% depth
        t3 = p_win[np.where(f_win < level_90)[0][-1]]  # Last time below 90% depth
        t4 = p_win[np.where(f_win < level_10)[0][-1]]  # Last time below 10% depth
        
        ingress_duration = (t2 - t1) * 24.0 # hours
        egress_duration = (t4 - t3) * 24.0  # hours
        
        # Total transit duration (T14)
        T14 = (t4 - t1)
        
        if T14 > 0:
            shape_param = ( (t2 - t1) + (t4 - t3) ) / T14
        else:
            shape_param = 1.0
            
        v_shape_score = shape_param
        u_shape_score = 1.0 - shape_param
        
    except IndexError:
        ingress_duration = 0.0
        egress_duration = 0.0
        v_shape_score = 0.0
        u_shape_score = 0.0
        
    return ingress_duration, egress_duration, symmetry_score, v_shape_score, u_shape_score

def run_blend_detection(tic_num, search_radius_arcsec=60):
    try:
        target = Catalogs.query_criteria(catalog='Tic', ID=tic_num)
        if len(target) == 0:
            return 0.0, 0
            
        target_ra = target['ra'][0]
        target_dec = target['dec'][0]
        target_mag = target['Tmag'][0]
        if np.ma.is_masked(target_mag) or np.isnan(target_mag):
            target_mag = 10.0
            
        target_flux = calculate_flux(target_mag)
        neighbors = Catalogs.query_region(f'{target_ra} {target_dec}', radius=search_radius_arcsec*u.arcsec, catalog='TIC')
        valid_neighbors = [row for row in neighbors if row['ID'] != str(tic_num) and not np.ma.is_masked(row['Tmag'])]
        
        neighbor_count = len(valid_neighbors)
        if neighbor_count == 0:
            return 0.0, 0
            
        mags = [row['Tmag'] for row in valid_neighbors]
        dists = [row['dstArcSec'] for row in valid_neighbors]
        
        weighted_fluxes = [calculate_flux(m) * calculate_gaussian_weight(d) for m, d in zip(mags, dists)]
        weighted_contamination = sum(weighted_fluxes) / target_flux
        
        base_prob = 1.0 - np.exp(-20 * weighted_contamination)
        neighbors_1px = sum(1 for d in dists if d <= 21)
        base_prob += 0.15 * neighbors_1px
        
        brightest_mag = min(mags)
        if (target_mag - brightest_mag) > 0:
            base_prob += 0.2
            
        blend_prob = min(1.0, max(0.0, base_prob))
        return blend_prob, neighbor_count
    except Exception:
        return 0.0, 0

def fit_transit(lc, period, t0, duration_hours, depth):
    if not BATMAN_AVAILABLE:
        return {}
    
    def transit_model(t, fit_t0, fit_period, rp, a):
        params = batman.TransitParams()
        params.t0 = fit_t0
        params.per = fit_period
        params.rp = rp
        params.a = a
        params.inc = 90.0
        params.ecc = 0.0
        params.w = 90.0
        params.u = [0.3, 0.2]
        params.limb_dark = "quadratic"
        m = batman.TransitModel(params, t)
        return m.light_curve(params)
        
    rp_rs_init = np.sqrt(depth) if depth > 0 else 0.1
    a_init = period / (np.pi * (duration_hours / 24.0))
    if a_init < 1.0: a_init = 10.0
    
    p0 = [t0, period, rp_rs_init, a_init]
    bounds = (
        [t0 - 0.5, period - 0.5, 0.0001, 1.0],
        [t0 + 0.5, period + 0.5, 0.9999, 500.0]
    )
    
    try:
        popt, _ = curve_fit(transit_model, lc.time.value, lc.flux.value, p0=p0, bounds=bounds)
        fit_t0, fit_period, fit_rp, fit_a = popt
        return {
            'fit_rp_rs': fit_rp,
            'fit_a_rs': fit_a,
            'fit_period': fit_period
        }
    except Exception:
        return {}

def process_target(file_path):
    filename = os.path.basename(file_path)
    # E.g. TIC_100100827_TESS_Sector_02_detrended.fits
    tic_id = filename.split('_TESS')[0]
    try:
        tic_num = int(tic_id.replace("TIC_", ""))
    except ValueError:
        return {'tic_id': tic_id, 'status': 'error', 'error': 'Invalid TIC ID format'}
        
    result = {
        'tic_id': tic_id,
        'status': 'success',
        'best_period': None,
        'depth': None,
        'duration': None,
        'snr': None,
        'ingress_duration': None,
        'egress_duration': None,
        'transit_symmetry_score': None,
        'v_shape_score': None,
        'u_shape_score': None,
        'predicted_class': 'Unknown',
        'classification_probability': 0.0,
        'transit_prob': 0.0,
        'binary_prob': 0.0,
        'blend_prob': 0.0,
        'other_prob': 0.0,
        'confidence_level': 'LOW',
        'blend_probability': 0.0,
        'error': ''
    }
    
    try:
        lc = lk.read(file_path)
        
        # Flatten if not already detrended (assuming raw FITS or general FITS)
        if 'detrended' not in filename.lower():
            lc = lc.flatten(window_length=101)
            
        baseline = lc.time[-1].value - lc.time[0].value
        if baseline < 1.0:
            raise ValueError("Baseline too short")
            
        max_period = min(365.0, baseline / 3.0)
        period_grid = np.linspace(0.5, max_period, 10000)
        periodogram = lc.to_periodogram(method='bls', period=period_grid)
        
        period = periodogram.period_at_max_power.value
        duration_hours = periodogram.duration_at_max_power.value * 24.0
        depth = periodogram.depth_at_max_power
        t0 = periodogram.transit_time_at_max_power.value
        power = periodogram.max_power.value
        
        snr = compute_snr(lc, period, t0, duration_hours, depth)
        odd_even_sigma = compute_odd_even(lc, period, t0, duration_hours)
        blend_prob, neighbor_count = run_blend_detection(tic_num)
        
        num_transits = max(2, int(baseline / period))
        ing_dur, eg_dur, sym_score, v_score, u_score = compute_transit_shape_features(lc, period, t0, duration_hours)
        
        result['best_period'] = period
        result['depth'] = depth
        result['duration'] = duration_hours
        result['snr'] = snr
        result['ingress_duration'] = ing_dur
        result['egress_duration'] = eg_dur
        result['transit_symmetry_score'] = sym_score
        result['v_shape_score'] = v_score
        result['u_shape_score'] = u_score
        result['blend_probability'] = blend_prob
        result['odd_even_sigma'] = odd_even_sigma
        result['bls_peak_power'] = power
        result['neighbor_count'] = neighbor_count
        result['num_observed_transits'] = num_transits
        
        if classifier_model and label_encoder:
            feature_dict = {
                'Period_days': period,
                'Duration_hours': duration_hours,
                'Depth': depth,
                'SNR': snr,
                'Odd_Even_Sigma': odd_even_sigma,
                'blend_probability': blend_prob,
                'neighbor_count': neighbor_count,
                'bls_peak_power': power,
                'transit_symmetry_score': sym_score,
                'v_shape_score': v_score,
                'u_shape_score': u_score,
                'ingress_duration': ing_dur,
                'egress_duration': eg_dur,
                'num_observed_transits': num_transits
            }
            
            # Feature validation step
            missing_or_invalid = [k for k, v in feature_dict.items() if v is None or np.isnan(v)]
            if missing_or_invalid:
                print(f"Feature Validation Error on {tic_id}: Missing/Invalid features: {missing_or_invalid}")
                raise ValueError(f"Missing required features: {missing_or_invalid}")
                
            features = pd.DataFrame([feature_dict])
            
            # Predict
            pred_probs = classifier_model.predict_proba(features)[0]
            from confidence_engine import process_classification
            class_metadata = process_classification(pred_probs, label_encoder.classes_)
            
            result['predicted_class'] = class_metadata['predicted_class']
            result['classification_probability'] = float(np.max(pred_probs))
            result['transit_prob'] = class_metadata['transit_prob']
            result['binary_prob'] = class_metadata['binary_prob']
            result['blend_prob'] = class_metadata['blend_prob']
            result['other_prob'] = class_metadata['other_prob']
            result['confidence_level'] = class_metadata['confidence_level']
            
            if class_metadata['predicted_class'] == 'Transit':
                fit_results = fit_transit(lc, period, t0, duration_hours, depth)
                result.update(fit_results)
                
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        
    return result
