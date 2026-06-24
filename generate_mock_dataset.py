import pandas as pd
import numpy as np

def generate_mock_dataset(num_samples=1500, output_path="curated_dataset.csv"):
    np.random.seed(42)
    
    # Define class distributions
    classes = ['Transit', 'Eclipsing Binary', 'Blend', 'Other']
    probs = [0.2, 0.3, 0.25, 0.25]  # Roughly imbalanced
    
    labels = np.random.choice(classes, size=num_samples, p=probs)
    
    data = []
    for i, label in enumerate(labels):
        tic_id = f"TIC_MOCK_{900000 + i}"
        
        # Base features
        period = np.random.uniform(0.5, 20.0)
        num_transits = max(2, int((27.0 / period) * np.random.uniform(0.5, 1.0)))
        
        if label == 'Transit':
            depth = np.random.uniform(0.0001, 0.02)
            duration = np.random.uniform(1.0, 5.0)
            snr = np.random.uniform(10.0, 100.0)
            odd_even = np.random.uniform(0.0, 2.5)  # low
            blend_prob = np.random.uniform(0.0, 0.2)
            neighbor_count = np.random.randint(0, 5)
            bls_power = np.random.uniform(50, 500)
            symmetry = np.random.uniform(0.8, 1.0)
            v_shape = np.random.uniform(0.1, 0.3)
            u_shape = 1.0 - v_shape
            ingress = np.random.uniform(0.1, 0.5)
            egress = ingress * np.random.uniform(0.9, 1.1)
            
        elif label == 'Eclipsing Binary':
            depth = np.random.uniform(0.05, 0.5)  # deep
            duration = np.random.uniform(2.0, 8.0)
            snr = np.random.uniform(50.0, 500.0)
            odd_even = np.random.uniform(3.0, 20.0)  # high
            blend_prob = np.random.uniform(0.0, 0.3)
            neighbor_count = np.random.randint(0, 10)
            bls_power = np.random.uniform(100, 1000)
            symmetry = np.random.uniform(0.5, 0.9)
            v_shape = np.random.uniform(0.7, 1.0)
            u_shape = 1.0 - v_shape
            ingress = np.random.uniform(0.5, 2.0)
            egress = ingress * np.random.uniform(0.8, 1.2)
            
        elif label == 'Blend':
            depth = np.random.uniform(0.001, 0.1)
            duration = np.random.uniform(1.0, 6.0)
            snr = np.random.uniform(10.0, 200.0)
            odd_even = np.random.uniform(0.0, 5.0)
            blend_prob = np.random.uniform(0.6, 1.0) # high
            neighbor_count = np.random.randint(5, 50) # high
            bls_power = np.random.uniform(30, 300)
            symmetry = np.random.uniform(0.6, 0.9)
            v_shape = np.random.uniform(0.3, 0.8)
            u_shape = 1.0 - v_shape
            ingress = np.random.uniform(0.3, 1.5)
            egress = ingress * np.random.uniform(0.5, 1.5)
            
        else: # Other (noise/systematics)
            depth = np.random.uniform(0.0001, 0.005)
            duration = np.random.uniform(0.5, 10.0)
            snr = np.random.uniform(2.0, 9.0) # low
            odd_even = np.random.uniform(0.0, 10.0)
            blend_prob = np.random.uniform(0.0, 1.0)
            neighbor_count = np.random.randint(0, 30)
            bls_power = np.random.uniform(5, 40) # low
            symmetry = np.random.uniform(0.1, 0.7) # low
            v_shape = np.random.uniform(0.0, 1.0)
            u_shape = np.random.uniform(0.0, 1.0)
            ingress = np.random.uniform(0.1, 3.0)
            egress = np.random.uniform(0.1, 3.0)
            
        data.append({
            'TIC_ID': tic_id,
            'Period_days': period,
            'Duration_hours': duration,
            'Depth': depth,
            'SNR': snr,
            'Odd_Even_Sigma': odd_even,
            'blend_probability': blend_prob,
            'neighbor_count': neighbor_count,
            'bls_peak_power': bls_power,
            'transit_symmetry_score': symmetry,
            'v_shape_score': v_shape,
            'u_shape_score': u_shape,
            'ingress_duration': ingress,
            'egress_duration': egress,
            'num_observed_transits': num_transits,
            'Label': label
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated mock dataset with {num_samples} samples at {output_path}")
    print("\nClass distribution:")
    print(df['Label'].value_counts())

if __name__ == "__main__":
    generate_mock_dataset()
