import numpy as np

def extract_probabilities(pred_probs, classes):
    """
    Given the probability array and classes list from the label encoder,
    extract the exact probabilities for our 4 target categories.
    """
    prob_dict = dict(zip(classes, pred_probs))
    
    return {
        'transit_prob': prob_dict.get('Transit', 0.0),
        'binary_prob': prob_dict.get('Eclipsing Binary', 0.0),
        'blend_prob': prob_dict.get('Blend', 0.0),
        'other_prob': prob_dict.get('Other', 0.0)
    }

def calculate_confidence(max_prob):
    """
    Assigns a confidence tier based on the winning probability.
    """
    if max_prob > 0.90:
        return 'HIGH'
    elif max_prob >= 0.70:
        return 'MEDIUM'
    else:
        return 'LOW'

def process_classification(pred_probs, classes):
    """
    Returns the full classification metadata dictionary.
    """
    probs = extract_probabilities(pred_probs, classes)
    
    # Determine the winning class and max prob
    pred_idx = np.argmax(pred_probs)
    pred_class = classes[pred_idx]
    max_prob = pred_probs[pred_idx]
    
    confidence = calculate_confidence(max_prob)
    
    return {
        'predicted_class': pred_class,
        'transit_prob': probs['transit_prob'],
        'binary_prob': probs['binary_prob'],
        'blend_prob': probs['blend_prob'],
        'other_prob': probs['other_prob'],
        'confidence_level': confidence
    }
