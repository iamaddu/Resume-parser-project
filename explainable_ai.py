def explain_candidate(features, model):
    # features: dict of cognitive features for a candidate
    # model: trained classifier
    # Returns: dict with explanation
    importances = model.feature_importances_
    feature_names = list(features.keys())
    sorted_feats = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    top_factors = [f"{name}: {features[name]}" for name, _ in sorted_feats[:3]]
    return {
        "why_this_candidate": f"Top factors: {', '.join(top_factors)}",
        "growth_potential": "High" if features["learning_velocity"] < 1.5 else "Moderate",
        "cognitive_style": "Specialist" if features["knowledge_depth_vs_breadth"] > 0.7 else "Generalist"
    }