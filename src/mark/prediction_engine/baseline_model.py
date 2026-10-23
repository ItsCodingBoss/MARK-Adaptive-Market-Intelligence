from mark.feature_engine.feature_engineering import prepare_model_data   

def baseline_model(train_data, val_data): 
    baseline_class   = train_data["target_up"].mode().iloc[0]
    baseline_prediction = [baseline_class]*len(val_data)
    baseline_accuracy = (val_data["target_up"]== baseline_prediction).mean()
    print("\n--- Baseline Model ---")
    print("Baseline class:", baseline_class)
    print(f"Validation baseline accuracy: {baseline_accuracy:.2%}")
    return baseline_class, baseline_accuracy
