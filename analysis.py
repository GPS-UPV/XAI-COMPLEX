import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def main():
    
    df = pd.read_csv("all_features_and_shap.csv")  
    
    df = df.set_index(df["instance_id"])
        
    shap_columns = [c for c in df.columns if "shap" in c]
    
    df_shap = df[shap_columns]
    
    df_shap.set_index(df.index)
    
    df_shap_norm = 2 * (df_shap - df_shap.min()) / (df_shap.max() - df_shap.min()) - 1
    
    df_shap_norm.to_csv("normalised_shap_values.csv")

    optimal_mask, feasible_mask, timeout_mask = [], [], []
    
    for i in df.index:
        if "OPTIMAL" in df.loc[i, "quality_tag"].strip().upper():
            optimal_mask.append(i)
        elif "FEASIBLE" in df.loc[i, "quality_tag"].strip().upper():
            feasible_mask.append(i)
        elif "TIMEOUT" in df.loc[i, "quality_tag"].strip().upper():
            timeout_mask.append(i)

    df_optimal = df_shap_norm.loc[optimal_mask]
    
    df_feasible = df_shap_norm.loc[feasible_mask]
    
    df_timeout = df_shap_norm.loc[timeout_mask]
    
    res = pd.DataFrame()
    
    res["optimal_feasible_diff"] = df_optimal.mean(0) - df_feasible.mean(0)
    
    res["optimal_timeout_diff"] = df_optimal.mean(0) - df_timeout.mean(0)
    
    res["timeout_feasible_diff"] = df_timeout.mean(0) - df_feasible.mean(0)
    
    res["std_optimal"] = df_optimal.std()
    
    res["std_feasible"] = df_feasible.std()
    
    res["std_timeout"] = df_timeout.std()
    
    res.to_csv("stats.csv")
    
    print("Archivo guardado")
    
        
if __name__ == "__main__":
    main()