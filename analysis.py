import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def main():
    
    df = pd.read_csv("all_features_and_shap.csv", index_col=0)  
            
    df_W = pd.read_csv("./graphs/complexity_scores_W.csv")
    
    df_W = df_W.set_index(df_W["instance_id"])
        
    shap_columns = [c for c in df.columns if "shap" in c]
    
    df_shap = df[shap_columns]
    
    df_shap.set_index(df.index)
    
    df_shap_norm = 2 * (df_shap - df_shap.min()) / (df_shap.max() - df_shap.min()) - 1
    
    df_shap_norm.to_csv("normalised_shap_values.csv")

    easy_mask, medium_mask, hard_mask = [], [], []
    
    for i in df_W["instance_id"]:
        if "easy" in df_W.loc[i, "category"].strip():
            easy_mask.append(i)
        elif "medium" in df_W.loc[i, "category"].strip():
            medium_mask.append(i)
        elif "hard" in df_W.loc[i, "category"].strip():
            hard_mask.append(i)

    df_easy = df_shap_norm.loc[easy_mask]
    
    df_medium = df_shap_norm.loc[medium_mask]
    
    df_hard = df_shap_norm.loc[hard_mask]
    
    res = pd.DataFrame()
    
    feats = ["makespan_lb_meanload", "betweenness_mean", "betweenness_range", "deg_d_mean", "num_disjunctive_edges", "makespan_range", "num_edges_total"]
    
    for c in feats:    
        res.at["easy", f"mean_{c}"] = df_easy[f"{c}_shap"].mean()
        res.at["easy", f"std_{c}"] = df_easy[f"{c}_shap"].std()
        
        res.at["medium", f"mean_{c}"] = df_medium[f"{c}_shap"].mean()
        res.at["medium", f"std_{c}"] = df_medium[f"{c}_shap"].std()
        
        res.at["hard", f"mean_{c}"] = df_hard[f"{c}_shap"].mean()
        res.at["hard", f"std_{c}"] = df_hard[f"{c}_shap"].std()    
    
    res.to_csv("stats.csv")

    print("Archivo guardado")
    
        
if __name__ == "__main__":
    main()