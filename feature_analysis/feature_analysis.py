import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    df = pd.read_csv("../all_features_and_shap.csv", index_col=0)
    
    df_t = pd.read_csv("../all_taillard_and_shap.csv", index_col=0)
    
    feats = ["makespan_lb_meanload", "betweenness_mean", "betweenness_range", "deg_d_mean", "num_disjunctive_edges", "makespan_range", "num_edges_total"]
    feats_shap = [c + "_shap" for c in feats]
    feats_name = [r"$\underline{C}_{\max}^{\,\mathrm{load}}$", r"$\mathcal{B}_{mean}$", r"$\mathcal{B}_{range}$", r"$\deg_d^{mean}$", r"$|E_d|$", r"$C_{range}$", r"$|E|$"]
    
    for c in feats_shap:
        df[c] = 2 * (df[c] - df[c].min()) / (df[c].max() - df[c].min()) - 1
    
    for c in feats_shap:
        df_t[c] = 2 * (df_t[c] - df_t[c].min()) / (df_t[c].max() - df_t[c].min()) - 1
    
    df_makespan_meanload = df.sort_values(by="makespan_lb_meanload_shap", axis=0)    
    
    df_betweenness_mean = df.sort_values(by="betweenness_mean_shap", axis=0)
    
    df_betweenness_range = df.sort_values(by="betweenness_range_shap", axis=0)
    
    df_deg_d_mean = df.sort_values(by="deg_d_mean_shap", axis=0)

    df_num_disjunctive_edges = df.sort_values(by="num_disjunctive_edges_shap", axis=0)

    df_makespan_range = df.sort_values(by="makespan_range_shap", axis=0)

    df_num_edges_total = df.sort_values(by="num_edges_total_shap", axis=0)
    
    
    df_t_makespan_meanload = df_t.sort_values(by="makespan_lb_meanload_shap", axis=0)
    
    df_t_betweenness_mean = df_t.sort_values(by="betweenness_mean_shap", axis=0)
    
    df_t_betweenness_range = df_t.sort_values(by="betweenness_range_shap", axis=0)
    
    df_t_deg_d_mean = df_t.sort_values(by="deg_d_mean_shap", axis=0)

    df_t_num_disjunctive_edges = df_t.sort_values(by="num_disjunctive_edges_shap", axis=0)

    df_t_makespan_range = df_t.sort_values(by="makespan_range_shap", axis=0)

    df_t_num_edges_total = df_t.sort_values(by="num_edges_total_shap", axis=0)

    # correlation matrices
    plt.rcParams['text.usetex'] = True
    plt.ticklabel_format(useMathText=False)
    
    sns.set_theme(
        context="paper",
        style="white",
        font_scale=1.2
    )
    
    # IGJSP
    
    plt.figure()
    matrix = df[feats].corr(method="pearson")
    matrix1 = matrix
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_all.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_makespan_meanload[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_makespan_meanload_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_makespan_meanload[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_makespan_meanload_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_makespan_meanload[:10], df_makespan_meanload[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_makespan_meanload_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_betweenness_mean[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_betweenness_mean_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_betweenness_mean[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_betweenness_mean_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_betweenness_mean[:10], df_betweenness_mean[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_betweenness_mean_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_betweenness_range[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_betweenness_range_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_betweenness_range[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_betweenness_range_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_betweenness_range[:10], df_betweenness_range[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_betweenness_range_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_deg_d_mean[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_deg_d_mean_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_deg_d_mean[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_deg_d_mean_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_deg_d_mean[:10], df_deg_d_mean[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_deg_d_mean_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_num_disjunctive_edges[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_num_disjunctive_edges_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_num_disjunctive_edges[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_num_disjunctive_edges_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_num_disjunctive_edges[:10], df_num_disjunctive_edges[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_num_disjunctive_edges_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_makespan_range[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_makespan_range_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_makespan_range[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_makespan_range_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_makespan_range[:10], df_makespan_range[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_makespan_range_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_num_edges_total[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_num_edges_total_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_num_edges_total[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_num_edges_total_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_num_edges_total[:10], df_num_edges_total[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/feat/corr_matrix_num_edges_total_high_low.png", dpi=400)
    plt.close()
    
    
    
    plt.figure()
    matrix = df[feats_shap].corr(method="pearson")
    matrix2 = matrix
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_all_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_makespan_meanload[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_makespan_meanload_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_makespan_meanload[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_makespan_meanload_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_makespan_meanload[:10], df_makespan_meanload[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_makespan_meanload_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_betweenness_mean[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_betweenness_mean_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_betweenness_mean[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_betweenness_mean_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_betweenness_mean[:10], df_betweenness_mean[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_betweenness_mean_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_betweenness_range[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_betweenness_range_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_betweenness_range[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_betweenness_range_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_betweenness_range[:10], df_betweenness_range[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_betweenness_range_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_deg_d_mean[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_deg_d_mean_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_deg_d_mean[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_deg_d_mean_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_deg_d_mean[:10], df_deg_d_mean[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_deg_d_mean_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_num_disjunctive_edges[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_num_disjunctive_edges_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_num_disjunctive_edges[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_num_disjunctive_edges_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_num_disjunctive_edges[:10], df_num_disjunctive_edges[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_num_disjunctive_edges_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_makespan_range[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_makespan_range_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_makespan_range[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_makespan_range_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_makespan_range[:10], df_makespan_range[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_makespan_range_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_num_edges_total[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_num_edges_total_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_num_edges_total[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_num_edges_total_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_num_edges_total[:10], df_num_edges_total[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./IGJSP/shap/corr_matrix_num_edges_total_high_low_shap.png", dpi=400)
    plt.close()
    
    
    
    # JSPLIB
    
    plt.figure()
    matrix = df_t[feats].corr(method="pearson")
    matrix3 = matrix
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_all.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_makespan_meanload[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_makespan_meanload_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_makespan_meanload[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_makespan_meanload_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_makespan_meanload[:10], df_t_makespan_meanload[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_makespan_meanload_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_betweenness_mean[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_betweenness_mean_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_betweenness_mean[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_betweenness_mean_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_betweenness_mean[:10], df_t_betweenness_mean[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_betweenness_mean_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_betweenness_range[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_betweenness_range_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_betweenness_range[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_betweenness_range_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_betweenness_range[:10], df_t_betweenness_range[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_betweenness_range_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_deg_d_mean[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_deg_d_mean_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_deg_d_mean[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_deg_d_mean_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_deg_d_mean[:10], df_t_deg_d_mean[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_deg_d_mean_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_num_disjunctive_edges[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_num_disjunctive_edges_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_num_disjunctive_edges[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_num_disjunctive_edges_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_num_disjunctive_edges[:10], df_t_num_disjunctive_edges[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_num_disjunctive_edges_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_makespan_range[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_makespan_range_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_makespan_range[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_makespan_range_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_makespan_range[:10], df_t_makespan_range[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_makespan_range_high_low.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_num_edges_total[:10][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_num_edges_total_low.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_num_edges_total[-10:][feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_num_edges_total_high.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_num_edges_total[:10], df_t_num_edges_total[-10:]])[feats].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/feat/corr_matrix_t_num_edges_total_high_low.png", dpi=400)
    plt.close()
    
    
    
    plt.figure()
    matrix = df_t[feats_shap].corr(method="pearson")
    matrix4 = matrix
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_all_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_makespan_meanload[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_makespan_meanload_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_makespan_meanload[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_makespan_meanload_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_makespan_meanload[:10], df_t_makespan_meanload[-10:]])[feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_makespan_meanload_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_betweenness_mean[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_betweenness_mean_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_betweenness_mean[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_betweenness_mean_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_betweenness_mean[:10], df_t_betweenness_mean[-10:]])[feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_betweenness_mean_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_betweenness_range[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_betweenness_range_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_betweenness_range[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_betweenness_range_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_betweenness_range[:10], df_t_betweenness_range[-10:]])[feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_betweenness_range_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_deg_d_mean[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_deg_d_mean_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_deg_d_mean[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_deg_d_mean_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_deg_d_mean[:10], df_t_deg_d_mean[-10:]])[feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_deg_d_mean_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_num_disjunctive_edges[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_num_disjunctive_edges_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_num_disjunctive_edges[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_num_disjunctive_edges_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_num_disjunctive_edges[:10], df_t_num_disjunctive_edges[-10:]])[feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_num_disjunctive_edges_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_makespan_range[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_makespan_range_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_makespan_range[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_makespan_range_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_makespan_range[:10], df_t_makespan_range[-10:]])[feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_makespan_range_high_low_shap.png", dpi=400)
    plt.close()
    
    
    plt.figure()
    matrix = df_t_num_edges_total[:10][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_num_edges_total_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = df_t_num_edges_total[-10:][feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_num_edges_total_high_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = pd.concat([df_t_num_edges_total[:10], df_t_num_edges_total[-10:]])[feats_shap].corr(method="pearson")
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./JSPLIB/shap/corr_matrix_t_num_edges_total_high_low_shap.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = (matrix1 + matrix3) / 2.0
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"corr_matrix_all_sum.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix = (matrix2 + matrix4) / 2.0
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"corr_matrix_all_shap_sum.png", dpi=400)
    plt.close()
    
    plt.figure()
    matrix2.columns = feats
    matrix2.index = feats
    matrix4.columns = feats
    matrix4.index = feats
    matrix = (matrix1 + matrix2 + matrix3 + matrix4) / 4.0
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        linecolor="white",
        xticklabels=feats_name,
        yticklabels=feats_name,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"corr_matrix_all_feats_shap_sum.png", dpi=400)
    plt.close()
    

if __name__ == "__main__":
    main()