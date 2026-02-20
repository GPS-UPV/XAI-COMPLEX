import dtale
import dtale.global_state as global_state
import pandas as pd

global_state.set_chart_settings({"scatter_points": 150000, "3d_points": 400000})
global_state.set_app_settings(dict(enable_custom_filters=True,enable_web_uploads=True))

df = pd.read_csv("./graphs/complexity_scores.csv")
df_W = pd.read_csv("./graphs/complexity_scores_W.csv")
df_W = pd.read_csv("./graphs/weighted_y_components.csv")

d = dtale.show(df_W,subprocess=False)
d.open_browser()