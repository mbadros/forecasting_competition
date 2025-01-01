import pandas as pd
import numpy as np
from copy import deepcopy
import sys
from pathlib import Path

if len(sys.argv) == 1:
    # Not interactive
    base_dir = Path(__file__).parent.absolute()
else:
    # Interactive
    base_dir = Path.home().joinpath(
        "Documents/PycharmProjects/forecasting_competition/forecasting"
    )
    print(base_dir)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 1000)

with pd.ExcelFile(
    base_dir.joinpath("data/forecasting2025.xlsx"), "openpyxl"
) as xl_file:
    data = pd.read_excel(
        xl_file,
        sheet_name="Master",
        usecols="B,D,E,K:FN",
        skiprows=1,
        nrows=31,
        index_col=0,
    )

data["Category"].value_counts()

short_name = [
    "NFL wildcard home team wins",  # 1/13/2025
    "DJT market cap",  # 1/20/2025
    "Sabrina Carpenter Grammy",  # 2/4/2025
    "Liechtenstein elections",  # 2/9/2025
    "House Speaker Mike Johnson",  # 2/20/2025
    "Astronauts back on Earth",  # 3/1/2025
    "Telsa Cyberbeast MSRP",  # 3/15/2025
    "Taylor Swift engaged",  # 4/1/2025
    "German Chancellor Merz",  # 4/22/2025
    "EU member Kosovo",  # 5/10/2025
    "P Diddy convicted",  # 6/1/2025
    "Sea surface temperature",  # 6/15/2025
    "Bulgaria in EU",  # 7/1/2025
    "World's Strongest Man non-NAm",  # 7/5/2025
    "FIFA Club World Cup is from EU",  # 7/13/2025
    "US SCt justice retires",  # 7/20/2025
    "Cabinet secretary leaves",  # 8/1/2025
    "Prime number with 50mm digits",  # 8/15/2025
    "Magnif 7 ETF vs SPX",  # 9/1/2025
    "France wins Ocean Race Europe",  # 9/10/2025
    "Greens win in Australian elections",  # 9/27/2025
    "Ohtani's MLB regular season stats",  # 9/29/2025
    "Phoenix is above 100 degrees x 75 days",  # 10/1/2025
    "Grand Theft Auto VI",  # 10/6/2025
    "Econ Nobel Prize winner is from MIT",  # 10/13/2025
    "Trudeau PM of Canada",  # 10/25/2025
    "Dem wins VA governors race",  # 11/4/2025
    "Argentine monthly inflation is negative",  # 11/12/205
    "Avatar Fire & Ash running time",  # 11/25/2025
    "Max US personal income tax rate capped at 37%",  # 12/10/2025
]

data["Prop"] = short_name
data.reset_index(inplace=True)
data.set_index("Prop", inplace=True)

median = data.median(axis=1, numeric_only=True)
mean = data.mean(axis=1, numeric_only=True)
quartile_1 = data.drop("Date", axis=1).quantile(q=0.25, axis=1, numeric_only=True)
quartile_3 = data.drop("Date", axis=1).quantile(q=0.75, axis=1, numeric_only=True)
std_dev = data.std(axis=1, numeric_only=True)
prop_max = data.drop("Date", axis=1).max(axis=1, numeric_only=True)
prop_min = data.drop("Date", axis=1).min(axis=1, numeric_only=True)

summary_stats = pd.concat(
    {
        "minimum": prop_min,
        "25th pctl": quartile_1,
        "median": median,
        "75th pctl": quartile_3,
        "maximum": prop_max,
        "mean": mean,
        "std dev": std_dev,
    },
    axis=1,
)

events_by_std_dev = summary_stats.sort_values(by="std dev", axis=0, ascending=False)
events_by_mean = summary_stats.sort_values(by="mean", axis=0, ascending=False)
events_by_median = summary_stats.sort_values(by="median", axis=0, ascending=False)

data_w_median = deepcopy(data)
data_w_median["median"] = median
data_w_median["coinflip"] = 50

correls_no_median = data.corr("pearson", numeric_only=True)
correls_w_median = data_w_median.corr("pearson", numeric_only=True)
correls_to_median = (
    correls_w_median["median"].sort_values(ascending=False).drop("median")
)

upper = correls_no_median.where(
    np.triu(np.ones(correls_no_median.shape), k=1).astype(bool)
)

top_corr = upper.stack().sort_values(axis=0, ascending=False)

off_diag = correls_no_median.where(
    (
        np.ones(correls_no_median.shape)
        - np.diag(np.ones(correls_no_median.shape[0]), k=0)
    ).astype(bool)
)

top_5_correls_by_player = pd.concat(
    {
        g: df.sort_values(0, ascending=False).head(5)
        for g, df in off_diag.stack().reset_index().groupby("level_0")
    }
)
top_5_correls_by_player.drop("level_0", axis=1, inplace=True)

high_correl_by_player = pd.concat(
    {
        g: df.sort_values(0, ascending=False).head(1)
        for g, df in off_diag.stack().reset_index().groupby("level_0")
    }
)
high_correl_by_player = high_correl_by_player.droplevel(1).drop("level_0", axis=1)

sorted_high_correl_by_player = high_correl_by_player.sort_values(0, ascending=False)
sorted_high_correl_by_player.value_counts("level_1")

players_by_std = data_w_median.std(axis=0, numeric_only=True).sort_values(
    ascending=False
)
players_by_dist_50 = (
    data_w_median.drop(["Event", "Date", "Category"], axis=1)
    .apply(lambda x: abs(x - 50))
    .mean(axis=0)
    .sort_values(ascending=False)
)


def calc_score(forecast_vect, resolved_bools, fill_val):
    # resolved_vect should be True, False, or pd.NA
    # Express forecast values as percentages (i.e., from 0 to 100) not as decimals
    resolved_vect = resolved_bools * 100
    resolved_vect.fillna(fill_val, inplace=True)
    return ((resolved_vect - forecast_vect) ** 2).sum()


# all_events_resolve_true = pd.Series(True, index=data.index)
# all_events_resolve_false = pd.Series(False, index=data.index)
# random_reso = pd.Series([bool(i) for i in np.random.randint(0,2,30)] , index=data.index)
#
# calc_score(data_w_median['coinflip'], all_events_resolve_true, pd.NA)
# calc_score(data_w_median['coinflip'], all_events_resolve_false, pd.NA)
# calc_score(data_w_median['coinflip'], random_reso, pd.NA)


# Formatting and Naming
sorted_high_correl_by_player.index.name = "Player"
sorted_high_correl_by_player.columns = ["Most Correlated to", "Correlation"]

top_corr.index.names = ["Pair - Player 1", "Pair - Player 2"]
top_corr.name = "Correlation"

top_5_correls_by_player.index.name = ("Player", "ID")
top_5_correls_by_player.columns = ["Most Correlated to", "Correlation"]

correls_to_median.index.name = "Player"
correls_to_median.name = "Correlation to Median"

players_by_std.name = "Standard Deviation"
players_by_dist_50.columns = "Distance from 50"

correl_fmt = "%.5f"
alt_fmt = "%.2f"
score_fmt = "%.0f"

resolved_bools = pd.Series(pd.NA, index=data.index)

# ********************************
# Resolution of events goes here
#

evts_resolved = {
    "NFL wildcard home team wins": None,  # 1/13/2025
    "DJT market cap": None,  # 1/20/202n5
    "Sabrina Carpenter Grammy": None,  # 2/4/2025
    "Liechtenstein elections": None,  # 2/9/2025
    "House Speaker Mike Johnson": None,  # 2/20/2025
    "Astronauts back on Earth": None,  # 3/1/2025
    "Telsa Cyberbeast MSRP": None,  # 3/15/2025
    "Taylor Swift engaged": None,  # 4/1/2025
    "German Chancellor Merz": None,  # 4/22/2025
    "EU member Kosovo": None,  # 5/10/2025
    "P Diddy convicted": None,  # 6/1/2025
    "Sea surface temperature": None,  # 6/15/2025
    "Bulgaria in EU": None,  # 7/1/2025
    "World's Strongest Man non-NAm": None,  # 7/5/2025
    "FIFA Club World Cup is from EU": None,  # 7/13/2025
    "US SCt justice retires": None,  # 7/20/2025
    "Cabinet secretary leaves": None,  # 8/1/2025
    "Prime number with 50mm digits": None,  # 8/15/2025
    "Magnif 7 ETF vs SPX": None,  # 9/1/2025
    "France wins Ocean Race Europe": None,  # 9/10/2025
    "Greens win in Australian elections": None,  # 9/27/2025
    "Ohtani's MLB regular season stats": None,  # 9/29/2025
    "Phoenix is above 100 degrees x 75 days": None,  # 10/1/2025
    "Grand Theft Auto VI": None,  # 10/6/2025
    "Econ Nobel Prize winner is from MIT": None,  # 10/13/2025
    "Trudeau PM of Canada": None,  # 10/25/2025
    "Dem wins VA governors race": None,  # 11/4/2025
    "Argentine monthly inflation is negative": None,  # 11/12/205
    "Avatar Fire & Ash running time": None,  # 11/25/2025
    "Max US personal income tax rate capped at 37%": None,  # 12/10/2025
}

resolved_bools.update({k: v for k, v in evts_resolved.items() if v is not None})

entrants_matrix = data.drop(["Category", "Date", "Event"], axis=1)

# resolved events are 0 or 1; other events are median
median_prob_true = (resolved_bools * 100).fillna(data_w_median["median"]) / 100
median_prob_false = 1 - median_prob_true

entrant_false = entrants_matrix.mul(entrants_matrix)
entrants_true = (100 - entrants_matrix).mul(100 - entrants_matrix)

# pd.concat({
#     'points_if_true': entrants_true['Katie Bruce'],
#     'points_if_false': entrant_false['Katie Bruce'],
#     'median_prob_true': median_prob_true,
#     'median_prob_false': median_prob_false,
#     'points_for_false': entrant_false['Katie Bruce'].mul(median_prob_false, axis=0),
#     'ponts_for_true': entrants_true['Katie Bruce'].mul(median_prob_true, axis=0),
#     'points_act_or_exp': (
#         entrant_false.mul(median_prob_false, axis=0) +
#         entrants_true.mul(median_prob_true, axis=0))['Katie Bruce']
# }, axis=1)

leader_median = (
    (
        entrant_false.mul(median_prob_false, axis=0)
        + entrants_true.mul(median_prob_true, axis=0)
    )
    .sum()
    .sort_values()
)


leader_resolved_only = (
    data.drop(["Category", "Date", "Event"], axis=1)
    .apply(lambda x: calc_score(x, resolved_bools, pd.NA))
    .sort_values()
)

with pd.ExcelWriter(base_dir.joinpath("results/rankings_2025.xlsx")) as f:
    sorted_high_correl_by_player.to_excel(f, sheet_name="Top Correl by Player")
    top_5_correls_by_player.to_excel(f, sheet_name="Top 5 Correls by Player")
    correls_to_median.to_excel(f, sheet_name="Correl to Median")
    top_corr.to_excel(f, sheet_name="Top Correlations")
    summary_stats.to_excel(f, sheet_name="Summary Stats")
    events_by_std_dev.to_excel(f, sheet_name="Events Ranked by Std Dev")
    events_by_mean.to_excel(f, sheet_name="Events Ranked by Mean")
    events_by_median.to_excel(f, sheet_name="Events Ranked by Median")
    players_by_std.to_excel(f, sheet_name="Players Ranked by Std Dev")
    players_by_dist_50.to_excel(f, sheet_name="Players Dist from 50")
    leader_median.to_excel(f, sheet_name="Leaders - Blended Med")
    leader_resolved_only.to_excel(f, sheet_name="Leaders - Resolved Only")
