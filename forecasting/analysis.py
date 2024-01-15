import pandas as pd
import numpy as np
from copy import deepcopy

# forecasting_directory = Path(__file__).parent.resolve()

with pd.ExcelFile("../forecasting/data/forecasting2024.xlsx", "openpyxl") as xl_file:
    data = pd.read_excel(
        xl_file,
        sheet_name="Master",
        usecols="B,D,E,J:EQ",
        skiprows=1,
        nrows=30,
        index_col=0,
    )

data["Category"].value_counts()

short_name = [
    "Lai Ching-te",
    "Apple TV+",
    "CRBN vs XOP",
    "Australian heat",
    "New Delhi air quality",
    "EU tax haven blacklist",
    "Millennium Prize",
    "India spaceflight",
    "WWE Universal Champion",
    "Harden on LA Clippers",
    "HRH Prince Harry?",
    "Lyft acquired",
    "X (Twitter) renamed?",
    "US governors resign",
    "Israeli PM Netanyahu",
    "US SCt justice resigns",
    "US artist auction price record",
    "US House Speaker Johnson",
    "Jets QB Aaron Rodgers",
    "US and China Olympic medal counts",
    "Top 10 hotels worldwide",
    "NY Mets better record than NY Yankees",
    "Non-US winner of economics Nobel",
    "US Congressman felony conviction",
    "Zuckerberg tweets",
    "Neither Trump nor Biden elected",
    "Tesla recalls",
    "Gladiator 2 vs Beetlejuice 2 on Rotten Tomatoes",
    "Highest COL cities in Asia",
    "Formula 1 champion",
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
    "Lai Ching-te": True,  # 2024-01-13
    "Apple TV+": False,  # 2024-01-14
}
# 'CRBN vs XOP',  # 2024-02-01
# 'Australian heat',  # 2024-02-15
# 'New Delhi air quality',  # 2024-03-01
# 'EU tax haven blacklist',  # 2024-03-08
# 'Millennium Prize',  # 2024-03-15
# 'India spaceflight',  # 2024-03-31
# 'WWE Universal Champion',  # 2024-04-07
# 'Harden on LA Clippers',  # 2024-04-14
# 'HRH Prince Harry?',  # 2024-05-01
# 'Lyft acquired',  # 2024-05-15
# 'X (Twitter) renamed?',  # 2024-06-01
# 'US governors resign',  # 2024-06-15
# 'Israeli PM Netanyahu',  # 2024-06-30
# 'US SCt justice resigns',  # 2024-07-15
# 'US artist auction price record',  # 2024-08-01
# 'US House Speaker Johnson',  # 2024-08-20
# 'Jets QB Aaron Rodgers',  # 2024-09-05
# 'US and China Olympic medal counts',  # 2024-09-08
# 'Top 10 hotels worldwide',  # 2024-09-20
# 'NY Mets better record than NY Yankees',  # 2024-09-29
# 'Non-US winner of economics Nobel',  # 2024-10-10
# 'US Congressman felony conviction',  # 2024-10-15
# 'Zuckerberg tweets',  # 2024-10-25
# 'Neither Trump nor Biden elected',  # 2024-11-06
# 'Tesla recalls',  # 2024-11-15
# 'Gladiator 2 vs Beetlejuice 2 on Rotten Tomatoes',  # 2024-11-26
# 'Highest COL cities in Asia',  # 2024-12-01
# 'Formula 1 champion'  # 2024-12-08

resolved_bools.update(evts_resolved)

leader_median = (
    data.drop(["Category", "Date", "Event"], axis=1)
    .apply(lambda x: calc_score(x, resolved_bools, data_w_median["median"]))
    .sort_values()
)
leader_resolved_only = (
    data.drop(["Category", "Date", "Event"], axis=1)
    .apply(lambda x: calc_score(x, resolved_bools, pd.NA))
    .sort_values()
)

with pd.ExcelWriter("../forecasting/results/rankings.xlsx") as f:
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
