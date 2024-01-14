import pandas as pd
import numpy as np
from pathlib import Path
import openpyxl
from copy import deepcopy

current_directory = Path.cwd()

with pd.ExcelFile(current_directory.joinpath("forecasting/data/forecasting2024.xlsx"), "openpyxl") as xl_file:
    data = pd.read_excel(xl_file, sheet_name="Master", usecols='B,D,E,J:EQ', skiprows=1, nrows=30, index_col=0)

data['Category'].value_counts()

short_name = ['Lai Ching-te',
              'Apple TV+',
              'CRBN vs XOP',
              'Australian heat',
              'New Delhi air quality',
              'EU tax haven blacklist',
              'Millennium Prize',
              'India spaceflight',
              'WWE Universal Champion',
              'Harden on LA Clippers',
              'HRH Prince Harry?',
              'Lyft acquired',
              'X (Twitter) renamed?',
              'US governors resign',
              'Israeli PM Netanyahu',
              'US SCt justice resigns',
              'US artist auction price record',
              'US House Speaker Johnson',
              'Jets QB Aaron Rodgers',
              'US and China Olympic medal counts',
              'Top 10 hotels worldwide',
              'NY Mets better record than NY Yankees',
              'Non-US winner of economics Nobel',
              'US Congressman felony conviction',
              'Zuckerberg tweets',
              'Neither Trump nor Biden elected',
              'Tesla recalls',
              'Gladiator 2 vs Beetlejuice 2 on Rotten Tomatoes',
              'Highest COL cities in Asia',
              'Formula 1 champion']

data['Prop'] = short_name
data.reset_index(inplace=True)
data.set_index('Prop', inplace=True)

median = data.median(axis=1, numeric_only=True)
mean = data.mean(axis=1, numeric_only=True)
quartile_1 = data.drop('Date', axis=1).quantile(q=.25, axis=1, numeric_only=True)
quartile_3 = data.drop('Date', axis=1).quantile(q=.75, axis=1, numeric_only=True)
std_dev = data.std(axis=1, numeric_only=True)
prop_max = data.drop('Date', axis=1).max(axis=1, numeric_only=True)
prop_min = data.drop('Date', axis=1).min(axis=1, numeric_only=True)

summary_stats = pd.concat({'minimum': prop_min,
                           '25th pctl': quartile_1,
                           'median': median,
                           'mean': mean,
                           'std dev': std_dev,
                           '75th pctl': quartile_3,
                           'maximum': prop_max}, axis=1)

data_w_median = deepcopy(data)
data_w_median['median'] = median

correls_no_median = data.corr('pearson', numeric_only=True)
correls_w_median = data_w_median.corr('pearson', numeric_only=True)
correls_to_median = correls_w_median['median'].sort_values(ascending=False).drop('median')

upper = correls_no_median.where(np.triu(np.ones(correls_no_median.shape), k=1).astype(bool))

top_corr = upper.stack().sort_values(axis=0, ascending=False)

off_diag = correls_no_median.where((np.ones(correls_no_median.shape) - np.diag(np.ones(correls_no_median.shape[0]), k=0)).astype(bool))

top_5_correls_by_player = pd.concat({g: df.sort_values(0, ascending=False).head(5)
                                     for g, df in off_diag.stack().reset_index().groupby('level_0')})
top_5_correls_by_player.drop('level_0', axis=1, inplace=True)

high_correl_by_player = pd.concat({g: df.sort_values(0, ascending=False).head(1)
                                   for g, df in off_diag.stack().reset_index().groupby('level_0')})
high_correl_by_player = high_correl_by_player.droplevel(1).drop('level_0', axis=1)

sorted_high_correl_by_player = high_correl_by_player.sort_values(0, ascending=False)
sorted_high_correl_by_player.value_counts('level_1')


with pd.ExcelWriter(current_directory.joinpath('forecasting/results/rankings.xlsx')) as f:
    top_corr.to_excel(f, sheet_name='Top Correlations')
    sorted_high_correl_by_player.to_excel(f, sheet_name='Top Correl by Player')
    correls_to_median.to_excel(f, sheet_name='Correl to Median')
    top_5_correls_by_player.to_excel(f, sheet_name='Top 5 Correls by Player')
    summary_stats.to_excel(f, sheet_name='Summary Stats')
