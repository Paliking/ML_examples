# rounding big probabilities to 1
# it is very bad idea in this competition

import pandas as pd

file_path = '../sub/LtIsLit_XGB_brandon7.csv'
df = pd.read_csv(file_path, index_col='listing_id')
print(df)

def rounder(row):
	zero_row = row*0
	for target in ['low', 'medium', 'high']:
		if row[target] > 0.90:
			zero_row[target] = 1
			return zero_row
		else:
			return row

df_rounded = df.apply(rounder, axis=1)
print(df_rounded)
# df_rounded.to_csv('../sub/brandon7_rounder90.csv', index=True)