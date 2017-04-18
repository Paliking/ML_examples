import pandas as pd
import glob
import os


def AVG_subs(glob_path, subs):
	''' 
	Average choosen submissions 
	subs: list with file names
	'''
	dfs4AVG = []
	for file_path in glob.glob(glob_path):
		if os.path.basename(file_path) in subs:
			df = pd.read_csv(file_path, index_col='listing_id')
			dfs4AVG.append(df)

	df_SUMed = dfs4AVG[0]
	for df in dfs4AVG[1:]:
		df_SUMed = df_SUMed + df

	df_AVGed = df_SUMed / len(dfs4AVG)
	print('####################################')
	print('Number of averaged submissions: ', len(dfs4AVG))
	print('####################################')
	return df_AVGed


def comp_correlation(glob_path):
	# check correlations between all submissions in folder
	index_order = None
	df4corel = pd.DataFrame()
	for file_path in glob.glob(glob_path):
		file_name = os.path.basename(file_path)
		sub = pd.read_csv(file_path, index_col='listing_id')
		if index_order is None:
			index_order = sub.index
		df4corel[file_name] = sub[['low', 'medium', 'high']].reindex(index_order).stack().reset_index(level=1, drop=True)

	print(df4corel.corr())
	return df4corel.corr()


# path (glob) with submissions
glob_path = "../sub/*.csv"
comp_correlation(glob_path)

# choosen submissions to average
subs = ['sub51.csv', 'LtIsLit_XGB_brandon7.csv', 'stacker2_starter6.csv']
df_ensembled = AVG_subs(glob_path, subs)
# print(df_ensembled)
# df_ensembled.to_csv('../sub/AVGedSubs_sub51_brand7_stack6.csv', index=True)