# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:34:59 2017

@author: Pablo

https://www.kaggle.com/beyondbeneath/talkingdata-mobile-user-demographics/geolocation-visualisations

"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

df_events = pd.read_csv("data/events.csv", dtype={'device_id': np.str})
# Set up plot
# random subsample
df_events_sample = df_events.sample(n=100000)

plt.figure(1, figsize=(12,6))

# Mercator of World
m1 = Basemap(projection='merc',
             llcrnrlat=-60,
             urcrnrlat=65,
             llcrnrlon=-180,
             urcrnrlon=180,
             lat_ts=0,
             resolution='c')

m1.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes
m1.drawmapboundary(fill_color='#000000')                # black background
m1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders

# Plot the data
mxy = m1(df_events_sample["longitude"].tolist(), df_events_sample["latitude"].tolist())
m1.scatter(mxy[0], mxy[1], s=3, c="#1292db", lw=0, alpha=1, zorder=5)

plt.title("Global view of events")
plt.show()