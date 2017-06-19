import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('lyrics_11_artists.csv', sep=',', index_col=0)
for index, row in tqdm(df.iterrows()):
    with open("songs/{}---{}.txt".format(row['artist'],
              row['title'].replace('/', '')), "a+") as myfile:
        myfile.write("{}\n".format(row['line']))
