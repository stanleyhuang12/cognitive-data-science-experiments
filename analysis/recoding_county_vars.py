import pandas as pd
import os
import requests

url = "https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt"
response = requests.get(url)

lines = response.text.splitlines()
fips_lines = [line.strip() for line in lines if line.strip()[:5].isdigit()]

data = [(line[:5], line[6:]) for line in fips_lines]
fips_df = pd.DataFrame(data, columns=["FIPS", "County"])


fips_df
fips_geo_df = pd.read_csv('https://gist.githubusercontent.com/russellsamora/12be4f9f574e92413ea3f92ce1bc58e6/raw/3f18230058afd7431a5d394dab7eeb0aafd29d81/us_county_latlng.csv', dtype=object)

fips_geo_df.rename(columns={"fips_code": "FIPS", "name": "County"}, inplace=True)
fips_geo_df['County'] = fips_geo_df['County'].map(lambda x: x+" County")

pd.merge(left=fips_df, right=fips_geo_df, how='left', on='FIPS')

fips_geo_merge = fips_df.merge(fips_geo_df, how='inner', on='FIPS')

file_path = os.getcwd() + "/data" + "fips_geo_merge.csv"
fips_geo_merge.to_csv(file_path)

df = pd.read_csv('data/CCES2012_CSVFormat_NEW_countycode (1).csv')
df.head()

fips_geo_merge['FIPS'] = fips_geo_merge['FIPS'].astype(float)
fips_geo_merge.drop(columns=['County_x', 'County_y'], inplace=True)

df = df.merge(fips_geo_merge, left_on='countycode', right_on='FIPS', how='left')

df.drop(columns=['countycode', 'FIPS'], inplace=True)

df.dropna(axis=0, how='any', inplace=True)

df = df.reset_index(drop=True)

df.to_csv('data/CCES2012_CSVFormat_NEW_GEOCODED.csv', index=False)

