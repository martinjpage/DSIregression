# -*- coding: utf-8 -*-


def transform_targets(y_v):
  df = y_v.copy()
  columns = df.columns
  for index, row in df.iterrows():
    cols = [cname for cname in df.columns if row[cname]==1]
    if (cols != []):
      dfupdate = row[cols]
      dfupdate1=dfupdate.sample(1, random_state = 3)
      dfupdate1[0]=0
      dfupdate.update(dfupdate1)
      row.update(dfupdate)  
      row = pd.DataFrame(row.values.reshape(1,-1))
      row.columns = columns
      df.loc[index]=row.values
  return df
