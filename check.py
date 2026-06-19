import pandas as pd

df = pd.read_csv("C:/Users/chand/Downloads/archive/gym_members_exercise_tracking.csv")

print(df.head())
print(df.info())
print(df.columns)