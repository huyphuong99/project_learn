import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=14)


def read_csv(path: str):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def read_csv_standing(path: str):
    df = pd.read_csv(path)
    return df


def first_method(df):
    won_last = defaultdict(int)
    df["HomeLastWin"] = False
    df["VisitorLastWin"] = False
    for idx, row in tqdm(df.sort_values("Date").iterrows()):
        home_team = row["HomeTeam"]
        visitor_team = row["VisitorTeam"]
        row["HomeLastWin"] = won_last[home_team]
        row["VisitorLastWin"] = won_last[visitor_team]
        won_last[home_team] = row["HomeWin"]
        won_last[visitor_team] = not row["HomeWin"]
        df.iloc[idx] = row
    return df


def second_method(df_set, standing):
    df_set["HomeTeamRanksHighter"] = 0
    for idx, row in tqdm(df_set.iterrows()):
        home_team = row["HomeTeam"]
        visitor_team = row["VisitorTeam"]
        try:
            home_rank = standing[standing["Team"] == home_team]["Rk"].values[0]
            visitor_rank = standing[standing["Team"] == visitor_team]["Rk"].values[0]
            row["HomeTeamRanksHighter"] = int(home_rank > visitor_rank)
            df_set.iloc[idx] = row
        except:
            print(home_team, visitor_team)
    return df_set


def third_method(df):
    won_last = defaultdict(int)
    df["HomeTeamWonLast"] = 0
    for idx, row in tqdm(df.iterrows()):
        home_team = row["HomeTeam"]
        visitor_team = row["VisitorTeam"]
        teams = tuple(sorted([home_team, visitor_team]))
        row["HomeTeamWonLast"] = 1 if won_last[teams] == row["HomeTeam"] else 0
        df.iloc[idx] = row
        winner = row["HomeTeam"] if row["HomeWin"] else row["VisitorTeam"]
        won_last[teams] = winner
    return df


def decision_tree(X, y_true):
    scores = cross_val_score(clf, X, y_true, scoring="accuracy")
    return scores


if __name__ == "__main__":
    path = "./data/basket_ball/*"
    list_path = glob.glob(path)
    list_df = []
    for p in tqdm(list_path):
        d = read_csv(p)
        list_df.append(d)
        # break
    df = pd.concat(list_df, ignore_index=True)
    columns = ["Date", "Start", "VisitorTeam", "VisitorPts", "HomeTeam", "HomePts", "ScoreType", "OT?", "Attend.",
               "Notes"]
    df.columns = columns

    df["HomeWin"] = df["HomePts"] > df["VisitorPts"]
    df = first_method(df)

    y_true = df["HomeWin"].values
    X1 = df[["HomeLastWin", "VisitorLastWin"]].values
    scores_1 = decision_tree(X1, y_true)
    print(np.mean(scores_1))

    standing_path = "data/standing/standing1.csv"
    standing_file = read_csv_standing(standing_path)
    df1 = second_method(df, standing_file)
    X2 = df1[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHighter"]].values

    scores_2 = decision_tree(X2, y_true)
    print(np.mean(scores_2))

    df2 = third_method(df1)
    X3 = df2[["HomeTeamRanksHighter", "HomeTeamWonLast"]].values
    scores_3 = decision_tree(X3, y_true)
    print(np.mean(scores_3))
