import os
from tkinter import Y
import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    input_path = "/home/rupesh/projects/Melonoma_Deep_Learning/input/"
    df = pd.read_csv(os.path.join(input_path,"train.csv"))
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=10)
    for f, (t_,v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_,"kfold"] = f
    df.to_csv(os.path.join(input_path,"train_folds.csv"),index=False)