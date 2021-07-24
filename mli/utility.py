import pandas as pd


def get_coef(estimator, var_names):
    """特徴量名と回帰係数が対応したデータフレームを作成する"""
    
    # 切片含む回帰係数と特徴量の名前を抜き出してデータフレームにまとめる
    df = pd.DataFrame(
        data={"coef": [estimator.intercept_] + estimator.coef_.tolist()}, 
        index=["intercept"] + var_names
    )
    
    return df