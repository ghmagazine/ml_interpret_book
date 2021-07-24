import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def regression_metrics(estimator, X, y):
    """回帰精度の評価指標をまとめて返す関数"""

    # テストデータで予測
    y_pred = estimator.predict(X)

    # 評価指標をデータフレームにまとめる
    df = pd.DataFrame(
        data={
            "RMSE": [mean_squared_error(y, y_pred, squared=False)],
            "R2": [r2_score(y, y_pred)],
        }
    )

    return df