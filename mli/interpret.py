from __future__ import annotations  # 型ヒント用
from typing import Any  # 型ヒント用
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # matplotlibの日本語表示対応
from sklearn.metrics import mean_squared_error
from scipy.special import factorial


@dataclass
class PermutationFeatureImportance:
    """Permutation Feature Importance (PFI)
     
    Args:
        estimator: 全特徴量を用いた学習済みモデル
        X: 特徴量
        y: 目的変数
        var_names: 特徴量の名前
    """
    
    estimator: Any
    X: np.ndarray
    y: np.ndarray
    var_names: list[str]
        
    def __post_init__(self) -> None:
        # シャッフルなしの場合の予測精度
        # mean_squared_error()はsquared=TrueならMSE、squared=FalseならRMSE
        self.baseline = mean_squared_error(
            self.y, self.estimator.predict(self.X), squared=False
        )

    def _permutation_metrics(self, idx_to_permute: int) -> float:
        """ある特徴量の値をシャッフルしたときの予測精度を求める

        Args:
            idx_to_permute: シャッフルする特徴量のインデックス
        """

        # シャッフルする際に、元の特徴量が上書きされないよう用にコピーしておく
        X_permuted = self.X.copy()

        # 特徴量の値をシャッフルして予測
        X_permuted[:, idx_to_permute] = np.random.permutation(
            X_permuted[:, idx_to_permute]
        )
        y_pred = self.estimator.predict(X_permuted)

        return mean_squared_error(self.y, y_pred, squared=False)

    def permutation_feature_importance(self, n_shuffle: int = 10) -> None:
        """PFIを求める

        Args:
            n_shuffle: シャッフルの回数。多いほど値が安定する。デフォルトは10回
        """

        J = self.X.shape[1]  # 特徴量の数

        # J個の特徴量に対してPFIを求めたい
        # R回シャッフルを繰り返して平均をとることで値を安定させている
        metrics_permuted = [
            np.mean(
                [self._permutation_metrics(j) for r in range(n_shuffle)]
            )
            for j in range(J)
        ]

        # データフレームとしてまとめる
        # シャッフルでどのくらい予測精度が落ちるかは、
        # 差(difference)と比率(ratio)の2種類を用意する
        df_feature_importance = pd.DataFrame(
            data={
                "var_name": self.var_names,
                "baseline": self.baseline,
                "permutation": metrics_permuted,
                "difference": metrics_permuted - self.baseline,
                "ratio": metrics_permuted / self.baseline,
            }
        )

        self.feature_importance = df_feature_importance.sort_values(
            "permutation", ascending=False
        )

    def plot(self, importance_type: str = "difference") -> None:
        """PFIを可視化

        Args:
            importance_type: PFIを差(difference)と比率(ratio)のどちらで計算するか
        """

        fig, ax = plt.subplots()
        ax.barh(
            self.feature_importance["var_name"],
            self.feature_importance[importance_type],
            label=f"baseline: {self.baseline:.2f}",
        )
        ax.set(xlabel=importance_type, ylabel=None)
        ax.invert_yaxis() # 重要度が高い順に並び替える
        ax.legend(loc="lower right")
        fig.suptitle(f"Permutationによる特徴量の重要度({importance_type})")
        
        fig.show()
        

class GroupedPermutationFeatureImportance(PermutationFeatureImportance):
    """Grouped Permutation Feature Importance (GPFI)"""

    def _permutation_metrics(
        self,
        var_names_to_permute: list[str]
    ) -> float:
        """ある特徴量群の値をシャッフルしたときの予測精度を求める

        Args:
            var_names_to_permute: シャッフルする特徴量群の名前
        """

        # シャッフルする際に、元の特徴量が上書きされないよう用にコピーしておく
        X_permuted = self.X.copy()

        # 特徴量名をインデックスに変換
        idx_to_permute = [
            self.var_names.index(v) for v in var_names_to_permute
        ]

        # 特徴量群をまとめてシャッフルして予測
        X_permuted[:, idx_to_permute] = np.random.permutation(
            X_permuted[:, idx_to_permute]
        )
        y_pred = self.estimator.predict(X_permuted)

        return mean_squared_error(self.y, y_pred, squared=False)

    def permutation_feature_importance(
        self,
        var_groups: list[list[str]] | None = None,
        n_shuffle: int = 10
    ) -> None:
        """GPFIを求める

        Args:
            var_groups:
                グループ化された特徴量名のリスト。例：[['X0', 'X1'], ['X2']]
                Noneを指定すれば通常のPFIが計算される
            n_shuffle:
                シャッフルの回数。多いほど値が安定する。デフォルトは10回
        """

        # グループが指定されなかった場合は1つの特徴量を1グループとする。PFIと同じ。
        if var_groups is None:
            var_groups = [[j] for j in self.var_names]

        # グループごとに重要度を計算
        # R回シャッフルを繰り返して値を安定させている
        metrics_permuted = [
            np.mean(
                [self._permutation_metrics(j) for r in range(n_shuffle)]
            )
            for j in var_groups
        ]

        # データフレームとしてまとめる
        # シャッフルでどのくらい予測精度が落ちるかは、差と比率の2種類を用意する
        df_feature_importance = pd.DataFrame(
            data={
                "var_name": ["+".join(j) for j in var_groups],
                "baseline": self.baseline,
                "permutation": metrics_permuted,
                "difference": metrics_permuted - self.baseline,
                "ratio": metrics_permuted / self.baseline,
            }
        )

        self.feature_importance = df_feature_importance.sort_values(
            "permutation", ascending=False
        )


@dataclass
class PartialDependence:
    """Partial Dependence (PD)

    Args:
        estimator: 学習済みモデル
        X: 特徴量
        var_names: 特徴量の名前
    """
    
    estimator: Any
    X: np.ndarray
    var_names: list[str]
    
    def _counterfactual_prediction(
        self,
        idx_to_replace: int,
        value_to_replace: float
    ) -> np.ndarray:
        """ある特徴量の値を置き換えたときの予測値を求める

        Args:
            idx_to_replace: 値を置き換える特徴量のインデックス
            value_to_replace: 置き換える値
        """

        # 特徴量の値を置き換える際、元データが上書きされないようコピー
        X_replaced = self.X.copy()

        # 特徴量の値を置き換えて予測
        X_replaced[:, idx_to_replace] = value_to_replace
        y_pred = self.estimator.predict(X_replaced)

        return y_pred

    def partial_dependence(
        self,
        var_name: str,
        n_grid: int = 50
    ) -> None:
        """PDを求める

        Args:
            var_name: 
                PDを計算したい特徴量の名前
            n_grid: 
                グリッドを何分割するか
                細かすぎると値が荒れるが、粗すぎるとうまく関係を捉えられない
                デフォルトは50
        """
        
        # 可視化の際に用いるのでターゲットの変数名を保存
        self.target_var_name = var_name  
        # 変数名に対応するインデックスをもってくる
        var_index = self.var_names.index(var_name)

        # ターゲットの変数を、取りうる値の最大値から最小値まで動かせるようにする
        value_range = np.linspace(
            self.X[:, var_index].min(), 
            self.X[:, var_index].max(), 
            num=n_grid
        )

        # インスタンスごとのモデルの予測値を平均
        average_prediction = np.array([
            self._counterfactual_prediction(var_index, x).mean()
            for x in value_range
        ])

        # データフレームとしてまとめる
        self.df_partial_dependence = pd.DataFrame(
            data={var_name: value_range, "avg_pred": average_prediction}
        )

    def plot(self, ylim: list[float] | None = None) -> None:
        """PDを可視化

        Args:
            ylim: 
                Y軸の範囲
                特に指定しなければavg_predictionの範囲となる
                異なる特徴量のPDを比較したいときなどに指定する
        """

        fig, ax = plt.subplots()
        ax.plot(
            self.df_partial_dependence[self.target_var_name],
            self.df_partial_dependence["avg_pred"],
        )
        ax.set(
            xlabel=self.target_var_name,
            ylabel="Average Prediction",
            ylim=ylim
        )
        fig.suptitle(f"Partial Dependence Plot ({self.target_var_name})")
        
        fig.show()

        
class IndividualConditionalExpectation(PartialDependence):
    """Indivudual Conditional Expectation"""

    def individual_conditional_expectation(
        self, 
        var_name: str, 
        ids_to_compute: list[int], 
        n_grid: int = 50
    ) -> None:
        """ICEを求める

        Args:
            var_name:
                ICEを計算したい変数名
            ids_to_compute:
                ICEを計算したいインスタンスのリスト
            n_grid: 
                グリッドを何分割するか
                細かすぎると値が荒れるが、粗すぎるとうまく関係をとらえられない
                デフォルトは50
        """
        
        # 可視化の際に用いるのでターゲットの変数名を保存
        self.target_var_name = var_name 
        # 変数名に対応するインデックスをもってくる
        var_index = self.var_names.index(var_name)

        # ターゲットの変数を、取りうる値の最大値から最小値まで動かせるようにする
        value_range = np.linspace(
            self.X[:, var_index].min(),
            self.X[:, var_index].max(),
            num=n_grid
        )

        # インスタンスごとのモデルの予測値
        # PDの_counterfactual_prediction()をそのまま使っているので
        # 全データに対して予測してからids_to_computeに絞り込んでいるが
        # 本当は絞り込んでから予測をしたほうが速い
        individual_prediction = np.array([
            self._counterfactual_prediction(var_index, x)[ids_to_compute]
            for x in value_range
        ])

        # ICEをデータフレームとしてまとめる
        self.df_ice = (
            # ICEの値
            pd.DataFrame(data=individual_prediction, columns=ids_to_compute)
            # ICEで用いた特徴量の値。特徴量名を列名としている
            .assign(**{var_name: value_range})
            # 縦持ちに変換して完成
            .melt(id_vars=var_name, var_name="instance", value_name="ice")
        )

        # ICEを計算したインスタンスについての情報も保存しておく
        # 可視化の際に実際の特徴量の値とその予測値をプロットするために用いる
        self.df_instance = (
            # インスタンスの特徴量の値
            pd.DataFrame(
                data=self.X[ids_to_compute],
                columns=self.var_names
            )
            # インスタンスに対する予測値
            .assign(
                instance=ids_to_compute,
                prediction=self.estimator.predict(self.X[ids_to_compute]),
            )
            # 並べ替え
            .loc[:, ["instance", "prediction"] + self.var_names]
        )

    def plot(self, ylim: list[float] | None = None) -> None:
        """ICEを可視化

        Args:
            ylim: Y軸の範囲。特に指定しなければiceの範囲となる。
        """

        fig, ax = plt.subplots()
        # ICEの線
        sns.lineplot(
            self.target_var_name,
            "ice",
            units="instance",
            data=self.df_ice,
            lw=0.8,
            alpha=0.5,
            estimator=None,
            zorder=1,  # zorderを指定することで、線が背面、点が前面にくるようにする
            ax=ax,
        )
        # インスタンスからの実際の予測値を点でプロットしておく
        sns.scatterplot(
            self.target_var_name, 
            "prediction", 
            data=self.df_instance, 
            zorder=2, 
            ax=ax
        )
        ax.set(xlabel=self.target_var_name, ylabel="Prediction", ylim=ylim)
        fig.suptitle(
            f"Individual Conditional Expectation({self.target_var_name})"
        )
        
        fig.show()
        
        


@dataclass
class ShapleyAdditiveExplanations:
    """SHapley Additive exPlanations
    
    Args:
        estimator: 学習済みモデル
        X: SHAPの計算に使う特徴量
        var_names: 特徴量の名前
    """
    
    estimator: Any
    X: np.ndarray
    var_names: list[str]
        
    def __post_init__(self) -> None:
        # ベースラインとしての平均的な予測値
        self.baseline = self.estimator.predict(self.X).mean()

        # 特徴量の総数
        self.J = self.X.shape[1]

        # あり得るすべての特徴量の組み合わせ
        self.subsets = [
            s
            for j in range(self.J + 1)
            for s in combinations(range(self.J), j)
        ]

    def _get_expected_value(self, subset: tuple[int, ...]) -> np.ndarray:
        """特徴量の組み合わせを指定するとその特徴量が場合の予測値を計算

        Args:
            subset: 特徴量の組み合わせ
        """
        
        _X = self.X.copy()  # 元のデータが上書きされないように

        # 特徴量がある場合は上書き。なければそのまま。
        if subset is not None:
            # 元がtupleなのでリストにしないとインデックスとして使えない
            _s = list(subset)
            _X[:, _s] = _X[self.i, _s]

        return self.estimator.predict(_X).mean()

    def _calc_weighted_marginal_contribution(
        self,
        j: int,
        s_union_j: tuple[int, ...]
    ) -> float:
        """限界貢献度x組み合わせ出現回数を求める

        Args:
            j: 限界貢献度を計算したい特徴量のインデックス
            s_union_j: jを含む特徴量の組み合わせ
        """
        
        # 特徴量jがない場合の組み合わせ
        s = tuple(set(s_union_j) - set([j]))

        # 組み合わせの数
        S = len(s)

        # 組み合わせの出現回数
        # ここでfactorial(self.J)で割ってしまうと丸め誤差が出てるので、あとで割る
        weight = factorial(S) * factorial(self.J - S - 1)

        # 限界貢献度
        marginal_contribution = (
            self.expected_values[s_union_j] - self.expected_values[s]
        )

        return weight * marginal_contribution

    def shapley_additive_explanations(self, id_to_compute: int) -> None:
        """SHAP値を求める

        Args:
            id_to_compute: SHAPを計算したいインスタンス
        """

        # SHAPを計算したいインスタンス
        self.i = id_to_compute

        # すべての組み合わせに対して予測値を計算
        # 先に計算しておくことで同じ予測を繰り返さずに済む
        self.expected_values = {
            s: self._get_expected_value(s) for s in self.subsets
        }

        # ひとつひとつの特徴量に対するSHAP値を計算
        shap_values = np.zeros(self.J)
        for j in range(self.J):
            # 限界貢献度の加重平均を求める
            # 特徴量jが含まれる組み合わせを全部もってきて
            # 特徴量jがない場合の予測値との差分を見る
            shap_values[j] = np.sum([
                self._calc_weighted_marginal_contribution(j, s_union_j)
                for s_union_j in self.subsets
                if j in s_union_j
            ]) / factorial(self.J)
        
        # データフレームとしてまとめる
        self.df_shap = pd.DataFrame(
            data={
                "var_name": self.var_names,
                "feature_value": self.X[id_to_compute],
                "shap_value": shap_values,
            }
        )

    def plot(self) -> None:
        """SHAPを可視化"""
        
        # 下のデータフレームを書き換えないようコピー
        df = self.df_shap.copy()
        
        # グラフ用のラベルを作成
        df['label'] = [
            f"{x} = {y:.2f}" for x, y in zip(df.var_name, df.feature_value)
        ]
        
        # SHAP値が高い順に並べ替え
        df = df.sort_values("shap_value").reset_index(drop=True)
        
        # 全特徴量の値がときの予測値
        predicted_value = self.expected_values[self.subsets[-1]]
        
        # 棒グラフを可視化
        fig, ax = plt.subplots()
        ax.barh(df.label, df.shap_value)
        ax.set(xlabel="SHAP値", ylabel=None)
        fig.suptitle(f"SHAP値 \n(Baseline: {self.baseline:.2f}, Prediction: {predicted_value:.2f}, Difference: {predicted_value - self.baseline:.2f})")

        fig.show()