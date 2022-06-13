# ml_interpret_book

「機械学習を解釈する技術〜予測力と説明力を両立する実践テクニック」のサンプルコードです。


## 書籍情報

2021年8月4日紙版発売<br>
森下光之助　著<br>
A5判／256ページ<br>
定価2,948円（本体2,680円＋税10%）<br>
ISBN 978-4-297-12226-3

## 出版社サポートサイト

https://gihyo.jp/book/2021/978-4-297-12226-3

## 動作環境

Pythonのバージョンは3.8を、パッケージの管理にはpoetryを利用しています。
サンプルコードを`git clone`でダウンロードし、フォルダ直下で`poetry install`を実行すると、本書と同じ環境を構築することが出来ます。

 ```
 git clone https://github.com/ghmagazine/ml_interpret_book.git
 cd ml_interpret_book
 poetry env use python3.8
 poetry install
 ```

本書の環境で利用しているPython及びパッケージのバージョンは以下になります。

```
[tool.poetry.dependencies]
python = "^3.8"
jupyter = "^1.0.0"
jupyterlab = "^3.0.14"
numpy = "^1.20.2"
pandas = "^1.2.4"
matplotlib = "^3.4.1"
seaborn = "^0.11.1"
scikit-learn = "^0.24.1"
shap = "^0.39.0"
japanize-matplotlib = "^1.1.3"
statsmodels = "^0.12.2"
```
