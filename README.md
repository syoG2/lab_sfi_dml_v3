# Semantic Frame Induction with Deep Metric Learning (sfi-dml)

これは、EACL 2023のMainで採択された[Semantic Frame Induction with Deep Metric Learning](https://aclanthology.org/2023.eacl-main.134/)のリポジトリです。

## インストール

以下のコマンドを実行することで、必要なパッケージをインストールすることができます。
```sh
# Before installation, upgrade pip and setuptools.
$ pip install -U pip setuptools

# Install other dependencies.
$ pip install -r requirements.txt
```

## 使用方法

**`source/`にあるソースコードを実行するためのスクリプトは全て`scripts/`に格納されていおり、スクリプトの各ファイルは`(directory name)/(file name).sh`という名称となっています。また、出力結果等は、初期設定では`data/`に格納されるようになっています。**

### 1. 前処理 (`preprocessing/`)

このディレクトリでは、データの前処理を行います。
まず、`make_exemplars.py`にて、NLTKライブラリにあるFrameNet 1.7から用例文を抽出します。
次に、`apply_stanza.py`にて、Stanzaと呼ばれるテキスト解析ツールを用いて、データの整形を行います。
主に文字レベルで付与されたラベルを単語レベルに変換しています。

### 2. 意味フレーム推定実験 (`verb_clustering/`)

このディレクトリでは、意味フレーム推定実験を行います。
まず、`make_dataset.py`にて、この実験に対応するようにデータセットを作成します。
ここで、学習セットと、開発セット、テストセットの分割も行っています。

次に、`train_model.py`にて、BERTの深層距離学習によるfine-tuningを行います。
その後、`get_embedding.py`にて、BERTによる動詞の埋め込みを獲得します。
`find_best_params_*_clustering.py`を実行することで、1段階クラスタリングと2段階クラスタリングにおける最良パラメータを探索し、`perform_*_clustering.py`を実行することで、クラスタリングを実行します。
`evaluate_clustering.py`を実行することで、クラスタリングした結果の評価を行います。
交差検証を行っているため、`aggregate_scores.py`でスコアの集約を行えます。

`visualize_embeddings_*.py`で埋め込みの可視化を行うことができ、`evaluate_ranking_*.py`でランキングによる評価を行うことができます。


### 3. 学習事例数を変えた意味フレーム推定実験 (`verb_clustering_chaning_n_examples/`)

このディレクトリでは、学習事例数を変えた意味フレーム推定実験を行います。
`make_dataset.py`でデータセットを作成します。
それ以降の処理は、`verb_clustering/`と同様です。

## 引用

Please cite our paper if this source code is helpful in your work.

```bibtex
@inproceedings{yamada-etal-2023-semantic,
    title = "Semantic Frame Induction with Deep Metric Learning",
    author = "Yamada, Kosuke  and
      Sasano, Ryohei  and
      Takeda, Koichi",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    year = "2023",
    url = "https://aclanthology.org/2023.eacl-main.134",
    pages = "1833--1845",
}
```
