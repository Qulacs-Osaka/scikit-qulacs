# Contributing scikit-qulacs

## Start coding
プロジェクトをセットアップします．
1. リポジトリをクローンします．
```bash
git clone git@github.com:Qulacs-Osaka/scikit-qulacs.git
cd scikit-qulacs
```

2. 依存ライブラリや開発用ツールをインストールします．
```bash
pip install -r requirements-dev.txt
# This installs dependencies and creates a symbolic link to this directory in 
# the site-packages directory.
make install
```

次は毎回のコードの編集からマージまでの流れについてです．
3. `main` と同期します(初回は不要)
```bash
git switch main
git pull # Shorthand for `git pull origin main`
```

4. ブランチを切ります． 開発する内容をおおまかに説明するブランチ名にします．
```bash
git switch -c 99-wonderful-model
```

5. コミットの前にフォーマットとリント，テストを実行します． 
```bash
make check
make test
```

コードフォーマットといくつかのリントエラーは `make fix` で修正できます．
それ以外のリントエラーはエラーメッセージに沿って手で直す必要があります．

6. 編集したファイルをコミットしてプッシュします．
```bash
git add MODIFIED_FILE
git commit
# For the first push in the branch
git push -u origin 99-wonderful-model
# After first push
git push
```

7. そのブランチで開発すべき機能ができたらプルリクエスト(PR)を出します． 基本的に他の人にレビューを受けるようにします． ただし軽微な変更の場合はレビューをスキップしても問題ない場合もあります．

## Testing
新しい機能を開発したときにはテストを書くようにします． このテストは基本的に自動で実行されるものです．

1. `tests` ディレクトリに `test_*.py` というファイルを作ります． テストの内容を大まかに表すファイル名をつけます．
2. そのファイルの中に `test_` で始まる関数を作ります． そこに実装した機能が満たすべき条件を書きます． たとえば和を計算する関数のテストは以下のようになります．
```python
from skqulacs import add # This function does not exist in the module.

def test_add():
    assert 3 == add(1, 2)
```

3. テストを実行します．
```bash
make test
```
アサーションに失敗すると赤色で内容が表示されます． それが表示されなければ全てのテストに通っています．

特定のファイルにあるテストだけを実行したいときがあると思います．
そういうときはテストしたいファイルとともに `make` を実行してください．
```
make tests/test_circuit.py tests/test_qnn_regressor.py
```

テストには `pytest` を使用しています． 詳しい使い方は[ドキュメント](https://docs.pytest.org/en/6.2.x/)を参照してください．

## CI
GitHub Actions で CI を実行します． 基本的に CI に通らないとマージできません．
CI ではテストとコードフォーマット，リンタのエラーがないことの確認をします．
CI の目的には次のようなものがあります．
* コードが正常に確認していることを全体で共有する
* 手元では気づかなかったエラーを発見する
* コードがフォーマットされておりリンタのエラーがないことを強制することで，余計な diff が生まれないようにする

## Installation
`setup-tools` を使って　site-packages に skqulacs をインストールすることができます．
`make install` はこのディレクトリへのシンボリックリンクを作成するだけですが，この方法では完全なパッケージをビルドします．

まず，`build` をインストールします．
```bash
pip install build
```
ビルドしてインストールします．
```bash
python -m build
# This file name might be different among environments.
pip install dist/scikit_qulacs-0.0.1-py3-none-any.whl
```

## Documentation
このリポジトリのドキュメントには API ドキュメントと Jupyter Notebook 形式のチュートリアルがあります．

このライブラリの API ドキュメントはここから参照できます: https://qulacs-osaka.github.io/scikit-qulacs/index.html
このドキュメントは `main` ブランチにプッシュ(PR からのマージ)したときにビルドされ，デプロイされます．

### Build document
まず `doc` に移動します．
First, move into `doc`.
```bash
cd doc
```

以下のコマンドでドキュメントのビルドに必要な依存ライブラリをインストールできます．
```bash
pip install -r requirements-doc.txt
```

そして以下のコマンドを実行してください．
```bash
make html
```

`doc/build/html` に HTML ファイルなどのビルド成果物が入っています．

### Create Page from jupyter notebook
jupyter notebook からページを作ることができます．ライブラリの使用例などを書くのに便利です．
1. `doc/source/notebooks` に ipynb ファイルを作って編集する(0_example.ipynb とする)
2. `doc/source/notebooks/index.rst` にそのファイル名を拡張子なしで追記する(*)
3. `make html -C doc` を実行すると HTML が生成されるのでブラウザなどで開く

(*) `index.rst` の一部を抜粋します:
```
Notebooks
---------

.. toctree::

   0_tutorial
```

LaTeX や画像なども表示できます．