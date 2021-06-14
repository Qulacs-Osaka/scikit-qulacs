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
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

次は毎回のコードの編集からマージまでの流れについてです．
3. `main` と同期します(初回は不要)
```bash
git switch main
git pull main
```

4. ブランチを切ります． 開発する内容をおおまかに説明するブランチ名にします．
```bash
git switch -c 99-wonderful-model
```

5. コミットの前にテストとフォーマットを実行します．
```bash
make format
make test
```

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

テストには `pytest` を使用しています． 詳しい使い方は[ドキュメント](https://docs.pytest.org/en/6.2.x/)を参照してください．

## CI
GitHub Actions で CI を実行します． 基本的に CI に通らないとマージできません．
CI ではテストとコードフォーマットの確認をします．
CI の目的には次のようなものがあります．
* コードが正常に確認していることを全体で共有する
* 手元では気づかなかったエラーを発見する
* コードがフォーマットされていることを強制することで余計な diff が生まれないようにする