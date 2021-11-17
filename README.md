# ouqu-tp

英語対応してなくてごめん
いろいろ暫定です

# 使い方

これは、QASMをqulacsで実行したり、実機で可能なようにうまく回路を変形するライブラリです。


CNOTの制約とQASMファイルから、実機で可能なQASMファイルを作るtrance.shと、

QASMファイルを受け取り、量子状態を得た後、shotの回数だけ実行するsimulate.sh

の二つの機能があります。

入出力例として、サンプルの各ファイルが、すでにdataフォルダに入っています。参考にしてください。

注意点:このトランスパイラは、グローバル位相を完全に無視します。

## 必要な環境
python

qulacs(普通のでも、osakaでも可)

staq

が必要です。
windows以外での動作は可能かわかりません。


## trance.sh
CNOTの制約とQASMファイルから、実機で可能なQASMファイルを作ります

CNOTの制約はdata/CNOT_net.txtに書いてください

入力QASMファイルは、data/input.qasmに書いて下さい

出力QASMファイルは、data/output.qasmにあります

(data/cpl.qasm　は、中間表現です。QASM形式で、　UゲートとCNOTだけで構成されます)

### 初期状態にあるdata/CNOT_net.txtを例にした説明

```
1行目：名前 なんでもいい
2行目:qubit数
3行目:connected数
以降、connected数行:  control,tergetの順

例:

test
9
12
0,1
1,2
3,4
4,5
6,7
7,8
0,3
3,6
1,4
4,7
2,5
5,8


これは、
0-1-2
| | |
3-4-5
| | |
6-7-8
```
細かい仕様

3行目:connected数 は実は使っていなくて、　EOFまで読んでる
control,tergetのところに END というアルファベット3文字の入力が来ると、終了になる

## simulate.sh
QASMファイルを受け取り、量子状態を得た後、shotの回数だけ実行します。

とりあえず回数=100

入力QASMファイルは、data/input.qasmに書いて下さい

得られた結果は、data/kekka.txtにあります。

kekkaの各行が量子状態に対応していて、　一番右が0番のbitです。

