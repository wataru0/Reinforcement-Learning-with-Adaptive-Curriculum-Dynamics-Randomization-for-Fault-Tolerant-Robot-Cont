#!/bin/sh
# run.pyを複数回実行するためのスクリプト
# 実行コマンド
# bash run.sh --train ...

# python run.py --train --training_id=baseline --seed=1
# python run.py --train --training_id=baseline --seed=2
# python run.py --train --training_id=baseline --seed=3
# python run.py --train --training_id=baseline --seed=4
# python run.py --train --training_id=baseline --seed=5
python run.py --train --training_id=baseline-v2 --seed=1
python run.py --train --training_id=baseline-v2 --seed=2
python run.py --train --training_id=baseline-v2 --seed=3
python run.py --train --training_id=baseline-v2 --seed=4
python run.py --train --training_id=baseline-v2 --seed=5