#!/bin/sh
# run.pyを複数回実行するためのスクリプト
# 実行コマンド
# bash run.sh --train ...

# python run.py --train --training_id=baseline --seed=1
# python run.py --train --training_id=baseline --seed=2
# python run.py --train --training_id=baseline --seed=3
# python run.py --train --training_id=baseline --seed=4
# python run.py --train --training_id=baseline --seed=5
# python run.py --train --agent_id=baseline-v2 --seed=1
# python run.py --train --agent_id=baseline-v2 --seed=2
# python run.py --train --agent_id=baseline-v2 --seed=3
# python run.py --train --agent_id=baseline-v2 --seed=4
# python run.py --train --agent_id=baseline-v2 --seed=5

# python run.py --train --agent_id=udr --seed=1 --algo=UDR
# python run.py --train --agent_id=udr --seed=2 --algo=UDR
# python run.py --train --agent_id=udr --seed=3 --algo=UDR
# python run.py --train --agent_id=udr --seed=4 --algo=UDR
# python run.py --train --agent_id=udr --seed=5 --algo=UDR

# python run.py --train --agent_id=cdr-v1_ --seed=1 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_ --seed=2 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_ --seed=3 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_ --seed=4 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_ --seed=5 --algo=CDR-v1

# python run.py --train --agent_id=cdr-v2_ --seed=1 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_ --seed=2 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_ --seed=3 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_ --seed=4 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_ --seed=5 --algo=CDR-v2

# python run.py --train --agent_id=lcdr-v1 --seed=1 --algo=LCL-v1
# python run.py --train --agent_id=lcdr-v1 --seed=2 --algo=LCL-v1
# python run.py --train --agent_id=lcdr-v1 --seed=3 --algo=LCL-v1
# python run.py --train --agent_id=lcdr-v1 --seed=4 --algo=LCL-v1
# python run.py --train --agent_id=lcdr-v1 --seed=5 --algo=LCL-v1

# python run.py --train --agent_id=lcdr-v2 --seed=1 --algo=LCL-v2
# python run.py --train --agent_id=lcdr-v2 --seed=2 --algo=LCL-v2
# python run.py --train --agent_id=lcdr-v2 --seed=3 --algo=LCL-v2
# python run.py --train --agent_id=lcdr-v2 --seed=4 --algo=LCL-v2
# python run.py --train --agent_id=lcdr-v2 --seed=5 --algo=LCL-v2

# python run.py --train --agent_id=acdr_prot --seed=1 --algo=acdr
# python run.py --train --agent_id=acdr_prot --seed=2 --algo=acdr
# python run.py --train --agent_id=acdr_prot --seed=3 --algo=acdr
# python run.py --train --agent_id=acdr_prot --seed=4 --algo=acdr
# python run.py --train --agent_id=acdr_prot --seed=5 --algo=acdr

python run.py --train --agent_id=acdr_prot_softmax_t=1 --seed=1 --algo=acdr
python run.py --train --agent_id=acdr_prot_softmax_t=1 --seed=2 --algo=acdr
python run.py --train --agent_id=acdr_prot_softmax_t=1 --seed=3 --algo=acdr
python run.py --train --agent_id=acdr_prot_softmax_t=1 --seed=4 --algo=acdr
python run.py --train --agent_id=acdr_prot_softmax_t=1 --seed=5 --algo=acdr

# python run.py --train --agent_id=acdr_prot_bayes --seed=1 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes --seed=2 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes --seed=3 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes --seed=4 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes --seed=5 --algo=acdrb