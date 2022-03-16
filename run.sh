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

# python run.py --train --agent_id=acdr_prot_softmax_t=1 --seed=1 --algo=acdr
# python run.py --train --agent_id=acdr_prot_softmax_t=1 --seed=2 --algo=acdr
# python run.py --train --agent_id=acdr_prot_softmax_t=1 --seed=3 --algo=acdr
# python run.py --train --agent_id=acdr_prot_softmax_t=1 --seed=4 --algo=acdr
# python run.py --train --agent_id=acdr_prot_softmax_t=1 --seed=5 --algo=acdr

# python run.py --train --agent_id=acdr_prot_bayes --seed=1 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes --seed=2 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes --seed=3 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes --seed=4 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes --seed=5 --algo=acdrb

# ・acdr_softmaxはminmax normやってないやつもやる
# ・acdrbは4e6もやる

# python run.py --train --agent_id=acdr_prot_bayes_4e6 --seed=1 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes_4e6 --seed=2 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes_4e6 --seed=3 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes_4e6 --seed=4 --algo=acdrb
# python run.py --train --agent_id=acdr_prot_bayes_4e6 --seed=5 --algo=acdrb

# 4e6
# python run.py --train --agent_id=baseline_4e6 --seed=1 --algo=Baseline
# python run.py --train --agent_id=baseline_4e6 --seed=2 --algo=Baseline
# python run.py --train --agent_id=baseline_4e6 --seed=3 --algo=Baseline
# python run.py --train --agent_id=baseline_4e6 --seed=4 --algo=Baseline
# python run.py --train --agent_id=baseline_4e6 --seed=5 --algo=Baseline

# 16e6
# python run.py --train --agent_id=cdr-v1_16e6 --seed=1 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_16e6 --seed=2 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_16e6 --seed=3 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_16e6 --seed=4 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_16e6 --seed=5 --algo=CDR-v1

# python run.py --train --agent_id=cdr-v2_16e6 --seed=1 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_16e6 --seed=2 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_16e6 --seed=3 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_16e6 --seed=4 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_16e6 --seed=5 --algo=CDR-v2

# python run.py --train --agent_id=lcdr-v2_4e6 --seed=1 --algo=LCL-v2
# python run.py --train --agent_id=lcdr-v2_4e6 --seed=2 --algo=LCL-v2
# python run.py --train --agent_id=lcdr-v2_4e6 --seed=3 --algo=LCL-v2
# python run.py --train --agent_id=lcdr-v2_4e6 --seed=4 --algo=LCL-v2
# python run.py --train --agent_id=lcdr-v2_4e6 --seed=5 --algo=LCL-v2

# buffer_size=50
# python run.py --train --agent_id=cdr-v1_bf50 --seed=1 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50 --seed=2 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50 --seed=3 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50 --seed=4 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50 --seed=5 --algo=CDR-v1

# python run.py --train --agent_id=cdr-v2_bf50 --seed=1 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50 --seed=2 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50 --seed=3 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50 --seed=4 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50 --seed=5 --algo=CDR-v2

# buffer_size=10
# python run.py --train --agent_id=cdr-v1_bf10 --seed=1 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf10 --seed=2 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf10 --seed=3 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf10 --seed=4 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf10 --seed=5 --algo=CDR-v1

# python run.py --train --agent_id=cdr-v2_bf10 --seed=1 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf10 --seed=2 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf10 --seed=3 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf10 --seed=4 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf10 --seed=5 --algo=CDR-v2

# buffer size: 50, time steps: 8e6
# python run.py --train --agent_id=cdr-v1_bf50_8e6 --seed=1 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50_8e6 --seed=2 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50_8e6 --seed=3 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50_8e6 --seed=4 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50_8e6 --seed=5 --algo=CDR-v1

# python run.py --train --agent_id=cdr-v2_bf50_8e6 --seed=1 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50_8e6 --seed=2 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50_8e6 --seed=3 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50_8e6 --seed=4 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50_8e6 --seed=5 --algo=CDR-v2

# buffer size: 50, time steps: 16e6
# python run.py --train --agent_id=cdr-v1_bf50_16e6 --seed=1 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50_16e6 --seed=2 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50_16e6 --seed=3 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50_16e6 --seed=4 --algo=CDR-v1
# python run.py --train --agent_id=cdr-v1_bf50_16e6 --seed=5 --algo=CDR-v1

# python run.py --train --agent_id=cdr-v2_bf50_16e6 --seed=1 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50_16e6 --seed=2 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50_16e6 --seed=3 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50_16e6 --seed=4 --algo=CDR-v2
# python run.py --train --agent_id=cdr-v2_bf50_16e6 --seed=5 --algo=CDR-v2

# ACDRB, maximize_flag:Trueにしたもの（capabilityを最大化するkを探索する手法，今まではFalseだった）
# python run.py --train --agent_id=acdrb_4e6_maximizeFlag=True --seed=1 --algo=acdrb
# python run.py --train --agent_id=acdrb_4e6_maximizeFlag=True --seed=2 --algo=acdrb
# python run.py --train --agent_id=acdrb_4e6_maximizeFlag=True --seed=3 --algo=acdrb
# python run.py --train --agent_id=acdrb_4e6_maximizeFlag=True --seed=4 --algo=acdrb
# python run.py --train --agent_id=acdrb_4e6_maximizeFlag=True --seed=5 --algo=acdrb

# kを固定したもの
# python run.py --train --agent_id=k00 --seed=1 --algo=Baseline --baseline_k=0.0
# python run.py --train --agent_id=k00 --seed=2 --algo=Baseline --baseline_k=0.0
# python run.py --train --agent_id=k00 --seed=3 --algo=Baseline --baseline_k=0.0
# python run.py --train --agent_id=k00 --seed=4 --algo=Baseline --baseline_k=0.0
# python run.py --train --agent_id=k00 --seed=5 --algo=Baseline --baseline_k=0.0

# python run.py --train --agent_id=k04 --seed=1 --algo=Baseline --baseline_k=0.4
# python run.py --train --agent_id=k04 --seed=2 --algo=Baseline --baseline_k=0.4
# python run.py --train --agent_id=k04 --seed=3 --algo=Baseline --baseline_k=0.4
# python run.py --train --agent_id=k04 --seed=4 --algo=Baseline --baseline_k=0.4
# python run.py --train --agent_id=k04 --seed=5 --algo=Baseline --baseline_k=0.4

# python run.py --train --agent_id=k08 --seed=1 --algo=Baseline --baseline_k=0.8
# python run.py --train --agent_id=k08 --seed=2 --algo=Baseline --baseline_k=0.8
# python run.py --train --agent_id=k08 --seed=3 --algo=Baseline --baseline_k=0.8
# python run.py --train --agent_id=k08 --seed=4 --algo=Baseline --baseline_k=0.8
# python run.py --train --agent_id=k08 --seed=5 --algo=Baseline --baseline_k=0.8

# python run.py --train --agent_id=k12 --seed=1 --algo=Baseline --baseline_k=1.2
# python run.py --train --agent_id=k12 --seed=2 --algo=Baseline --baseline_k=1.2
# python run.py --train --agent_id=k12 --seed=3 --algo=Baseline --baseline_k=1.2
# python run.py --train --agent_id=k12 --seed=4 --algo=Baseline --baseline_k=1.2
# python run.py --train --agent_id=k12 --seed=5 --algo=Baseline --baseline_k=1.2

python run.py --train --agent_id=k08_2 --seed=1 --algo=Baseline --baseline_k=0.8
python run.py --train --agent_id=k08_2 --seed=2 --algo=Baseline --baseline_k=0.8
python run.py --train --agent_id=k08_2 --seed=3 --algo=Baseline --baseline_k=0.8
python run.py --train --agent_id=k08_2 --seed=4 --algo=Baseline --baseline_k=0.8
python run.py --train --agent_id=k08_2 --seed=5 --algo=Baseline --baseline_k=0.8