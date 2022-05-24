# Adaptive Curriculum Dynamics Randomization
Add automatic curriculum function.

![ant](docs/ant.png)

# Set up
```
conda create -n acdr python=3.8
conda activate acdr
pip install -r requirements.txt
```

# Run
```
# How to run ACDR_easy2hard algorithm
python run.py --train --agent_id=baseline --seed=1 --algo=CDR-v1

# How to run ACDR_hard2easy algorithm
python run.py --train --agent_id=baseline --seed=1 --algo=CDR-v2
```
