# activate venv and set Python path
source ~/projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/TeCBench/

# AISOPOS
python main.py \
  tasks=[fit,predict,eval,tsne] \
  model=BERT \
  data=AISOPOS \
  data.num_workers=12