# activate venv and set Python path
source ~/projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/TeCBench/

# WEBKB
python main.py \
  tasks=[fit,predict,eval] \
  model=BERT \
  data=WEBKB \
  data.folds=[0,1,2] \
  data.batch_size=64 \
  data.num_workers=12