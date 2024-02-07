# activate venv and set Python path
source venv/bin/activate
export PYTHONPATH=PYTHONPATH:$pwd

python main.py \
  tasks=[fit] \
  trainer.precision=32 \
  model=BERT \
  data=20NG \
  data.folds=[0] \
  data.max_length=256 \
  data.batch_size=64




