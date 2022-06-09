# activate venv and set Python path
source ~/projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/TeCBench/

python main.py \
  tasks=[fit] \
  model=BERT \
  data=20NG \
  data.batch_size=32 \
  data.num_workers=12 \
  trainer.max_epochs=1 \
  trainer.gpus=0

# AISOPOS
python main.py \
  tasks=[fit] \
  model=BERT \
  data=AISOPOS \
  data.num_workers=12

python main.py \
  tasks=[fit] \
  model=BERT \
  data=20NG \
  data.batch_size=32 \
  data.num_workers=12 \
  trainer.max_epochs=1 \
  trainer.gpus=0