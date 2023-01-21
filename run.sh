# activate venv and set Python path
source ~/projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/TeCBench/

python main.py \
  tasks=[fit,predict,eval] \
  trainer.precision=32 \
  model=BERTimbau \
  data=DIARIOS

python main.py \
  tasks=[fit,predict,eval] \
  trainer.precision=32 \
  model=BERT \
  data=DIARIOS

python main.py \
  tasks=[fit,predict,eval] \
  trainer.precision=16 \
  model=LaBSE \
  data=DIARIOS