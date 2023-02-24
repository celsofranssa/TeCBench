# activate venv and set Python path
source ~/projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/TeCBench/

python main.py \
  tasks=[predict,eval] \
  trainer.precision=16 \
  model=LaBSE \
  data=DIARIOS