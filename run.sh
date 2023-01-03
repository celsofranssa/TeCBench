# activate venv and set Python path
source ~/projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/TeCBench/

python main.py \
  tasks=[fit,predict,eval] \
  trainer.precision=32 \
  model=LaBSE \
  data=DIARIOS \
  data.folds=[0]




