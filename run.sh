# activate venv and set Python path
source ~/projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/TeCBench/

python main.py \
  tasks=[fit,predict,eval] \
  trainer.max_epochs=100 \
  trainer.patience=30 \
  trainer.min_delta=0.05 \
  model=CNN \
  data=WEBKB




