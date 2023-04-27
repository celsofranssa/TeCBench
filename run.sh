# activate venv and set Python path
source ~/projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/TeCBench/

python main.py \
  tasks=[predict] \
  trainer.precision=32 \
  model=BERTimbau \
  data=RECLAMEAQUI \
  data.folds=[0] \
  data.max_length=256 \
  data.batch_size=64




