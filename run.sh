# activate venv and set Python path
source ~/projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/TeCBench/

python main.py \
  tasks=[predict,eval] \
  model=CNN \
  data=AISOPOS \
  data.folds=[0]




