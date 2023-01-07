# activate venv and set Python path
source ~/projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/TeCBench/

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=BOOKS

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=DBLP

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=AGNEWS

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=SOGOU




