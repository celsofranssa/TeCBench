# activate venv and set Python path
source ~/projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/TeCBench/

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=WEBKB

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=ACM

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=AISOPOS

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=PANGMOVIE

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=REUTERS

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=SST2

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=VADERMOVIE

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=VADERNYT

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=WEBKB

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=YELP2L





