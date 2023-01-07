# activate venv and set Python path
source /projects/venvs/TeCBench/bin/activate
export PYTHONPATH=$PATHONPATH:/projects/TeCBench/

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=IMDB

python main.py \
  tasks=[fit,predict,eval] \
  model=CNN \
  data=YELP2015




