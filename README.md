# Text Classification Benchmark based-on Fine-tuned Transformer

### 1. Quick Start

```shell script
# clone the project 
git clone git@github.com:celsofranssa/xCoFormer.git

# change directory to project folder
cd xCoFormer/

# Create a new virtual environment by choosing a Python interpreter 
# and making a ./venv directory to hold it:
virtualenv -p python3 ./venv

# activate the virtual environment using a shell-specific command:
source ./venv/bin/activate

# install dependecies
pip install -r requirements.txt

# setting python path
export PYTHONPATH=$PATHONPATH:<path-to-project-dir>/xCoFormer/

# (if you need) to exit virtualenv later:
deactivate
```

### 2. Datasets
After downloading the datasets from [Kaggle Datasets](https://www.kaggle.com/aldebbaran/code-search-datasets ), it should be placed inside the `resources/datasets/` folder as shown below:

```
xCoFormer/
|-- resources
|   |-- datasets
|   |   |-- java_v01
|   |   |   |-- test.jsonl
|   |   |   |-- train.jsonl
|   |   |   `-- val.jsonl
|   |   `-- python_v01
|   |       |-- test.jsonl
|   |       |-- train.jsonl
|   |       `-- val.jsonl
```

### 3. Test Run
The following bash command fits the RNN model over Java dataset using batch_size=128 and a single epoch.
```
python xCoFormer.py tasks=[fit] model=rnn data=java_v01 data.batch_size=128 trainer.max_epochs=1
```
If all goes well the following output should be produced:
```
GPU available: True, used: True
[2020-12-31 13:44:42,967][lightning][INFO] - GPU available: True, used: True
TPU available: None, using: 0 TPU cores
[2020-12-31 13:44:42,967][lightning][INFO] - TPU available: None, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[2020-12-31 13:44:42,967][lightning][INFO] - LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name       | Type                         | Params
------------------------------------------------------------
0 | x1_encoder | RNNEncoder                   | 45.5 M
1 | x2_encoder | RNNEncoder                   | 45.5 M
2 | loss_fn    | MultipleNegativesRankingLoss | 0     
3 | mrr        | MRRMetric                    | 0     
------------------------------------------------------------
91.0 M    Trainable params


Epoch 0: 100%|███████████████████████████████████████████████████████| 5199/5199 [13:06<00:00,  6.61it/s, loss=5.57, v_num=1, val_mrr=0.041, val_loss=5.54]
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 288/288 [00:17<00:00, 16.83it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'m_test_mrr': tensor(0.0410),
 'm_val_mrr': tensor(0.0410),
 'test_mrr': tensor(0.0410),
 'val_loss': tensor(5.5390, device='cuda:0'),
 'val_mrr': tensor(0.0410)}
--------------------------------------------------------------------------------
```
