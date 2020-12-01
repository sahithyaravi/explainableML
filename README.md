# Explainable active learning
We aim to add the concept of explainability to active learning. This is composed of two parts:
- Explaining active learning batches
    - Generating batches that are more transparent and easier to analyze and annotate
    - Incorporating shapely explanations for explanations model behavior

- Explaning Active learning selection strategies: This part is addressed in a separate repo: https://github.com/sahithyaravi1493/alre
    - Visualizing active learning selection strategies – Insight into AL algorithm
    - Uncertainty maps, cluster visualization and uncertainty changes over different batches
    - Model results are shown after each batch
    - Supports 3 BMAL algorithms: 
    - ranked batch mode, k-means uncertain, k-means closest



## Steps to replicate this prototype on localhost

### Step 1
Clone this repo.
Install requirements using `pip install -r requirements.txt`

### Step 2
#### Local database setup for login
Change the database URL and password in config.py
based on your local database URL.

`flask db init`

`flask db migrate -m 'init'`

`flask db upgrade`

### Mysql local database for cluster.

- Open mysql shell
- Switch to mysql using \sql
- Connect to server using \connect root@localhost
- create database shapely;
- Now, run training.py. It should create new tables for storing the batches to be labelled.

### run server.py
