# Explainable active learning
We aim to add the concept of explainability to active learning. This is composed of two parts:
- Explaining active learning batches
    - Generating batches that are more transparent and easier to analyze
    - Incorporating shapely explanations for explanating model behavior
    - Uncertainty of the explainable batches - How to quanity uncertainty, understand explainability with uncertainty 

- Explaning Active learning selection strategies: This part is addressed in a separate repo: https://github.com/sahithyaravi1493/alre
    - Visualizing active learning selection strategies – Insight into AL algorithm
    - Uncertainty maps, cluster visualization and uncertainty changes over different batches
    - Model results are shown after each batch
    - Supports 3 BMAL algorithms: 
    - ranked batch mode, k-means uncertain, k-means closest

## File structure
- Project Overview.pptx - Gives an overview of our project and future directions.
- notebooks : This folder contains machine learning experiments/ tutorial examples carried out for different datasets
- app : This folder contains the code for the flask and dash web applications.
- migrations: This folder contains the database migrations.
- models : This folder contains some built-in functions which help with modeling, explanations, clustering, plotting etc.

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

### Mysql local database 
You need to create a new database called shapely in your local mysql server:
- Open mysql shell
- Switch to mysql using \sql
- Connect to server using \connect root@localhost
- create database shapely;


### Generate annotation data
- Now, use run either generate_annotation_data.py or for more detailed descriptions, notebooks/guided_training-tutorial.ipynb. 
It should create new tables in shapely database. These tables will store the batches to be labelled.
These tables are used by the flask and dash applications.

### Run application
- run server.py
You should be able to see the login screen now
