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
- scripts: This folder contains some scripts required for result analysis, cron jobs and so on.

## Steps to replicate this prototype on localhost

### Step 1
Clone this repo.
Install requirements using `pip install -r requirements.txt`

### Step 2: Database setup
#### Mysql local database 
You need to create a new database called shapely in your local mysql server:
Some basic steps:
- Open mysql shell
- Switch to mysql using \sql
- Connect to server using \connect root@localhost
- type create database shapely;

(or)

You could use mysql workbench to create the new database

#### Configure the app
Please set the PASSWORD variable in app/run_config.py to your local DB password.
Set SETUP = "local".

#### Database migrations for flask
Change the database URL and password in config.py
based on your local database URL.

`flask db init`

`flask db migrate -m 'init'`

`flask db upgrade`




### Step 3: Generate annotation data
- Now, use run either `generate_annotation_data.py` or for more detailed descriptions, `notebooks/guided_training-tutorial.ipynb`. 
- Running either of these should create new tables in shapely database. These tables will store the batches to be labelled.
These tables are used by the flask and dash applications.
- If you are running for the first time, make sure the tables are added in your local database.


### Step 4: Run application
- run server.py
- You should be able to see the login/register screen now.
Select the guided or unguided version of the dataset for which you performed step 3.


