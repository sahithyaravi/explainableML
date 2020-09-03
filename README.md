# Guided learning
A prototype to test shapely based guided learning.
Created using plotly and dash.
Flask is used for backend + login.
Dash lets you create a simple front end without using js.
Plus it supports awesome visualizations.

### Step 1
Instlal requirements using `pip install -r requirements.txt`

### Step 2
#### Local database setup for login
Change the database URL and password in config.py and models/guided_learning.py
based on your local database URL.

`flask db init`

`flask db migrate -m 'init'`

`flask db upgrade`

### Mysql local database for cluster.

- Open mysql shell
- Switch to mysql using \sql
- Connect to server using \connect root@localhost
- create database shapely;
- Now, run training.py. It should create new tables for storing the clusters to be labelled.

### run server.py