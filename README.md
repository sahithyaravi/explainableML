# Guided learning
A prototype to test shapely based guided learning


### Local database setup for login


`flask db init`

`flask db migrate -m 'init'`

`flask db upgrade`

## Mysql local database for labels

- Open mysql shell
- Switch to mysql using \sql
- Connect to server using \connect root@localhost
- create database shapely;
- Now u have database created, Use it for creating tables and storing labels.
