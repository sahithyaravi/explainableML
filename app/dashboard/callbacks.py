from dash.dependencies import Output, Input, State
import dash_html_components as html
import pandas as pd
import os
import dash_table
import logging
import os
from flask_login import current_user

logging.basicConfig(level=logging.INFO)


def register_callbacks(app):
    @app.callback(Output('store_clicks', 'data'),
                  [Input('start', 'n_clicks')],
                  [State('select_dataset', 'value'),
                   State('store_clicks', 'data')],
                  )
    def get_dataset(clicks, dataset, stored_clicks):
        stored_clicks = 0 if stored_clicks is None else stored_clicks
        if clicks is not None and clicks > stored_clicks:
            logging.info("Fetch dataset for labelling: {}".format(dataset))
            #df = pd.read_csv(f"../datasets/{dataset}_cluster.csv")
            PASSWORD = '1993sahi11'
            database_url = f"mysql+pymysql://root:{PASSWORD}@localhost/shapely"
            df = pd.read_sql_table(f"{dataset}_cluster", database_url)
            df['labelled'] = False
            df['label'] = 0
            print(df.head())
            df.to_pickle('queries.pkl')

            # Plot ground truth data.
            # Plot clusters
        return clicks

    @app.callback(Output('queries', 'children'),
                  [Input('next_round', 'n_clicks'),
                   Input('radio_label', 'value')],
                  [State('select_dataset', 'value')])
    def get_queries(next_round, label, dataset):
        output = ""
        if next_round is not None:
            if next_round > 1:
                logging.debug("Updating labels for round", next_round-1)
                update_labels(label, dataset)
            df = fetch_queries(dataset=dataset)
            display_table = create_table(pd.DataFrame(df['text']))
            output = html.Div(display_table)
        return output


def update_labels(label, dataset):
    print("update labels")
    df = pd.read_pickle('queries.pkl')
    df_false = df[df["labelled"] == False]
    min_cluster = df_false['cluster_id'].min()
    df["labelled"].loc[df["cluster_id"] == min_cluster] = True
    df["label"].loc[df["cluster_id"] == min_cluster] = label
    labelled_df = df[df["cluster_id"] == min_cluster]
    # Insert the labels in to database using pymysql
    from app.utils import write_to_db
    write_to_db(labelled_df, dataset=dataset)
    df.to_pickle('queries.pkl')
    df.to_csv('queries.csv')


def fetch_queries(dataset):
    # fetch dataset queries from pickle for now
    logging.debug(f"directory{os.curdir}")
    df = pd.read_pickle('queries.pkl')
    PASSWORD = '1993sahi11'
    database_url = f"mysql+pymysql://root:{PASSWORD}@localhost/shapely"
    labels = pd.read_sql_table("label", database_url)
    id = current_user.id
    omit_indices = labels[labels["user_id"]== id and labels["dataset"]]
    df = df[df["labelled"] == False]
    min_cluster = df["cluster_id"].min()
    current_cluster = df[df["cluster_id"] == min_cluster]
    print(current_cluster.cluster_id.unique())
    return pd.DataFrame(current_cluster)


def create_table(df):
    table = dash_table.DataTable(

        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_as_list_view=True,
        filter_action="native",
        id='datatable',
        style_header={'backgroundColor': 'white', 'fontWeight': 'bold', 'display':'none'},
        style_cell={'textAlign': 'left', 'backgroundColor': 'white',
                    "fontFamily": "Nutino Sans", 'textOverflow': 'ellipsis', "fontSize": 14},
        style_table={'minHeight': '400px',
                     'maxWidth': '1000px',
                     'maxHeight': '400px',
                     'overflowY': 'scroll',
                     'overflowX':'scroll'},
        page_action='none',

    )
    return table
