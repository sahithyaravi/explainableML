from dash.dependencies import Output, Input, State
from flask_login import current_user
import dash_html_components as html
import pandas as pd
import os
import dash_table
print(str(current_user))
from logging import Logger

logger = Logger(name='shapely', level=1)


def register_callbacks(app):
    @app.callback(Output('store_clicks', 'data'),
                  [Input('start', 'n_clicks')],
                  [State('select_dataset', 'value'),
                   State('store_clicks', 'data')],
                  )
    def callback(clicks, dataset, stored_clicks):
        stored_clicks = 0 if stored_clicks is None else stored_clicks
        if clicks is not None and clicks > stored_clicks:
            logger.debug("Fetch dataset for labelling: {}".format(dataset))
            # Plot ground truth data.
            # Plot clusters
        return clicks








    # def callback(clicks, dataset):
    #     output = html.Div()
    #     print(clicks, current_user)
    #     if clicks is not None:
    #         df = fetch_queries(dataset)
    #         display_table = create_table(df)
    #         if clicks == 1:
    #             output = html.Div(display_table)
    #     return output


def fetch_queries(dataset):
    # fetch dataset queries from pickle for now
    print(os.curdir)
    df = pd.read_pickle('queries.pkl')
    df = df[df["labelled"] == False]
    min_cluster = df["cluster_id"].min()
    current_cluster = df[df["cluster_id"] == min_cluster]
    print(current_cluster.cluster_id.unique())
    return pd.DataFrame(current_cluster)


def create_table(df):
    table = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_as_list_view=True,
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'left',
            'height': 'auto',
            'minWidth': '0px', 'maxWidth': '180px',
            'whiteSpace': 'normal'
        }
    )
    return table
