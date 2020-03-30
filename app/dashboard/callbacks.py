from dash.dependencies import Output, Input, State
import dash_html_components as html
import pandas as pd
import os
import dash_table
import logging
import os
from flask_login import current_user
rnd = 1
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
            from ..utils import fetch_all_unlabelled_data
            df = fetch_all_unlabelled_data(dataset)
            df['labelled'] = False
            df['label'] = 0
            logging.info(df.head())
            df.to_pickle('queries.pkl')
        return clicks

    @app.callback(Output('queries', 'children'),
                  [Input('next_round', 'n_clicks'),
                   ],
                  [State('radio_label', 'value'),
                   State('select_dataset', 'value')])
    def get_queries_write_labels(next_round, label, dataset):
        output = ""
        if next_round is not None:
            if next_round > 1:
                logging.debug("Updating labels for round", next_round-1)

            df = fetch_queries(dataset, next_round, label)
            table_text = []
            for index, row in df.iterrows():
                row["text"] = " " + row["text"] + " "
                str1 = row["text"].split(row["keywords"])
                keyword = row["keywords"]
                out = f"{str1[0]}**{keyword}**{str1[1]}" if len(str1) > 1 else f"{str1[0]}**{keyword}**"
                table_text.append(out)
            if table_text:
                display_table = create_table(pd.DataFrame(table_text))
                output = html.Div(display_table)
            else:
                output = html.Div(" Batch Completed, Model retraining")
        return output


def fetch_queries(dataset, next_round, labels):
    # fetch dataset queries from pickle for now
    logging.debug(f"directory{os.curdir}")
    df = pd.read_pickle('queries.pkl')
    round = df["round"][0]
    # Select unlabelled min cluster
    from ..utils import get_labelled_indices
    labelled_indices = get_labelled_indices(dataset, current_user.id, round)
    df_unlabelled = df[~df["index"].isin(labelled_indices)]
    min_cluster = df_unlabelled["cluster_id"].min()
    logging.info(f" Writing batch {min_cluster} to db")
    if next_round > 1:
        df["label"].loc[df["cluster_id"] == min_cluster] = labels
        labelled_df = df[df["cluster_id"] == min_cluster]
        # Insert the labels in to database using pymysql
        from app.utils import write_to_db
        write_to_db(labelled_df, dataset=dataset)
        df_unlabelled = df_unlabelled[~df_unlabelled["index"].isin(labelled_df['index'].values)]
        min_cluster = df_unlabelled["cluster_id"].min()
    current_cluster = df[df["cluster_id"] == min_cluster]
    logging.info(f" Reading batch {min_cluster} to query")
    return current_cluster


def create_table(df):
    table = dash_table.DataTable(
        columns=[dict(name=i, id=i, presentation='markdown', type='text') for i in df.columns],
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
                     'overflowX': 'scroll',
                     'marginBottom': '50px'},
        page_action='none',

    )
    return table
