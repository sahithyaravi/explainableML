import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import dash_table

import logging
import os
import time
from flask_login import current_user

from app.run_config import *
logging.basicConfig(level=logging.INFO)


def register_callbacks(app):
    @app.callback([Output('store_clicks', 'data'),
                   Output('info', 'children'),
                   Output('next', 'children'),
                   Output('dummy', 'children'),
                   Output('timer', 'children')
                   ],
                  [Input('start', 'n_clicks')],
                  [State('select_dataset', 'value'),
                   State('store_clicks', 'data'),
                   ],
                  )
    def get_dataset(clicks, dataset, stored_clicks):
        dummy_table_layout = dash_table.DataTable(

            selected_rows=[],
            id='datatable',
            page_action='none',

        )
        start_time = dcc.Store(id='start_time')
        stored_clicks = 0 if stored_clicks is None else stored_clicks
        # style = {'marginTop': '10px', 'marginBottom': '50px', 'display': 'none'}
        text = ""
        logging.info(f"{dataset}")
        next = ""
        if clicks is not None and clicks > stored_clicks and dataset is not None:
            user_file = pd.DataFrame()
            if not os.path.exists(f"{current_user.id}_{dataset}_user.pkl"):
                user_file.to_pickle(f"{current_user.id}_{dataset}_user.pkl")
            if dataset == "davidson_dataset_cluster":
                text = dcc.Markdown('''
                * For this experiment, you will be presented groups of sentences.
                * Select the sentences in the group which contain **hate-speech***.
                * **Similar** sentences are present within the group. So it is possible to label all sentences by
                looking at 1-2 samples in the group.
                * Important words may be highlighted in **bold**
                * Click next to continue.
                ''')
                next = html.Button('NEXT', id='next_round', autoFocus=True,
                                   style={'color': 'white', 'background-color': 'green', 'marginLeft': '100px'})
            elif dataset == "yelp_dataset_cluster" or dataset=="bank_dataset_cluster":
                text = dcc.Markdown('''
                           * For this experiment, you will be presented groups of sentences.
                           * Select the sentences in the group which contain **positive review***.
                           * **Similar** sentences are present within the group. So it is possible to label all sentences by
                           looking at 1-2 samples in the group.
                           * Important words may be highlighted in **bold**
                           * Click next to continue.
                           ''')
                next = html.Button('NEXT', id='next_round', autoFocus=True,
                                   style={'color': 'white', 'background-color': 'green', 'marginLeft': '100px'})
            elif dataset == "yelp_dataset_noshap":
                text = dcc.Markdown('''
                * For this experiment, you will be presented a list of sentences.
                * Select the sentences in the list which contain **positive review**.
                * **Random/ Dissimilar** sentences are listed together.
                Click next to continue.
                           ''')
                next = html.Button('NEXT', id='next_round', autoFocus=True,
                                   style={'color': 'white', 'background-color': 'green', 'marginLeft': '100px'})
            else:
                text = dcc.Markdown('''
                * For this experiment, you will be presented a list of sentences.
                * Select the sentences in the list which contain **hate-speech**.
                * **Random/ Dissimilar** sentences are listed together.
                Click next to continue.
                ''')
                next = html.Button('NEXT', id='next_round', autoFocus=True,
                                   style={'color': 'white', 'background-color': 'blue', 'marginLeft': '100px'})

            logging.info("Fetch dataset for labelling: {}".format(dataset))
            logging.info("Fetch dataset for labelling: {}".format(dataset))
            from ..utils import fetch_all_unlabelled_data
            df = fetch_all_unlabelled_data(dataset)
            df['labelled'] = False
            df['label'] = 0
            logging.info(df.head())
            df.to_pickle(f"{dataset}queries.pkl")
            style = {'marginTop': '10px', 'marginBottom': '50px',  'display': 'block'}
        return clicks, text, html.Div(next), dummy_table_layout, start_time

    @app.callback([Output('queries', 'children'),
                   Output('start_time', 'data')],
                  [Input('next_round', 'n_clicks'),
                   ],
                  [State('datatable', 'selected_rows'),
                   State('select_dataset', 'value'),
                   State('start_time', 'data')])
    def get_queries_write_labels(next_round,  selected_rows, dataset, start_time):
        output = ""
        print(next_round, " next round ", dataset)

        if next_round is not None and next_round:
            if next_round == 1:
                start_time = time.time()
            if next_round > 1:
                logging.debug("Updating labels for next round", next_round-1)

            df = fetch_queries(dataset, next_round, selected_rows)

            table_text = []
            for index, row in df.iterrows():
                row["text"] = " " + row["text"] + " "
                if "keywords" in df.columns:
                    keyword = row["keywords"]
                    pos = row["positive"]
                    neg = row["negative"]
                    out = row["text"]
                    if keyword is not None and keyword != None:
                        str1 = row["text"].split(row["keywords"])
                        out = f"{str1[0]}**{keyword}**{str1[1]}" if len(str1) > 1 else f"{str1[0]}**{keyword}**"
                    # if pos is not None and pos != None:
                    #     str1 = out.split(pos)
                    #     out = f"{str1[0]}**{pos}**{str1[1]}" if len(str1) > 1 else f"{str1[0]}**{pos}**"
                    # if neg is not None and neg!= None:
                    #     str1 = out.split(neg)
                    #     out = f"{str1[0]}**{neg}**{str1[1]}" if len(str1) > 1 else f"{str1[0]}**{neg}**"

                else:
                    out = row["text"]
                table_text.append(out)
            if table_text:
                note = dcc.Markdown(''' * Use **select all checkbox** if all sentences in the group are hate speech''',
                style={'marginLeft': '50px'})
                checkall = dcc.Checklist(options=
                                         [{'label': 'select/unselect all', 'value': 'all'},
                                          ],
                                         value=[],
                                         id='checkall',
                                         style={'marginLeft': '50px'}
                                         )
                display_table = create_table(pd.DataFrame(table_text))
                output = html.Div([note, checkall, display_table])
            else:
                end = time.time()
                time_elapsed = end-start_time
                # Insert the labels in to database using pymysql
                from app.utils import time_to_db, write_to_db_pkl

                time_to_db(current_user.id, time_elapsed, dataset)
                if SETUP == "local":
                    import os
                    labelled_df = pd.read_pickle(f"{current_user.id}_{dataset}.pkl")
                    write_to_db_pkl(labelled_df, dataset=dataset)
                    os.remove(f"{current_user.id}_{dataset}.pkl")
                output = html.P(f" Great! Done with this dataset."
                                f"Select the next dataset for labelling.", style={'marginLeft': '50px'})
        return output, start_time

    @app.callback(
        [Output('datatable', 'selected_rows')],
        [Input('checkall', 'value'),
         ],
        [State('datatable', 'derived_virtual_data'), ]
    )
    def select_deselect(value, selected_rows):
        logging.info("select deselect")

        if value:
            if selected_rows is None:
                return [[]]
            else:
                logging.info("return all rows")
                return [[i for i in range(len(selected_rows))]]
        else:
            logging.info("return empty")
            return [[]]


########################################################################################################################
def fetch_queries(dataset, next_round, selected_rows):

    # fetch dataset queries from pickle for now
    logging.debug(f"directory{os.curdir}")
    df = pd.read_pickle(f"{dataset}queries.pkl")
    round = df["round"].iloc[0]

    from ..utils import get_labelled_indices,get_labelled_indices_pkl

    # Get all labelled indices of the user
    labelled_indices = get_labelled_indices_pkl(dataset, current_user.id, round)
    # extract unlabelled df from whole df
    df_unlabelled = df[~df["index"].isin(labelled_indices)]
    if df_unlabelled.empty:
        return ""
    # find minimum unlabelled cluster, this would be the cluster we got labels for
    curr_cluster = df_unlabelled["cluster_id"].min()
    logging.info(f" Writing labels for cluster {curr_cluster} ")
    next_cluster = curr_cluster  # for first round

    if next_round > 1:

        labelled_df = df[df["cluster_id"] == curr_cluster]
        labelled_df["label"] = 0
        # If any row is selected
        if selected_rows:
            selected_rows = [r + labelled_df['index'].values[0] for r in selected_rows]
            labelled_df[labelled_df['index'].isin(selected_rows)]["label"] = 1

        # Insert the labels in to database using pymysql
        from app.utils import write_to_db, write_to_pkl
        write_to_pkl(labelled_df, dataset=dataset)
        df_unlabelled = df_unlabelled[~df_unlabelled["index"].isin(labelled_df['index'].values)]
        next_cluster = df_unlabelled["cluster_id"].min()
    queries = df[df["cluster_id"] == next_cluster]
    logging.info(f" Reading cluster {next_cluster} for user to label")
    return queries


def create_table(df):
    table = dash_table.DataTable(
        columns=[dict(name=i, id=i, presentation='markdown', type='text') for i in df.columns],
        data=df.to_dict('records'),
        # style_as_list_view=True,
        # filter_action="native",
        row_selectable='multi',
        selected_rows=[],
        id='datatable',
        style_header={'backgroundColor': 'white', 'fontWeight': 'bold', 'display':'none'},
        style_cell={'textAlign': 'left', 'backgroundColor': 'white', 'height':'auto',
                    'whiteSpace': 'normal',
                    "fontFamily": "Nutino Sans",  "fontSize": 16},
        style_table={'minHeight': '400px',
                     'maxWidth': '1200px',
                     'maxHeight': '800px',
                     'overflowY': 'scroll',
                     'overflowX': 'scroll',
                     'marginBottom': '50px',
                     'marginLeft': '100px'},
        page_action='none',

    )
    return table

