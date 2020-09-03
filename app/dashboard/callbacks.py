import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
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
    @app.callback([Output('store_clicks', 'data'),
                   Output('info', 'children')],
                  [Input('start', 'n_clicks')],
                  [State('select_dataset', 'value'),
                   State('store_clicks', 'data'),
                   ],
                  )
    def get_dataset(clicks, dataset, stored_clicks):
        stored_clicks = 0 if stored_clicks is None else stored_clicks
        style = {'marginTop': '10px', 'marginBottom': '50px', 'display': 'none'}
        text = ""
        logging.info(f"{dataset}")
        if clicks is not None and clicks > stored_clicks and dataset is not None:
            if dataset == "davidson_dataset_cluster":
                text = dcc.Markdown('''
                * For this experiment, you will be presented groups of sentences.
                * Select the sentences in the group which contain **hate-speech***.
                * **Similar** sentences are present within the group.
                * Important words may be highlighted in **bold**
                * Click next to continue.
                ''')
            else:
                text = dcc.Markdown('''
                * For this experiment, you will be presented a list of sentences.
                * Select the sentences in the list which contain **hate-speech**.
                * **Random/ Dissimilar** sentences are listed together.
                Click next to continue.
                ''')

            logging.info("Fetch dataset for labelling: {}".format(dataset))
            logging.info("Fetch dataset for labelling: {}".format(dataset))
            from ..utils import fetch_all_unlabelled_data
            df = fetch_all_unlabelled_data(dataset)
            df['labelled'] = False
            df['label'] = 0
            logging.info(df.head())
            df.to_pickle(f"{dataset}queries.pkl")
            style = {'marginTop': '10px', 'marginBottom': '50px',  'display': 'block'}

        return clicks, text

    @app.callback(Output('queries', 'children'),
                  [Input('next_round', 'n_clicks'),
                   ],
                  [State('datatable', 'selected_rows'),
                   State('select_dataset', 'value')])
    def get_queries_write_labels(next_round,  selected_rows, dataset):
        output = ""
        # print("selected rows", selected_rows)
        if next_round is not None:
            if next_round > 1:
                logging.debug("Updating labels for next round", next_round-1)

            df = fetch_queries(dataset, next_round, selected_rows)
            table_text = []
            for index, row in df.iterrows():
                row["text"] = " " + row["text"] + " "
                if "keywords" in df.columns:
                    keyword = row["keywords"]
                    if keyword is not None and keyword != None:
                        str1 = row["text"].split(row["keywords"])
                        out = f"{str1[0]}**{keyword}**{str1[1]}" if len(str1) > 1 else f"{str1[0]}**{keyword}**"
                    else:
                        out = row["text"]
                else:
                    out = row["text"]
                table_text.append(out)
            if table_text:
                note = html.H4("Use select all checkbox if all sentences in the group are hate speech", style={'marginLeft': '50px'})
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
                output = html.H3(" Great you have finished labelling this dataset. Thank you for your time :)")
        return output

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


def fetch_queries(dataset, next_round, selected_rows):

    # fetch dataset queries from pickle for now
    logging.debug(f"directory{os.curdir}")
    df = pd.read_pickle(f"{dataset}queries.pkl")
    round = df["round"].iloc[0]

    from ..utils import get_labelled_indices

    # Get all labelled indices of the user
    labelled_indices = get_labelled_indices(dataset, current_user.id, round)
    # extract unlabelled df from whole df
    df_unlabelled = df[~df["index"].isin(labelled_indices)]
    # find minimum unlabelled cluster, this would be the cluster we got labels for
    curr_cluster = df_unlabelled["cluster_id"].min()
    logging.info(f" Writing labels for cluster {curr_cluster} to db")
    next_cluster = curr_cluster # for first round

    if next_round > 1:

        labelled_df = df[df["cluster_id"] == curr_cluster]
        labelled_df["label"] = 0
        # If any row is selected
        if selected_rows:
            selected_rows = [r + labelled_df['index'].values[0] for r in selected_rows]
            labelled_df[labelled_df['index'].isin(selected_rows)]["label"] = 1

        # Insert the labels in to database using pymysql
        from app.utils import write_to_db
        write_to_db(labelled_df, dataset=dataset)
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
        style_cell={'textAlign': 'left', 'backgroundColor': 'white',
                    "fontFamily": "Nutino Sans", 'textOverflow': 'ellipsis', "fontSize": 14},
        style_table={'minHeight': '400px',
                     'maxWidth': '1000px',
                     'maxHeight': '600px',
                     'overflowY': 'scroll',
                     'overflowX': 'scroll',
                     'marginBottom': '50px',
                     'marginLeft': '100px'},
        page_action='none',

    )
    return table


    # @app.callback(Output('queries', 'children'),
    #               [Input('next_round', 'n_clicks')
    #                ],
    #               [State('radio_label', 'value'),
    #                State('select_dataset', 'value')])
    # def get_queries_write_labels_radio(next_round, label, dataset):
    #     output = ""
    #     if next_round is not None:
    #         if next_round > 1:
    #             logging.debug("Updating labels for next round", next_round-1)
    #
    #         df = fetch_queries(dataset, next_round, label)
    #         table_text = []
    #         for index, row in df.iterrows():
    #             row["text"] = " " + row["text"] + " "
    #             str1 = row["text"].split(row["keywords"])
    #             keyword = row["keywords"]
    #             out = f"{str1[0]}**{keyword}**{str1[1]}" if len(str1) > 1 else f"{str1[0]}**{keyword}**"
    #             table_text.append(out)
    #         if table_text:
    #             display_table = create_table(pd.DataFrame(table_text))
    #             output = html.Div(display_table)
    #         else:
    #             output = html.Div(" Batch Completed, Model retraining")
    #     return output