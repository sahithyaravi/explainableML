import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table


font = ["Nunito Sans", "-apple-system", "BlinkMacSystemFont", '"Segoe UI"', "Roboto", '"Helvetica Neue"',
        "Arial", "sans-serif", '"Apple Color Emoji"', '"Segoe UI Emoji"', '"Segoe UI Symbol"']

# Components
home_page = html.A(dcc.Link('Home', refresh=True, href='/index/', className='li'),  className='nav-items')
brand_name = html.H1("GuidedML",  className="brand")
top_bar = html.Div(html.Div([html.A(brand_name),
                             home_page],
                            className="container-fluid"),
                   className='header')


choose_dataset = dcc.Dropdown(id='select_dataset',
                              options=[{'label': 'davidson_guided', 'value': 'davidson_dataset_cluster'},
                                       {'label': 'davidson_not_guided', 'value': 'davidson_dataset_noshap'},
                                       # {'label': 'yelp_guided', 'value': 'yelp_dataset_cluster'},
                                       # {'label': 'yelp_not_guided', 'value': 'yelp_dataset_noshap'},
                                       # {'label': 'bank_guided', 'value': 'bank_dataset_cluster'},

                                       # {'label': 'founta', 'value': 'founta_dataset'},
                                       # {'label': 'gao_guided', 'value': 'gao_dataset_cluster'},
                                       # {'label': 'waseem', 'value': 'waseem_dataset'},
                                       # {'label': 'mnist-mini', 'value': 'mnist'},
                                       ],
                              clearable=False,
                              searchable=False,
                              value=''
                              )

submit_dataset = html.Button('Submit', id='start', autoFocus=True)

queries = dcc.Loading(html.Div(id='queries'))

url = dcc.Location(id='url')

progress_bar = html.Progress( id='progress', value=0)

###################

radio_label = dcc.RadioItems(
    id='radio_label',
    options=[
        {'label': 'Hate', 'value': 1},
        {'label': 'Non-hate', 'value': 0},
        {'label': 'Bad cluster', 'value': -1}
    ],
    style={'display': 'none'}
)

stop_watch = html.Div([
    dcc.Interval(id='interval1', interval=5 * 1000, n_intervals=0),
    html.H1(id='label1', children='')
])

dummy_table = html.Div(id='dummy')


def serve_layout():
    layout = html.Div(
        children=[
            # TOP BAR AND BANNER
            url,
            top_bar,

            html.Div(
                className='control-section',
                children=[
                    html.Div(className='control-element',
                             children=[html.Div(children=["Choose a  Dataset and click submit"],
                                                style={'width': '30%'}),
                                       html.Div(choose_dataset, style={'width': '30%'})
                                       ]),
                    html.Div(className='control-element',
                             children=[html.Div(children=["  "],
                                                style={'width': '30%'}),
                                       html.Div(submit_dataset, style={'width': '30%'})
                                       ]),

                 ]),

            html.Div(id='info', style={'marginLeft': "50px", 'width': '60%'}),
            dummy_table,
            dcc.Store(id='store_clicks'),
            queries,
            html.Div(id='timer'),
            # radio_label,
            html.Div(id='next'),
            progress_bar,
            stop_watch,
        ],  style={"fontFamily": font, 'verticalAlign': 'middle'})
    return layout


