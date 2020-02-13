import dash_core_components as dcc
import dash_html_components as html
font = ["Nunito Sans", "-apple-system", "BlinkMacSystemFont", '"Segoe UI"', "Roboto", '"Helvetica Neue"',
        "Arial", "sans-serif", '"Apple Color Emoji"', '"Segoe UI Emoji"', '"Segoe UI Symbol"']
# Components
top_bar = html.Div(
            id='top-bar',
            className='row',
            style={'backgroundColor': '#fa4f56',
                   'height': '10px',
                   }
        )
title = html.H4(html.A('Shapely accelerated smart labelling',
                       style={'text-decoration': 'none', 'color': 'inherit'},
                       href='https://github.com/plotly/dash-svm'))

choose_dataset = dcc.Dropdown(id='select_dataset',
                              options=[{'label': 'davidson', 'value': 'davidson_dataset'},
                                       {'label': 'founta', 'value': 'founta_dataset'},
                                       {'label': 'gao', 'value': 'gao_dataset'},
                                       {'label': 'waseem', 'value': 'waseem_dataset'},
                                       #{'label': 'mnist-mini', 'value': 'mnist'},
                                       ],
                              clearable=False,
                              searchable=False,
                              value='davidson_dataset'
                              )

submit_dataset = html.Button('Start fresh', id='start')
queries = html.Div(id='queries')

url = dcc.Location(id='url')
home_page = dcc.Link('Home', refresh=True, href='/index/')
###################
next_button = html.Button('Fetch next batch', id='next_round', autoFocus=True,
                          style={'color': 'white', 'background-color': 'green'})
radio_label = dcc.RadioItems(
    id='radio_label',
    options=[
        {'label': '1', 'value': 1},
        {'label': '0', 'value': 0},
        {'label': 'Bad cluster', 'value': -1}
    ],
    value=0
)
layout = html.Div(
    children=[
        # TOP BAR AND BANNER
        url,
        home_page,
        top_bar,
        title,
        html.Div(
            className='control-section',
            children=[
                html.Div(className='control-element',
                         children=[html.Div(children=["Select Dataset:"],
                                            style={'width': '40%'}),
                                   html.Div(choose_dataset, style={'width': '60%'})
                                   ]),
             ]),
        submit_dataset,

        dcc.Store(id='store_clicks'),

        queries,
        radio_label,
        next_button,
    ], style={"fontFamily": font})
