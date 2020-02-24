import dash_core_components as dcc
import dash_html_components as html
font = ["Nunito Sans", "-apple-system", "BlinkMacSystemFont", '"Segoe UI"', "Roboto", '"Helvetica Neue"',
        "Arial", "sans-serif", '"Apple Color Emoji"', '"Segoe UI Emoji"', '"Segoe UI Symbol"']

# Components
home_page = dcc.Link('Home', refresh=True, href='/index/', style={'float': 'right', 'color': '#999',
                                                                  })
top_bar = html.Div(html.Div(html.A(html.H1("GuidedNLP",  className="brand")),
                            className="container-fluid"), className='header')


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
        top_bar,
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
    ],  style={"fontFamily": font})
