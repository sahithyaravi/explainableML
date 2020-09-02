import dash_core_components as dcc
import dash_html_components as html
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
                              options=[{'label': 'davidson', 'value': 'davidson_dataset'},
                                        {'label': 'davidson_noshap', 'value': 'davidson_noshap'},
                                       # {'label': 'founta', 'value': 'founta_dataset'},
                                       # {'label': 'gao', 'value': 'gao_dataset'},
                                       # {'label': 'waseem', 'value': 'waseem_dataset'},
                                       # {'label': 'mnist-mini', 'value': 'mnist'},
                                       ],
                              clearable=False,
                              searchable=False,
                              value=''
                              )

submit_dataset = html.Button('Submit', id='start', autoFocus=True)
queries = html.Div(id='queries')

url = dcc.Location(id='url')

###################
next_button = html.Button('Fetch next batch', id='next_round', autoFocus=True,
                          style={'color': 'white', 'background-color': 'green'})
radio_label = dcc.RadioItems(
    id='radio_label',
    options=[
        {'label': 'Hate', 'value': 1},
        {'label': 'Non-hate', 'value': 0},
        {'label': 'Bad cluster', 'value': -1}
    ],
    style={'marginTop': '10px', 'marginBottom': '50px'}
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


        dcc.Store(id='store_clicks'),
        queries,
        radio_label,
        next_button,
    ],  style={"fontFamily": font, 'verticalAlign': 'middle'})
