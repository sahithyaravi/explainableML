import dash_core_components as dcc
import dash_html_components as html


top_bar = html.Div(
            id='top-bar',
            className='row',
            style={'backgroundColor': '#fa4f56',
                   'height': '10px',
                   }
        )
title = html.H2(html.A('Active Learning Explorer',
                           style={'text-decoration': 'none', 'color': 'inherit'},
                           href='https://github.com/plotly/dash-svm'))

choose_dataset = dcc.Dropdown(id='select_dataset',
                              options=[{'label': 'davidson', 'value': 'davidson_dataset'},
                                       {'label': 'founta', 'value': 'founta_dataset'},
                                       {'label': 'gao', 'value': 'gao_dataset'},
                                       {'label': 'golbeck', 'value': 'golbeck_dataset'},
                                       {'label': 'waseem', 'value': 'waseem_dataset'},
                                       {'label': 'mnist-mini', 'value': 'mnist'},
                                       ],
                              clearable=False,
                              searchable=False,
                              value='davidson_dataset'
                              )

submit_dataset = html.Button('Start', id='start')
submit_notification = html.Div(id='notification')
radio_label = dcc.RadioItems(id='radio_label')
url = dcc.Location(id='url')
home_page = dcc.Link('Home', refresh=True, href='/index/')
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
        submit_notification,
        dcc.Store(id='store_clicks')

    ])
