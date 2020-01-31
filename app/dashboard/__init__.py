import dash
from app.dashboard.layout import layout
from app.dashboard.callbacks import register_callbacks
from flask_login import login_required
from flask.helpers import get_root_path
from flask_caching import Cache

def register_dashapps(app):
    # Meta tags for viewport responsiveness
    meta_viewport = {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}
    dashapp1 = dash.Dash(__name__,
                         server=app,
                         url_base_pathname='/dashboard/',
                         assets_folder=get_root_path(__name__) + '/assets/',
                         meta_tags=[meta_viewport])
    dashapp1.config.suppress_callback_exceptions = True
    cache = Cache(dashapp1.server, config={
        # try 'filesystem' if you don't want to setup redis
        'CACHE_TYPE': 'filesystem',
        'CACHE_DIR': 'flask-cache-dir'
    })

    with app.app_context():
        dashapp1.title = 'Dashapp 1'
        dashapp1.layout = layout
        register_callbacks(dashapp1)

    _protect_dashviews(dashapp1)


def _protect_dashviews(dashapp):
    for view_func in dashapp.server.view_functions:
        if view_func.startswith(dashapp.config.url_base_pathname):
            dashapp.server.view_functions[view_func] = login_required(dashapp.server.view_functions[view_func])

