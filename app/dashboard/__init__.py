import dash
from flask_login import login_required
from flask.helpers import get_root_path
from flask_caching import Cache
from app.dashboard.layout import serve_layout
from app.dashboard.callbacks import register_callbacks


def register_dashapps(app):
    # Meta tags for viewport responsiveness
    meta_viewport = {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}
    dashapp = dash.Dash(__name__,
                        server=app,
                        url_base_pathname='/dashboard/',
                        assets_folder=get_root_path(__name__) + '/assets/',
                        meta_tags=[meta_viewport])
    dashapp.config.suppress_callback_exceptions = True
    cache = Cache(dashapp.server, config={
        # try 'filesystem' if you don't want to setup redis
        'CACHE_TYPE': 'filesystem',
        'CACHE_DIR': 'flask-cache-dir'
    })

    with app.app_context():
        dashapp.title = 'GuidedML'
        dashapp.layout = serve_layout
        register_callbacks(dashapp)

    _protect_dashviews(dashapp)


def _protect_dashviews(dashapp):
    for view_func in dashapp.server.view_functions:
        if view_func.startswith(dashapp.config.url_base_pathname):
            dashapp.server.view_functions[view_func] = login_required(dashapp.server.view_functions[view_func])

