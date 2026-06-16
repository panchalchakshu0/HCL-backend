import argparse
from flask import Flask
from flask_cors import CORS
try:
    from .routes import bp
    from .database import init_db
except ImportError:
    from routes import bp
    from database import init_db

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/')
    def health_check():
        return {"status": "healthy"}, 200

    app.register_blueprint(bp, url_prefix='/api')
    init_db()
    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)
