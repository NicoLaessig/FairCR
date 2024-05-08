from flask import Flask
from flask_cors import CORS
from routes import Router
from config import Config




def start():

    app = Flask(__name__)
    CORS(app)
    app.config.from_object(Config)
    router = Router(app)
    print("Started Flask.app")
    app.run(port=5000,use_reloader=True)
    return app

start()