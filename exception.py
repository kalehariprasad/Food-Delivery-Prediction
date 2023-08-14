import os ,sys 
from flask import Flask
from src.logger import logging
from src.exception import CustomeExeption
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        raise Exception('we are testing our Exception file')
    except Exception as e:
        ML = CustomeExeption(e,sys)
        logging.info(ML.error_message)
        logging.info("We are testing our logging file")
        return "Welcome to my flask application"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
