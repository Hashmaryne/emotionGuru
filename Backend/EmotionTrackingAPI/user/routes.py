from flask import Flask
from main import app
from user.modules import User

@app.route('/user/signup', methods=['GET'])
def signup():
    return User().signup()