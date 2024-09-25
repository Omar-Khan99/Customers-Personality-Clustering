from flask import Flask,jsonify,request,render_template
from model import Customer_Personality_Analysis
app = Flask(__name__)
CP = Customer_Personality_Analysis() ## get data,preprocess and train
###########################
@app.route("/")
def home():
    
    return render_template('home.html')

@app.route("/cluster")
def cluster():
    CP.model_cluster()
    return render_template('result.html',message='model trained successfully')
@app.route("/conclusion")
def conclusion():
    
    return render_template('conclusion.html')

###########################
app.run()