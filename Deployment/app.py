import pickle, bz2, joblib
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained KMeans model from the file
KM = joblib.load('kmeans_model.pkl')

def load_label_encoder(filename):
    with bz2.BZ2File(filename, 'rb') as file:
        object_encoder = pickle.load(file)
    return object_encoder

# Path where the .pbz2 file is located
file_object = 'object_encoder.pbz2'
file_category = 'category_encoder.pbz2'

# Load the object encoder
object_encoder = load_label_encoder(file_object)
category_encoder = load_label_encoder(file_category)

#load scaled
scaler = joblib.load('scaler.pkl')

#Load PCA
pca = joblib.load('pca_model.pkl')

def class_purchase(total_purchases):
    b = 5
    for i in range(12):
        e = b+210
        if b<=total_purchases and total_purchases<=e:
            return f'class {i} : ({b}, {e})'
        b = e

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input data from the form
        Education = float(object_encoder.transform([request.form['Education']])[0])
        Income = float(request.form["Income"])
        Recency = float(request.form["Recency"]) 
        Wines = float(request.form["MntWines"])
        Fruits = float(request.form["MntFruits"])
        Meat = float(request.form["MntMeatProducts"])
        Fish = float(request.form["MntFishProducts"])
        Sweet = float(request.form["MntSweetProducts"])
        Kidhome = float(request.form["Kidhome"])
        Teenhome = float(request.form["Teenhome"])
        Num_Children = Kidhome + Teenhome
        Gold = float(request.form["MntGoldProds"])
        NumDealsPurchases = float(request.form["NumDealsPurchases"])
        NumWebPurchases = float(request.form["NumWebPurchases"])
        NumCatalogPurchases = float(request.form["NumCatalogPurchases"])
        NumStorePurchases = float(request.form["NumStorePurchases"])
        NumWebVisitsMonth = float(request.form["NumWebVisitsMonth"])
        Customer_From_days = float(request.form["Customer_From_days"])
        Age = float(request.form["Age"])
        Family_Size = float(request.form["Family_Size"])
        total_purchases = Gold + Sweet + Fish + Meat + Fruits + Wines
        purchase_quantity = category_encoder.transform([class_purchase(total_purchases)])[0]
        Total_Promos = float(request.form["Total_Promos"])
        
        # Use the KMeans model to predict the cluster

        features = [
                Education,Income,Recency,Wines,Fruits,Meat,Fish,Sweet,Kidhome,Teenhome,
                Num_Children,Gold,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,
                NumWebVisitsMonth,Customer_From_days,Age,Family_Size,total_purchases,purchase_quantity,Total_Promos
            ]
        features = scaler.transform([features])
        features = pca.transform(features)
        prediction = KM.predict(features)
        
    
        # Render the result on the page
        return render_template("index.html", prediction=prediction[0], feature_values=features)
        
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
