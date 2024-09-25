import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering

class Customer_Personality_Analysis:
    def __init__(self):
        self.model_get_data()
        self.clean()
        self.model_preprocess()
        self.model_cluster()


    def model_get_data(self):
        self.data =pd.read_csv("C:\\Users\\User\\Desktop\\samsung\\project-samsung\\Final project\\marketing_campaign.csv",sep='\t',encoding='utf-8')

        
    def clean(self):
        self.data["Dt_Customer"] = pd.to_datetime(self.data["Dt_Customer"])
        dates = []
        for i in self.data["Dt_Customer"]:
            i = i.date()
            dates.append(i)
        d1 = max(dates)
        for i in dates:
            t=d1-i
        days = []
        for i in dates:
            delta = d1 - i
            days.append(delta)
        self.data["Customer_From_days"] = days
        self.data["Customer_From_days"] = pd.to_numeric(self.data["Customer_From_days"], errors="coerce")
        for i in range(len(self.data['Customer_From_days'])):
            t=0
            t=self.data['Customer_From_days'][i]
            self.data['Customer_From_days'][i]=t/60/60/24/1000000000
        self.data['Marital_Status'].replace('Alone','Single',inplace=True)
        self.data['Marital_Status'].replace('Absurd','Single',inplace=True)
        self.data['Marital_Status'].replace('YOLO','Single',inplace=True)

        self.data["Living_With"]=self.data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Widow":"Alone", "Divorced":"Alone" ,'Single':"Alone"})

        self.data["Num_Children"]=self.data["Kidhome"]+self.data["Teenhome"]

        self.data["Family_Size"] = self.data["Living_With"].replace({"Alone": 1, "Partner":2})+ self.data["Num_Children"]
        self.data['total_purchases']=self.data['MntFishProducts']+self.data["MntFruits"]+self.data['MntGoldProds']+self.data['MntMeatProducts']+self.data['MntSweetProducts']+self.data["MntWines"]
        inter=pd.interval_range(start=5,freq=210, end= 2525)
        inter
        s=5
        name_class=[]
        for i in range(12):
            t='class ' + str(i) +" : ("+str(s)+ ", " +str(s+210) +')'
            name_class.append(t)
            s=s+210
        inter=[5,215,425,635,845,1055,1265,1475,1685,1895,2105,2315,2525]
        self.data['purchase_quantity']=pd.cut(self.data['total_purchases'],bins=inter,labels=name_class)
        self.data["Total_Promos"] = self.data["AcceptedCmp1"]+ self.data["AcceptedCmp2"]+ self.data["AcceptedCmp3"]+ self.data["AcceptedCmp4"]+ self.data["AcceptedCmp5"]
        self.data["Age"] = 2021-self.data["Year_Birth"]
        fill_tobed=self.data['Income'].dropna()
        self.data['Income']=self.data['Income'].fillna(pd.Series(np.random.choice(fill_tobed,size=len(self.data.index))))
        fill_tobed=self.data['purchase_quantity'].dropna()
        self.data['purchase_quantity']=self.data['purchase_quantity'].fillna(pd.Series(np.random.choice(fill_tobed,size=len(self.data.index))))
        self.data=self.data[self.data['Income']<150000]
        self.data=self.data[self.data['Age']<100]

    
    def model_preprocess(self):
        
        self.data=self.data[['Education', 'Marital_Status', 'Income','Recency', 'MntWines', 'MntFruits',
        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
        'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
        'Customer_From_days', 'Age', 'Family_Size', 'total_purchases','purchase_quantity','Total_Promos']]
        s = (self.data.dtypes == 'object')
        n = (self.data.dtypes == 'category')
        object_cols = list(s[s].index)
        category_col = list(n[n].index)
        LE=LabelEncoder()
        for i in object_cols:
            self.data[i]=self.data[[i]].apply(LE.fit_transform)
        for i in category_col:
            self.data[i]=self.data[[i]].apply(LE.fit_transform)   
        ds = self.data.copy()
        scaler = StandardScaler()
        scaler.fit(ds)
        scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
        pca = PCA(n_components=3)
        pca.fit(scaled_ds)
        self.PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"]))
        

    def model_cluster(self):
        Elbow_M = KElbowVisualizer(KMeans(), k=10)
        #Elbow_M.fit(self.PCA_ds)
        AC = AgglomerativeClustering(n_clusters=4)
        yhat_AC = AC.fit_predict(self.PCA_ds)
        self.PCA_ds["Clusters"] = yhat_AC
        self.data["Clusters"]= yhat_AC
        

    #def model_predict(self,text):
    #    text = self.cv.transform([text])
    #    return self.clf.predict(text)[0]