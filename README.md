# Customer Personality Clustering

## Overview

This project focuses on customer personality clustering using machine learning techniques, andt it was the graduation project of Samsung's Artificial Intelligence course. The goal is to analyze customer data and segment them into different groups based on their purchasing behavior, demographics, and other relevant features. The analysis helps businesses tailor their marketing strategies effectively.



## Dataset

The dataset used in this project is sourced from Kaggle: [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis). It consists of customer demographic data, purchase history, and engagement metrics.

### Features in the Dataset

- **Demographics:** Year of birth, education, marital status, income, family size.
- **Purchase Behavior:** Number of purchases in different categories (wine, fruits, meat, fish, sweets, gold products).
- **Engagement:** Web visits, acceptance of promotional campaigns, complaints.
- **Derived Features:** Age, total purchases, and promotional engagement.

## Project Structure

- `customers-personality-clustering.ipynb`: Jupyter Notebook containing data preprocessing, exploratory data analysis (EDA), clustering, and evaluation.
- `Customers Personality Clustering.ppx`: A detailed report covering the objectives, business questions, methodology, and findings of the project.

## Methodology

1. **Exploratory Data Analysis (EDA)**

   - Understanding the distribution of customer attributes.
   - These are the questions i will try to answer by analyzing the data set in order to gain a greater understanding of the data and the relationship between them.
     ![image](https://github.com/user-attachments/assets/ced22e67-17da-4a1b-ae74-92fbc1ba81f0)
   - Analyzing relationships between different features.
   - These some images from notebook it display the relationship between some features:
     ![image](https://github.com/user-attachments/assets/3c0be0fc-e087-4a24-9816-c2bb6e285091)
     ![image](https://github.com/user-attachments/assets/ee319032-6b0d-4d60-92da-22a8486ba004)
     ![image](https://github.com/user-attachments/assets/ba184559-4136-4bde-9fab-4fe06093f72a)
     ![image](https://github.com/user-attachments/assets/61797a9b-b767-4f49-99e7-d019515ff151)
     ![image](https://github.com/user-attachments/assets/520c9cfc-c674-4e69-9806-881db93d427d)











2. **Data Preprocessing**

   - Handling missing values and outliers.
   - Encoding categorical variables and scaling numerical features:
```python
LE=LabelEncoder()
for i in object_cols:
    object_le=LE
    data[i]=object_le.fit_transform(data[i])
for i in category_col:
    category_le=LE
    data[i]=category_le.fit_transform(data[i]) 
```
```python
scaled=StandardScaler()
scaled.fit(ds)
scaled_ds = pd.DataFrame(scaled.transform(ds),columns= ds.columns )
```

   - Added some new features based on original features and combine some other:
```python
# We will classify the number of purchases into more than one category
s=5
name_class=[]
for i in range(12):
    t='class ' + str(i) +" : ("+str(s)+ ", " +str(s+210) +')'
    name_class.append(t)
    s=s+210
inter=[5,215,425,635,845,1055,1265,1475,1685,1895,2105,2315,2525]
data['purchase_quantity']=pd.cut(data['total_purchases'],bins=inter,labels=name_class)
```
   - Applying Principal Component Analysis (PCA) for dimensionality reduction:
```python
# Using PCA to reduce the dimensions of the data to 3 dimensions
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"]))
PCA_ds.describe().T
```     

3. **Clustering Models**

   - Using the Elbow method to determine the optimal number of clusters.
   ```python
   Elbow_M = KElbowVisualizer(KMeans(), k=10)
   Elbow_M.fit(PCA_ds)
   Elbow_M.show()
   ```
   ![image](https://github.com/user-attachments/assets/901d7fec-e36b-4b86-93b6-e2b4b1773e64)
   - Applying **Agglomerative Clustering** and **K-Means Clustering**.
     **Agglomerative Clustering**:
     ```python
     #Initiating the Agglomerative Clustering model 
     AC = AgglomerativeClustering(n_clusters=4)
     # fit model and predict clusters
     yhat_AC = AC.fit_predict(PCA_AC)
     PCA_AC["Clusters"] = yhat_AC
     #Adding the Clusters feature to the orignal dataframe.
     data_AC["Clusters"]= yhat_AC
     ```
     **K-Means Clustering**
     ```python
     #Initiating the K-Means model
     KM = KMeans(n_clusters=4)
     # fit model and predict clusters
     yhat_KM = KM.fit_predict(PCA_KM)
     PCA_KM["Clusters"] = yhat_KM
     #Adding the Clusters feature to the orignal dataframe.
     data_KM["Clusters"]= yhat_KM
     ```
     



4. **Evaluation**

   - Visualizing cluster distributions and feature relationships.
   - Assessing the effectiveness of clustering results.
     ![image](https://github.com/user-attachments/assets/d2eda0dc-af87-4341-958d-0e2cc28b4ff0)
     ![image](https://github.com/user-attachments/assets/c3b279f9-0b10-4412-95cb-6d1e70b90f49)


## Key Findings

- Customers were segmented into 4 distinct clusters based on income, purchase behavior, and family structure.
- Income showed a near-linear relationship with the number of purchases.
- Promotional campaigns had a low success rate, with 80% of customers not accepting offers.
- Customers who visited the website frequently were more likely to make purchases online.

## Usage
1. **The first method**
   1. Open the Jupyter Notebook:
      ```bash
      jupyter notebook customers-personality-clustering.ipynb
      ```
   2. Run the notebook cells sequentially to process the data and generate clusters.
   3. Then can you train the model with any data you want.
   
2. **The second method**
   1. You can use the Deployment file to try out the model.
   2. Just download the files and run the app file. 


## Conclusion

This project provides valuable insights into customer segmentation using machine learning techniques. Businesses can leverage these insights to enhance marketing strategies and improve customer engagement.

## Acknowledgments

- Samsung Innovation Campus for supporting this analysis.
- Kaggle for providing the dataset.

## License

This project is intended for educational purposes. Please check the dataset license before using it for commercial applications.

