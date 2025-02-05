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
   - Encoding categorical variables and scaling numerical features,
     ```python
LE=LabelEncoder()
for i in object_cols:
    object_le=LE
    data[i]=object_le.fit_transform(data[i])
for i in category_col:
    category_le=LE
    data[i]=category_le.fit_transform(data[i])
```
   - Added some new features based on original features and combine some other
   - Applying Principal Component Analysis (PCA) for dimensionality reduction.

3. **Clustering Models**

   - Using the Elbow method to determine the optimal number of clusters.
   - Applying **Agglomerative Clustering** and **K-Means Clustering**.



4. **Evaluation**

   - Visualizing cluster distributions and feature relationships.
   - Assessing the effectiveness of clustering results.

## Key Findings

- Customers were segmented into 4 distinct clusters based on income, purchase behavior, and family structure.
- Income showed a near-linear relationship with the number of purchases.
- Promotional campaigns had a low success rate, with 80% of customers not accepting offers.
- Customers who visited the website frequently were more likely to make purchases online.

## Dependencies

To run the Jupyter Notebook, install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook customers-personality-clustering.ipynb
   ```
2. Run the notebook cells sequentially to process the data and generate clusters.

## Conclusion

This project provides valuable insights into customer segmentation using machine learning techniques. Businesses can leverage these insights to enhance marketing strategies and improve customer engagement.

## Acknowledgments

- Samsung Innovation Campus for supporting this analysis.
- Kaggle for providing the dataset.

## License

This project is intended for educational purposes. Please check the dataset license before using it for commercial applications.

