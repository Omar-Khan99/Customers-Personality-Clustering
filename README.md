# Customer Personality Clustering

## Overview

This project focuses on customer personality clustering using machine learning techniques. The goal is to analyze customer data and segment them into different groups based on their purchasing behavior, demographics, and other relevant features. The analysis helps businesses tailor their marketing strategies effectively.



## Dataset

The dataset used in this project is sourced from Kaggle: [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis). It consists of customer demographic data, purchase history, and engagement metrics.

### Features in the Dataset

- **Demographics:** Year of birth, education, marital status, income, family size.
- **Purchase Behavior:** Number of purchases in different categories (wine, fruits, meat, fish, sweets, gold products).
- **Engagement:** Web visits, acceptance of promotional campaigns, complaints.
- **Derived Features:** Age, total purchases, and promotional engagement.

## Project Structure

- `customers-personality-clustering.ipynb`: Jupyter Notebook containing data preprocessing, exploratory data analysis (EDA), clustering, and evaluation.
- `Customers Personality Clustering.pdf`: A detailed report covering the objectives, business questions, methodology, and findings of the project.

## Methodology

1. **Exploratory Data Analysis (EDA)**

   - Understanding the distribution of customer attributes.
   - Analyzing relationships between different features.



2. **Data Preprocessing**

   - Handling missing values and outliers.
   - Encoding categorical variables and scaling numerical features.
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

