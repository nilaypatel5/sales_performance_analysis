# Sales Performance Analysis: Leveraging Customer Personality Insights

This project focuses on customer personality analysis to optimize sales performance. By segmenting customers, businesses can target specific groups, thereby improving marketing efficiency and increasing sales. We used clustering techniques and the Apriori algorithm to identify key customer segments and their buying behaviors.

## Getting Started

### Prerequisites
Make sure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `sklearn`
- `mlxtend`
- `dataprep`

### Dataset
Load the dataset (`sales_data.csv`):
```python
import pandas as pd

data = pd.read_csv('sales_data.csv')
data.head()
```

### Data Preparation
Create new features and clean the data:
```python
data['Age'] = 2014 - data['Year_Birth']
data['Spending'] = data[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
data['Seniority'] = (pd.to_datetime('2014-10-04') - pd.to_datetime(data['Dt_Customer'], dayfirst=True)).dt.days / 30

data['Marital_Status'] = data['Marital_Status'].replace({
    'Divorced': 'Alone', 'Single': 'Alone', 'Married': 'In couple',
    'Together': 'In couple', 'Absurd': 'Alone', 'Widow': 'Alone', 'YOLO': 'Alone'
})

data['Education'] = data['Education'].replace({
    'Basic': 'Undergraduate', '2n Cycle': 'Undergraduate',
    'Graduation': 'Postgraduate', 'Master': 'Postgraduate', 'PhD': 'Postgraduate'
})

data['Children'] = data['Kidhome'] + data['Teenhome']
data['Has_child'] = data['Children'].apply(lambda x: 'Has child' if x > 0 else 'No child')
```

### Clustering
Cluster customers into 4 segments:
```python
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture

scaler = StandardScaler()
X = normalize(scaler.fit_transform(data[['Income', 'Seniority', 'Spending']]), norm='l2')

gmm = GaussianMixture(n_components=4, covariance_type='spherical', max_iter=2000, random_state=5).fit(X)
data['Cluster'] = gmm.predict(X)

data['Cluster'] = data['Cluster'].replace({0: 'Need attention', 1: 'Stars', 2: 'High potential', 3: 'Leaky bucket'})
```

### Apriori Algorithm
Identify associations using the Apriori algorithm:
```python
from mlxtend.frequent_patterns import apriori, association_rules

data_dummies = pd.get_dummies(data)
frequent_items = apriori(data_dummies, min_support=0.08, use_colnames=True, max_len=10)
rules = association_rules(frequent_items, metric='lift', min_threshold=1)

product = 'Wines'
segment = 'Biggest consumer'
target = f"{{'{product}_segment_{segment}'}}"
results = rules[rules['consequents'].astype(str).str.contains(target)].sort_values(by='confidence', ascending=False)
print(results.head())
```

### Conclusion
From the analysis, we identified that the biggest consumers of wines are typically:
- Customers with an average income of ~$69,500.
- Customers with an average total spend of ~$1,252.
- Customers registered for ~21 months.
- Customers with a graduate degree.
- Heavy consumers of meat products.

This segmentation helps businesses target the most promising customers, improving marketing efficiency and sales performance.
