![image](https://github.com/user-attachments/assets/c2b2da51-f1d1-4d22-b3d9-c9d47b3110e7)


```
📦 Data Science Internship Project
│
├── 📄 LICENSE
├── 📄 README.md
│
├── 📁 LEVEL 1 - Data Exploration and Preprocessing
│   ├── 📄 DATAEXPLORATION AND PREPROCESSING.ipynb
│   ├── 📄 DESCRIPTIVE ANALYSIS.ipynb
│   ├── 📄 GEOSPATIAL ANALYSIS.ipynb
│   └── 🌐 restaurant_map.html
│
├── 📁 LEVEL 2 - Advanced Analysis
│   ├── 📄 TABLE BOOKING AND ONLINE DELEIVERY.ipynb
│   ├── 📄 PRICE RANGE ANALYSIS.ipynb
│   └── 📄 FEATURE ENGINEERING.ipynb
│
├── 📁 LEVEL 3 - Modeling and Visualization
│   ├── 📄 PREDICTIVE MODELING.ipynb
│   ├── 📄 CUSTOMER PREFERANCE ANALYSIS.ipynb
│   ├── 📄 Data Visualization.ipynb
│   ├── 📊 average_rating_by_cuisine.png
│   ├── 📊 boxplot_ratings_by_cuisine.png
│   ├── 📊 correlation_heatmap.png
│   ├── 📊 jointplot_votes_rating.png
│   ├── 📊 pair_plot.png
│   ├── 📊 pairplot_votes_rating.png
│   ├── 📊 rating_distribution_boxplot.png
│   ├── 📊 rating_distribution_histogram.png
│   ├── 📊 swarmplot_ratings_cuisines.png
│   ├── 📊 top_cuisines_avg_rating.png
│   ├── 📊 violinplot_votes_by_rating.png
│   ├── 📊 votes_vs_aggregate_rating.png
│   └── 🌐 bubble_chart_votes_rating.html
│
└── 📁 DATASETS
    └── (Dataset files)

```


## 📚 Libraries Used

- **Folium** 🗺️: For creating interactive maps to visualize restaurant locations.
- **Pandas** 🐼: For data manipulation and processing.
- **Matplotlib** 📊: For plotting static graphs to visualize restaurant distributions.
- **Seaborn** 🎨: For enhanced visualizations and creating scatter plots.
- **Scikit-learn** 🤖: For applying machine learning algorithms like KMeans clustering to group restaurant locations.

🚀 **Workflow Overview :**

### **1. Data Loading and Preprocessing** 🧹

We begin by loading the data and cleaning it. The dataset contains several cuisines with aggregate ratings, votes, and cuisine names. We clean the data, fill missing values, and prepare it for further analysis.

```python
import pandas as pd
import numpy as np

# Load the dataset
cuisine_data = pd.read_csv("Cuisine_Rating_Votes.csv")

# Fill missing values
cuisine_data.fillna(method='ffill', inplace=True)

# Summary of the dataset
cuisine_data.info()
```

- **Missing Values Handling**: The `fillna(method='ffill')` method is used to forward-fill any missing values.
- **Dataset Overview**: We get a basic overview of the dataset with `info()` to understand its structure.

---

### **2. Exploratory Data Analysis (EDA)** 🔍

#### **Cuisines with Consistent Ratings** 💯

Next, we identify the cuisines that have consistent ratings by calculating the standard deviation of the aggregate ratings.

```python
# Calculate standard deviation of ratings for each cuisine
rating_std = cuisine_data.groupby('Cuisines')['Aggregate rating'].std()

# Cuisines with lowest standard deviation (consistent ratings)
consistent_cuisines = rating_std.sort_values().head(10)
```

- **Consistent Ratings**: Cuisines like `Italian`, `Hawaiian`, and `American` are identified as having the most consistent ratings, with low standard deviation.

#### **Top Cuisines by Average Rating** 🌟

We then calculate the average rating for each cuisine to find out which ones have the best average rating.

```python
# Calculate the average rating by cuisine
avg_rating_by_cuisine = cuisine_data.groupby('Cuisines')['Aggregate rating'].mean()

# Top 10 cuisines with highest average ratings
top_cuisines = avg_rating_by_cuisine.sort_values(ascending=False).head(10)
```

- **Top Cuisines**: This code highlights the cuisines with the highest average ratings, such as `Italian`, `Hawaiian`, and `American`.

#### **Cuisines Rated by the Most People** 👥

We now identify which cuisines have the most number of ratings, as more ratings usually indicate more popularity.

```python
# Count the number of ratings for each cuisine
ratings_count = cuisine_data.groupby('Cuisines')['Votes'].sum()

# Top 10 cuisines rated by the most people
top_cuisines_by_votes = ratings_count.sort_values(ascending=False).head(10)
```

- **Most Rated Cuisines**: The most rated cuisines are those that have the highest number of votes, such as `American` and `Italian`.

---

### **3. Data Visualization** 📊

#### **Distribution of Aggregate Ratings** 📉

We visualize the distribution of ratings using a histogram to see the overall spread.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Histogram for Aggregate Ratings
sns.histplot(cuisine_data['Aggregate rating'], kde=True)
plt.title('Distribution of Aggregate Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()
```

- **Histogram**: The histogram shows the distribution of ratings across all cuisines, with a clear concentration of ratings between 4 and 5.

#### **Votes vs. Aggregate Rating** 📈

We use a scatter plot to visualize how the number of votes relates to the aggregate ratings.

```python
sns.scatterplot(x=cuisine_data['Votes'], y=cuisine_data['Aggregate rating'])
plt.title('Votes vs. Aggregate Rating')
plt.xlabel('Number of Votes')
plt.ylabel('Aggregate Rating')
plt.show()
```

- **Scatter Plot**: The plot shows that as the number of votes increases, the aggregate rating generally increases, with some outliers.

#### **Cuisines with the Most Consistent Ratings** 📏

We create a bar plot to display the cuisines with the most consistent ratings.

```python
sns.barplot(x=consistent_cuisines.index, y=consistent_cuisines.values)
plt.title('Cuisines with Most Consistent Ratings')
plt.xlabel('Cuisine')
plt.ylabel('Standard Deviation of Ratings')
plt.xticks(rotation=90)
plt.show()
```

- **Bar Plot**: The plot highlights the top cuisines with the lowest standard deviations in their ratings, indicating consistency.

---

### **4. Clustering Cuisines** 🤖

We apply KMeans clustering to group cuisines based on their `Votes` and `Aggregate rating` values. This allows us to find patterns in how cuisines are rated and voted upon.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Select relevant features for clustering
X = cuisine_data[['Votes', 'Aggregate rating']]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
cuisine_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizing the clusters
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=cuisine_data['Cluster'], palette='viridis')
plt.title('Clustering of Cuisines Based on Votes and Ratings')
plt.xlabel('Normalized Votes')
plt.ylabel('Normalized Aggregate Rating')
plt.show()
```

- **Clustering**: We use KMeans clustering to categorize cuisines into three groups based on their vote count and rating.
- **Visualization**: The scatter plot visualizes how different cuisines are clustered based on these features.

---

### **5. Insights and Summary** 💡

From the analysis, we gain the following insights:

- **Top Rated Cuisines**: `Italian`, `American`, and `Hawaiian` are among the top-rated cuisines.
- **Consistency**: Cuisines with low standard deviation in ratings, like `Italian`, `American`, and `Mexican`, are highly consistent in their ratings.
- **Popularity**: Cuisines with the most votes are generally those that have more global recognition, such as `Italian` and `American`.
- **Cluster Groupings**: Clustering based on `Votes` and `Aggregate rating` reveals that cuisines like `Italian` and `Mexican` form their own clusters based on higher ratings and votes.

---

## 🛠️ **Libraries Used**

- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For static and visualizations.
- **Scikit-learn**: For clustering techniques.
- **NumPy**: For numerical operations.
- **Plotly**: For creating interactive visualizations.


This provides an in-depth analysis of cuisine ratings, votes, and how they correlate with each other. By using clustering techniques, we uncover hidden patterns and gain insights into which cuisines are consistently rated highly and which ones are most popular based on votes. The combination of data cleaning, EDA, and clustering makes this analysis a comprehensive exploration of the cuisine ratings dataset.




To include the image from your GitHub repository and create a small-size dashboard for **Vites vs Aggregate Rating** in your README, here's how you can modify it:

---

### 📊 **Vites vs Aggregate Rating Dashboard** 🚀

The following visualization showcases the relationship between the **number of votes (Vites)** and **Aggregate Rating** for each cuisine. It helps us understand how higher ratings correlate with more votes, providing insights into the popularity and consistency of cuisines.

#### **Visualization** 🖼️

You can view the plot below, which visualizes the correlation between `Votes` and `Aggregate Rating` for each cuisine:

![Vites vs Aggregate Rating](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/Capture.PNG?raw=true)



## 📊 **Data Visualization Gallery** 🚀

### Here are various visualizations for better understanding of data:

| ![Average Rating by Cuisine](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/average_rating_by_cuisine.png?raw=true) | ![Boxplot Ratings by Cuisine](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/boxplot_ratings_by_cuisine.png?raw=true) |
| --- | --- |
| **Average Rating by Cuisine** | **Boxplot Ratings by Cuisine** |

| ![Correlation Heatmap](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/correlation_heatmap.png?raw=true) | ![Jointplot Votes vs Rating](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/jointplot_votes_rating.png?raw=true) |
| --- | --- |
| **Correlation Heatmap** | **Jointplot Votes vs Rating** |


| ![Pair Plot](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/pair_plot.png?raw=true) | ![Pairplot Votes vs Rating](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/pairplot_votes_rating.png?raw=true) |
| --- | --- |
| **Pair Plot** | **Pairplot Votes vs Rating** |


| ![Rating Distribution Boxplot](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/rating_distribution_boxplot.png?raw=true) | ![Rating Distribution Histogram](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/rating_distribution_histogram.png?raw=true) |
| --- | --- |
| **Rating Distribution Boxplot** | **Rating Distribution Histogram** |

| ![Swarmplot Ratings by Cuisines](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/swarmplot_ratings_cuisines.png?raw=true) | ![Top Cuisines Average Rating](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/top_cuisines_avg_rating.png?raw=true) |
| --- | --- |
| **Swarmplot Ratings by Cuisines** | **Top Cuisines Average Rating** |


| ![Violinplot Votes by Rating](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/violinplot_votes_by_rating.png?raw=true) | ![Votes vs Aggregate Rating](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/votes_vs_aggregate_rating.png?raw=true) |
| --- | --- |
| **Violinplot Votes by Rating** | **Votes vs Aggregate Rating** |

| ![Votes vs Rating Scatter](https://github.com/rubydamodar/Cognifyz-Data-Mastery-Program/blob/main/LEVEL%203%20TASK%203%20Data%20Visualization/votes_vs_rating_scatter.png?raw=true) |  |
| --- | --- |
| **Votes vs Rating Scatter** |  |

