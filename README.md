

# Customer Segmentation Using K-Means Clustering





Customer segmentation is a critical part of marketing strategy for retail stores. By dividing customers into groups based on their behavior and demographics, businesses can tailor their offerings and improve customer satisfaction. In this project, K-Means clustering is used to segment customers based on their **Age**, **Annual Income**, and **Spending Score**.

## Dataset

The dataset used in this project is the "Mall Customers" dataset, which contains demographic information and purchasing behavior of 200 customers. You can download it from the following Kaggle link:

- **Dataset**: [Mall_Customers.csv](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## Features

- **K-Means Clustering**: Used to segment customers into different clusters.
- **Data Preprocessing**: Handles missing data, data cleaning, and normalization.
- **Visualization**: Visualizes clusters to gain insights into customer segments.
- **User Input**: Allows the user to input a customer's data (age, income, spending score) and find the cluster they belong to.

## Requirements

- Python 3.7+
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

### Installing Dependencies

1. Clone this repository:
    
    git clone https://github.com/your-username/customer-segmentation-kmeans.git
    cd customer-segmentation-kmeans
    

2. Create and activate a virtual environment (optional but recommended):
    
    python3 -m venv env
    source env/bin/activate  # For Mac/Linux
    env\Scripts\activate     # For Windows
    

3. Install the required dependencies:
    
    pip install -r requirements.txt
    

## How to Run

### Data Loading

1. Download the **Mall_Customers.csv** dataset and place it in the project folder.
   
2. Load the data and perform basic information checks:

    ```python
    import pandas as pd
    data = pd.read_csv("Mall_Customers.csv")
    print(data.head())
    print(data.info())
    ```

### K-Means Clustering

1. Extract the relevant features from the dataset and apply K-Means clustering:

    ```python
    from sklearn.cluster import KMeans
    
    X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Elbow method to determine optimal clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    # Plot Elbow Method
    plt.plot(range(1, 11), wcss, marker="o", color='blue')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.show()
    ```

2. Using the optimal number of clusters (based on the elbow method), fit the K-Means model:

    ```python
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(X)
    data['Cluster'] = clusters
    ```

### Visualizing the Clusters

1. Visualize the customer segments based on their **Annual Income** and **Spending Score**:

    ```python
    plt.figure(figsize=(10, 6))
    for cluster_num in range(5):
        cluster_data = data[data['Cluster'] == cluster_num]
        plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                    label=f'Cluster {cluster_num}', cmap='viridis', marker='o', edgecolors='black', s=100)

    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    ```

### Predicting User's Cluster

1. After training the model, the user can input data (age, annual income, spending score) and predict the cluster the customer belongs to:

    ```python
    age = float(input("Enter customer's age: "))
    income = float(input("Enter customer's annual income (in k$): "))
    spending_score = float(input("Enter customer's spending score (1-100): "))

    user_data = pd.DataFrame({'Age': [age], 'Annual Income (k$)': [income], 'Spending Score (1-100)': [spending_score]})
    user_cluster = kmeans.predict(user_data)[0]
    print("The user's data belongs to Cluster:", user_cluster)
    ```

### Example Output

```bash
Enter customer's age: 20
Enter customer's annual income (in k$): 50
Enter customer's spending score (1-100): 40
The user's data belongs to Cluster: 1
```

