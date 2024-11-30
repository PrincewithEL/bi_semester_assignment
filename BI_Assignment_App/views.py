from django.shortcuts import render
from django.db import connection
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from wordcloud import WordCloud
# from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from decimal import Decimal
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
matplotlib.use("Agg")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_data(query):
    """Load data using raw SQL query."""
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
    return pd.DataFrame(rows, columns=columns)

from collections import Counter

# def generate_charts(request):
#     # LDA Word Cloud Data
#     products = load_data("SELECT productdescription FROM products")
#     text_data = " ".join(products['productdescription'].dropna().tolist())
#     word_frequencies = Counter(text_data.split())
#     wordcloud_data = [{"word": word, "weight": count} for word, count in word_frequencies.items() if count > 2]

#     # Load Data
#     customers = load_data("SELECT customernumber, creditlimit FROM customers")
#     payments = load_data("SELECT customernumber, SUM(amount) AS total_payments FROM payments GROUP BY customernumber")
#     clv_data = pd.merge(customers, payments, on="customernumber", how="left").fillna(0)
#     clv_data["creditlimit"] = clv_data["creditlimit"].astype(float)
#     clv_data["total_payments"] = clv_data["total_payments"].astype(float)

#     # K-Means Clustering
#     kmeans = KMeans(n_clusters=3, random_state=42)
#     clv_data["Cluster"] = kmeans.fit_predict(clv_data[["creditlimit", "total_payments"]])

#     # Plot the clusters
#     plt.figure(figsize=(8, 6))
#     for cluster in clv_data["Cluster"].unique():
#         cluster_data = clv_data[clv_data["Cluster"] == cluster]
#         plt.scatter(cluster_data["creditlimit"], cluster_data["total_payments"], label=f"Cluster {cluster}")
#     plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", marker="X", s=200, label="Centroids")
#     plt.xlabel("Credit Limit")
#     plt.ylabel("Total Payments")
#     plt.title("K-Means Clustering")
#     plt.legend()

#     # Save the plot to a Base64 string
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", bbox_inches="tight")
#     buf.seek(0)
#     image_base64 = base64.b64encode(buf.read()).decode("utf-8")
#     buf.close()

#     # Logistic Regression for Price Optimization
#     order_details = load_data("SELECT quantityordered, priceeach FROM orderdetails")
#     order_details['high_volume_purchase'] = (
#         order_details['quantityordered'] > order_details['quantityordered'].median()
#     ).astype(int)

#     X = order_details[["priceeach", "quantityordered"]]
#     y = order_details["high_volume_purchase"]

#     if len(y.unique()) > 1:  # Proceed only if there are at least two classes
#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )

#         # Balance classes using SMOTE
#         smote = SMOTE(random_state=42)
#         X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#         # Train Logistic Regression
#         logreg = LogisticRegression()
#         logreg.fit(X_train_resampled, y_train_resampled)

#         # Evaluate the model
#         y_prob = logreg.predict_proba(X_test)[:, 1]
#         fpr, tpr, _ = roc_curve(y_test, y_prob)
#         roc_auc = auc(fpr, tpr)
#     else:
#         # Default values if only one class is present
#         fpr, tpr, roc_auc = [], [], 0.0

#     # Prepare data for JavaScript
#     charts_data = {
#         "lda_wordcloud": wordcloud_data,
#         "kmeans": clv_data[["creditlimit", "total_payments", "Cluster"]].to_dict(orient="records"),
#         "kmeans_image": image_base64,
#         "logreg_roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc},
#     }
#     return render(request, "charts.html", {"charts_data": json.dumps(charts_data)})

def business_insights_view(request):
    # Load data
    orders = load_data("SELECT * FROM orders")
    orderdetails = load_data("SELECT * FROM orderdetails")
    customers = load_data("SELECT * FROM customers")
    employees = load_data("SELECT * FROM employees")
    products = load_data("SELECT * FROM products")

    # Calculate total revenue
    orderdetails['revenue'] = orderdetails['quantityordered'] * orderdetails['priceeach']
    total_revenue = orderdetails['revenue'].sum()

    # Average order size: average quantity per order
    average_order_size = orderdetails.groupby('ordernumber')['quantityordered'].sum().mean()

    # Average customer spend: total revenue divided by number of unique customers
    revenue_by_customer = orderdetails.merge(orders, on='ordernumber').groupby('customernumber')['revenue'].sum()
    average_customer_spend = revenue_by_customer.mean()

    # Top-selling products by total revenue
    top_selling_products = (
        orderdetails.groupby('productcode')['revenue']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .merge(products[['productcode', 'productname']], on='productcode')
    )

    # Product line performance: total revenue per product line
    product_line_revenue = (
        orderdetails.merge(products, on='productcode')
        .groupby('productline')['revenue']
        .sum()
        .reset_index()
    )

    # Customer purchase frequency: average orders per customer
    customer_order_count = orders.groupby('customernumber').size()
    avg_orders_per_customer = customer_order_count.mean()

    # Payment patterns: average payment amount per customer
    payments = load_data("SELECT * FROM payments")
    avg_payment_per_customer = payments.groupby('customernumber')['amount'].sum().mean()

    # Monthly revenue trend
    orders['orderdate'] = pd.to_datetime(orders['orderdate'])
    orders['year_month'] = orders['orderdate'].dt.to_period('M')
    monthly_revenue = (
        orders.merge(orderdetails, on='ordernumber')
        .groupby('year_month')['revenue']
        .sum()
        .reset_index()
    )

    # Convert 'year_month' to string for JSON serialization
    monthly_revenue['year_month'] = monthly_revenue['year_month'].astype(str)

    # Fulfillment time metrics
    orders['shippeddate'] = pd.to_datetime(orders['shippeddate'], errors='coerce')
    orders['requireddate'] = pd.to_datetime(orders['requireddate'], errors='coerce')
    orders['fulfillment_time'] = (orders['shippeddate'] - orders['requireddate']).dt.days
    avg_fulfillment_time = orders['fulfillment_time'].mean()

    # Employee performance metrics
    orders_with_sales_rep = orders.merge(customers[['customernumber', 'salesrepemployeenumber']], on='customernumber', how='left')
    employee_sales = orders_with_sales_rep.merge(orderdetails, on='ordernumber').groupby('salesrepemployeenumber')['revenue'].sum()
    employee_performance = pd.DataFrame({
        'sales': employee_sales,
        'avg_fulfillment_time': orders_with_sales_rep.groupby('salesrepemployeenumber')['fulfillment_time'].mean()
    }).sort_values(by='sales', ascending=False).reset_index()

    # Convert employee performance to float as well
    employee_performance = employee_performance.applymap(lambda x: float(x) if isinstance(x, (int, float, Decimal)) else x)
    employee_performance_json = employee_performance.head(10).to_dict(orient='records')

    # Convert Decimal to float in the data being passed to the template
    top_selling_products = top_selling_products.astype({'revenue': 'float'}).to_dict(orient='records')
    product_line_revenue = product_line_revenue.astype({'revenue': 'float'}).to_dict(orient='records')
    monthly_revenue = monthly_revenue.astype({'revenue': 'float'}).to_dict(orient='records')

    # Merge orders and orderdetails to get full purchase history per customer
    customer_purchase_history = orderdetails.merge(orders, on='ordernumber')
    customer_purchase_history = customer_purchase_history[['customernumber', 'productcode', 'quantityordered']]
    
    # Merge with the products table to get product names
    customer_purchase_history = customer_purchase_history.merge(products[['productcode', 'productname']], on='productcode', how='left')

    # Create a pivot table where rows are customers and columns are product names
    purchase_matrix = customer_purchase_history.pivot_table(index='customernumber', columns='productname', values='quantityordered', fill_value=0)

    # Normalize the purchase matrix to improve KNN performance
    scaler = StandardScaler()
    normalized_purchase_matrix = pd.DataFrame(scaler.fit_transform(purchase_matrix), columns=purchase_matrix.columns, index=purchase_matrix.index)
    
    # Get customer IDs for dropdown
    customer_ids = list(purchase_matrix.index)

    # Default to a target customer (e.g., 119) if no selection is made
    target_customer = request.GET.get('customer', 119)  # Get selected customer ID

    if target_customer and int(target_customer) in purchase_matrix.index:
        target_customer = int(target_customer)

        # Apply KNN to find similar customers
        knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn.fit(normalized_purchase_matrix)
        
        # Find the top 5 most similar customers to the target customer
        distances, indices = knn.kneighbors(normalized_purchase_matrix.loc[[target_customer]])

        # Extract the product codes purchased by the similar customers
        similar_customers = purchase_matrix.iloc[indices[0]]
        similar_customers = similar_customers.T
        
        # Identify products that the target customer has not purchased, but similar customers have
        target_customer_purchases = purchase_matrix.loc[target_customer]
        products_recommended = similar_customers.loc[similar_customers.sum(axis=1) > 0].index.difference(target_customer_purchases.index[target_customer_purchases > 0])

        recommended_products_data = {
            'product_names': products_recommended.tolist(),
            'quantity': [1] * len(products_recommended),  # Dummy data for charting
        }

        # Pie chart data: Distribution of recommended products
        recommendation_counts = similar_customers.loc[products_recommended].sum(axis=1)
        pie_data = {
            'labels': products_recommended.tolist(),
            'data': recommendation_counts.tolist(),
        }

        # Radar chart data: Top recommended products for radar chart visualization
        radar_data = {
            'labels': products_recommended.tolist(),
            'values': recommendation_counts.nlargest(5).values.tolist(),
        }

        # Heatmap data (simplified): We can create a 2D list for the heatmap
        heatmap_data = similar_customers.T.values.tolist()

        # Calculate Recency, Frequency, and Monetary values
        last_order_date = orders.groupby('customernumber')['orderdate'].max()
        last_order_date = pd.to_datetime(last_order_date)
        current_date = pd.to_datetime('today')

        # Calculate Recency (in days)
        recency = (current_date - last_order_date).dt.days

        # Calculate Frequency (number of orders)
        frequency = orders.groupby('customernumber').size()

        # Calculate Monetary (total spending)
        monetary = orderdetails.merge(orders, on='ordernumber').groupby('customernumber')['revenue'].sum()

        # Combine RFM metrics into one DataFrame
        rfm = pd.DataFrame({
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary
        })

        # Standardize the data before clustering
        scaler = StandardScaler()
        rfm_scaled = pd.DataFrame(scaler.fit_transform(rfm), columns=rfm.columns)

        # Apply K-means clustering to segment customers
        kmeans = KMeans(n_clusters=4, random_state=42)
        rfm['segment'] = kmeans.fit_predict(rfm_scaled)

        # Linear Regression for Predicted Lifetime Value
        X = rfm[['frequency', 'monetary']]
        y = rfm['recency']  # Using recency as a proxy for LTV
        lr = LinearRegression()
        lr.fit(X, y)
        rfm['predicted_LTV'] = lr.predict(X)

        # Convert 'monetary' to float to avoid serialization issues
        rfm['monetary'] = rfm['monetary'].astype(float)

        # Prepare the data to be sent to the frontend (JSON format)
        segment_data = {
            'recency': rfm['recency'].tolist(),
            'frequency': rfm['frequency'].tolist(),
            'monetary': rfm['monetary'].tolist(),
            'predicted_LTV': rfm['predicted_LTV'].tolist(),
            'segments': rfm['segment'].tolist()
        }

        # Get the selected product line from the request
        product_line = request.GET.get('product_line', 'Planes')  # Default to 'Planes' if not specified

        # Merge orders and orderdetails to get full purchase history per customer
        customer_purchase_history = pd.merge(orderdetails, orders, on='ordernumber', how='left')
        customer_purchase_history = customer_purchase_history[['customernumber', 'productcode', 'quantityordered', 'orderdate']]

        # Convert orderDate to datetime and extract year and month
        customer_purchase_history['orderdate'] = pd.to_datetime(customer_purchase_history['orderdate'])
        customer_purchase_history['year_month'] = customer_purchase_history['orderdate'].dt.to_period('M')

        # Aggregate product demand over time (e.g., monthly)
        product_demand = customer_purchase_history.groupby(['year_month', 'productcode'])['quantityordered'].sum().reset_index()

        # Get the products for the selected product line
        product_line_products = products[products['productline'] == product_line]['productcode']
        product_line_demand = product_demand[product_demand['productcode'].isin(product_line_products)]

        # Create pivot table: rows are months, columns are product codes
        demand_pivot = product_line_demand.pivot_table(
            index='year_month', columns='productcode', values='quantityordered', aggfunc='sum', fill_value=0
        )

        # Ensure 'year_month' is kept as Period for processing and a separate string version is used for serialization
        demand_pivot.index = demand_pivot.index.to_timestamp()  # Convert PeriodIndex to Timestamp for operations

        # Prepare data for forecasting
        X = np.array([i.month for i in demand_pivot.index]).reshape(-1, 1)  # Month as feature
        last_period = demand_pivot.index[-1]
        future_months = np.array([i.month for i in pd.date_range(start=last_period, periods=12, freq='M')]).reshape(-1, 1)

        # Predict demand for each product
        predicted_demand = {}
        for product in demand_pivot.columns:
            y = demand_pivot[product].values
            model = LinearRegression()
            model.fit(X, y)
            predicted_demand[product] = model.predict(future_months).tolist()

        # Convert demand_pivot index to string for JSON serialization
        demand_pivot.index = demand_pivot.index.strftime('%Y-%m')

        # Generate forecast dates for serialization
        forecast_dates = pd.date_range(start=last_period, periods=12, freq='M').strftime('%b %Y').tolist()

        # Preprocess the comments column
        comments = orders['comments'].dropna()  # Drop missing comments
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def preprocess_text(text):
            text = text.lower()  # Lowercase
            text = ''.join([char for char in text if char.isalpha() or char.isspace()])  # Remove special characters
            words = text.split()  # Tokenize
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
            return ' '.join(words)

        cleaned_comments = comments.apply(preprocess_text)

        # Vectorize comments
        vectorizer = CountVectorizer(max_features=1000)
        X = vectorizer.fit_transform(cleaned_comments)

        # Perform LDA
        n_topics = 5
        lda = LDA(n_components=n_topics, random_state=42)
        lda.fit(X)

        # Extract topics and top words
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]  # Top 10 words per topic
            topics.append({'topic': f"Topic {topic_idx + 1}", 'words': top_words})

        # Prepare word cloud data for D3.js
        word_cloud_data = []
        for topic_idx, topic in enumerate(lda.components_):
            word_freq = {feature_names[i]: topic[i] for i in topic.argsort()[:-11:-1]}
            word_cloud_data.append({'topic': f"Topic {topic_idx + 1}", 'words': word_freq})

        # Prepare topic distribution data for visualization
        topic_distribution = lda.transform(X).tolist()

    context = {
        "total_revenue": float(total_revenue),
        "average_order_size": float(average_order_size),
        "average_customer_spend": float(average_customer_spend),
        "top_selling_products": json.dumps(top_selling_products),
        "product_line_revenue": json.dumps(product_line_revenue),
        "avg_orders_per_customer": float(avg_orders_per_customer),
        "avg_payment_per_customer": float(avg_payment_per_customer),
        "monthly_revenue_trend": json.dumps(monthly_revenue),
        "avg_fulfillment_time": float(avg_fulfillment_time),
        "employee_performance": json.dumps(employee_performance_json),
        "employee_performance1": employee_performance.head(10).to_dict(orient='records'),
        "customer_ids": customer_ids,
        'recommended_products_data': json.dumps(recommended_products_data),
        "pie_data": json.dumps(pie_data),
        "radar_data": json.dumps(radar_data),
        "heatmap_data": json.dumps(heatmap_data),
        "target_customer": target_customer,
        "segment_data": json.dumps(segment_data),
        "demand_data": json.dumps(demand_pivot.reset_index().to_dict(orient='list')),  # Serialize pivot data
        "forecast_dates": json.dumps(forecast_dates),  # Serialize forecast dates
        "predicted_demand": json.dumps(predicted_demand),  # Serialize predicted demand
        "product_line": product_line,
        "topics": json.dumps(topics),
        "word_cloud_data": json.dumps(word_cloud_data),
        "topic_distribution": json.dumps(topic_distribution),
        "product_lines": products['productline'].unique()
    }

    return render(request, "business_insights.html", context)