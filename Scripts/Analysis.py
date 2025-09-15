import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------
# Streamlit setup
st.set_page_config(page_title="Zomato Dashboard", layout="wide")
st.title("🍽️ Zomato Restaurants Analysis & Rating Prediction")

# -----------------------
# Load dataset
df = pd.read_csv("Data/zomato.csv", encoding="latin1")
st.success("Data loaded successfully!")

# -----------------------
# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

# Clean ratings
df.rename(columns={"aggregate_rating": "rating"}, inplace=True)
df["rating"] = df["rating"].apply(lambda x: float(x) if pd.notnull(x) else np.nan)

# Clean cost
df["average_cost_for_two"] = df["average_cost_for_two"].astype(str).str.replace(",", "").astype(float)

# Fill missing values
df["cuisines"].fillna("Unknown", inplace=True)
df["rating"].fillna(df["rating"].mean(), inplace=True)

# -----------------------
# Sidebar filters
st.sidebar.header("Filter Restaurants")
selected_city = st.sidebar.selectbox("City", np.append("All", df["city"].unique()))
selected_cuisine = st.sidebar.selectbox("Cuisine", np.append("All", df["cuisines"].unique()))

filtered_df = df.copy()
if selected_city != "All":
    filtered_df = filtered_df[filtered_df["city"] == selected_city]
if selected_cuisine != "All":
    filtered_df = filtered_df[filtered_df["cuisines"] == selected_cuisine]

st.subheader(f"Filtered Restaurants ({filtered_df.shape[0]} rows)")
st.dataframe(filtered_df[["restaurant_name", "city", "cuisines", "average_cost_for_two", "rating", "votes"]])

# -----------------------
# Seaborn plots
st.subheader("Distribution of Ratings")
plt.figure(figsize=(8,5))
sns.countplot(x="rating", data=filtered_df)
plt.title("Distribution of Ratings")
st.pyplot(plt.gcf())
plt.clf()

st.subheader("Distribution of Rating Text")
plt.figure(figsize=(8,5))
sns.countplot(x="rating_text", data=filtered_df, order=filtered_df["rating_text"].value_counts().index)
plt.title("Distribution of Rating Text")
plt.xticks(rotation=45)
st.pyplot(plt.gcf())
plt.clf()

st.subheader("Top 10 Cuisines")
top_cuisines = filtered_df["cuisines"].value_counts().head(10)
plt.figure(figsize=(8,5))
sns.barplot(y=top_cuisines.index, x=top_cuisines.values, palette="viridis")
plt.title("Top 10 Cuisines")
st.pyplot(plt.gcf())
plt.clf()

st.subheader("Top 10 Cities by Restaurant Count")
top_cities = df["city"].value_counts().head(10)
plt.figure(figsize=(8,5))
sns.barplot(y=top_cities.index, x=top_cities.values, palette="magma")
plt.title("Top 10 Cities")
st.pyplot(plt.gcf())
plt.clf()

st.subheader("Online Order Availability")
plt.figure(figsize=(6,4))
sns.countplot(x="has_online_delivery", data=filtered_df, palette="coolwarm")
plt.title("Online Order Availability")
st.pyplot(plt.gcf())
plt.clf()

# -----------------------
# Prepare data for ML
X = df[["average_cost_for_two", "votes", "city", "cuisines"]].copy()
y = df["rating"]

X["average_cost_for_two"].fillna(X["average_cost_for_two"].mean(), inplace=True)
X["votes"].fillna(X["votes"].mean(), inplace=True)
X["city"].fillna("Unknown", inplace=True)
X["cuisines"].fillna("Unknown", inplace=True)

le_city = LabelEncoder()
le_cuisine = LabelEncoder()
X["city"] = le_city.fit_transform(X["city"])
X["cuisines"] = le_cuisine.fit_transform(X["cuisines"].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
st.write(f"R^2 Score: {r2_score(y_test, y_pred):.4f}")

# -----------------------
# Insights
st.subheader("Top 5 Cities by Average Rating")
top_cities_rating = df.groupby("city")["rating"].mean().sort_values(ascending=False).head(5)
st.write(top_cities_rating)

st.subheader("Top 5 Cuisines by Average Rating")
top_cuisines_rating = df.groupby("cuisines")["rating"].mean().sort_values(ascending=False).head(5)
st.write(top_cuisines_rating)

# -----------------------
# Predict rating
st.subheader("Predict Restaurant Rating")
st.write("Enter restaurant features to predict rating:")

input_avg_cost = st.number_input("Average Cost for Two", value=500)
input_votes = st.number_input("Votes", value=50)
input_city = st.selectbox("City", df["city"].unique(), key="predict_city")
input_cuisine = st.selectbox("Cuisine", df["cuisines"].unique(), key="predict_cuisine")

if st.button("Predict Rating"):
    input_df = pd.DataFrame({
        "average_cost_for_two": [input_avg_cost],
        "votes": [input_votes],
        "city": le_city.transform([input_city]),
        "cuisines": le_cuisine.transform([input_cuisine])
    })
    predicted_rating = model.predict(input_df)[0]
    st.success(f"Predicted Rating: {predicted_rating:.2f}")
