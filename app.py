import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Used Car Market Price Prediction & Car Selection Assistant", 
                   layout="wide")
st.title("Used Car Market Price Prediction & Car Selection Assistant")

# -------------------------------
# Load & Clean Data
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("malaysia_used_cars.csv", on_bad_lines='skip')
    except:
        # Fallback: read with different settings
        df = pd.read_csv("malaysia_used_cars.csv", 
                         on_bad_lines='skip',
                         quoting=1,    # QUOTE_ALL
                         encoding='utf-8')

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # Extract year from description
    df["year"] = df["description"].str.extract(r"(\d{4})")[0]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Car age
    df["car_age"] = 2026 - df["year"]

    # Clean price
    df["list_price"] = df["list_price"].astype(str).str.replace("\n", " ", regex=True)
    price_extract = df["list_price"].str.extract(r"(\d[\d,]*\.?\d*)")[0]
    df["list_price"] = pd.to_numeric(price_extract.str.replace(",", ""), errors="coerce")

    # Clean mileage
    df["milleage"] = df["milleage"].astype(str)
    mile_num = df["milleage"].str.extract(r"(\d+(?:\.\d+)?)")[0]
    mile_num = pd.to_numeric(mile_num, errors="coerce")
    has_k = df["milleage"].str.contains("[Kk]", na=False)
    df["milleage"] = mile_num.where(~has_k, mile_num * 1000)

    # Drop invalid rows
    df = df.dropna(subset=["list_price", "milleage", "year"])
    df = df[(df["list_price"] > 0) & (df["milleage"] >= 0) & (df["year"] >= 1990)]

    # Fill missing text fields
    for col in ["model", "gear_type", "location", "description"]:
        if col in df.columns:
            df[col] = df[col].fillna("Not available").replace("nan", "Not available")

    return df

# -------------------------------
# Train Model
# -------------------------------
@st.cache_resource
def train_model(df):
    le_model = LabelEncoder()
    le_gear = LabelEncoder()

    df_model = df.copy()
    df_model["model_encoded"] = le_model.fit_transform(df_model["model"].astype(str))
    df_model["gear_encoded"] = le_gear.fit_transform(df_model["gear_type"].astype(str))

    X = df_model[["year", "milleage", "car_age", "model_encoded", "gear_encoded"]]
    y = df_model["list_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    return model, le_model, le_gear, score

# -------------------------------
# Main App
# -------------------------------
df = load_data()

if df is not None and len(df) > 0:
    model, le_model, le_gear, score = train_model(df)

    if model is not None:
        st.sidebar.success(f"Model trained! R² Score: {score:.3f}")

        page = st.sidebar.radio("Navigation", ["Overview", "Price Prediction", "Recommendations"])

        # -------------------------------
        # Overview Page
        # -------------------------------
        if page == "Overview":
            st.header("Market Overview")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Cars", f"{len(df):,}")
            col2.metric("Avg Price", f"RM {df['list_price'].mean():,.0f}")
            col3.metric("Median Price", f"RM {df['list_price'].median():,.0f}")
            col4.metric("Avg Mileage", f"{df['milleage'].mean():,.0f} km")

            st.subheader("Top 10 Models")
            model_counts = df["model"].value_counts().head(10)
            st.bar_chart(model_counts)

            st.subheader("Price Distribution")
            import matplotlib.pyplot as plt
            st.subheader("Price Distribution")
            
            fig, ax = plt.subplots()
            ax.hist(df["list_price"], bins=30, color="skyblue", edgecolor="black")
            ax.set_xlabel("Price (RM)")
            ax.set_ylabel("Number of Cars")
            ax.set_title("Distribution of Used Car Prices")

            st.pyplot(fig)

        # -------------------------------
        # Price Prediction Page
        # -------------------------------
        elif page == "Price Prediction":
            st.header("Predict Car Price")
           
            col1, col2 = st.columns(2)
            
            with col1:
                selected_model = st.selectbox("Model", sorted(df["model"].unique()))
                year = st.number_input("Year", 1990, 2026, 2020)
                mileage = st.number_input("Mileage (km)", 0, 500000, 50000, step=5000)
            
            with col2:
                gear_type = st.selectbox("Gear Type", sorted(df["gear_type"].unique()))
            
            if st.button("Predict Price", type="primary"):
                try:
                    car_age = 2026 - year
                            
                    # Safe encoding
                    def safe_encode(encoder, value):
                        if value in encoder.classes_:
                            return encoder.transform([value])[0]
                        else:
                            encoder.classes_ = np.append(encoder.classes_, value)
                            return encoder.transform([value])[0]  
                    
                    model_encoded = safe_encode(le_model, selected_model)
                    gear_encoded = safe_encode(le_gear, gear_type)
                    
                    input_data = pd.DataFrame({
                        "year": [year],
                        "milleage": [mileage],
                        "car_age": [car_age],
                        "model_encoded": [model_encoded],
                        "gear_encoded": [gear_encoded],
                    })
                    
                    predicted_price = model.predict(input_data)[0]
                    
                    st.success("Prediction Complete!")
                    st.metric("Predicted Price", f"RM {predicted_price:,.0f}")
                
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

        # -------------------------------
        # PAGE 3: Recommendations
        # -------------------------------
        elif page == "Recommendations":
            st.header("Find Your Perfect Car")
            
            col1, col2 = st.columns(2)
            with col1:
                budget_min = st.number_input("Min Budget (RM)", 0, 500000, 30000, 5000)
                budget_max = st.number_input("Max Budget (RM)", 0, 1000000, 100000, 5000)
                max_mileage = st.number_input("Max Mileage (km)", 0, 500000, 100000, 10000)
            with col2:
                preferred_models = st.multiselect("Preferred Models (optional)", sorted(df["model"].unique()))
                min_year = st.number_input("Min Year", 1990, 2026, 2015)
                
            if st.button("Find Cars", type="primary"):
                filtered = df[
                    (df["list_price"] >= budget_min) &
                    (df["list_price"] <= budget_max) &
                    (df["milleage"] <= max_mileage) &
                    (df["year"] >= min_year)
                ].copy()

                if preferred_models:
                    filtered = filtered[filtered["model"].isin(preferred_models)]

                if len(filtered) == 0:
                    st.warning("No cars found. Try adjusting your filters.")
                else:
                    # Match scoring
                    filtered["price_score"] = 1 - (filtered["list_price"] - budget_min) / max(1, (budget_max - budget_min))
                    filtered["mileage_score"] = 1 - (filtered["milleage"] / max_mileage)
                    filtered["age_score"] = (filtered["year"] - min_year) / max(1, (2024 - min_year))
                    filtered["match_score"] = (
                        filtered["price_score"] * 0.3 +
                        filtered["mileage_score"] * 0.4 +
                        filtered["age_score"] * 0.3
                    ) * 100

                    top_cars = filtered.nlargest(10, "match_score")

                    st.success(f"Found {len(filtered)} cars. Showing top 10 recommendations:")

                    for idx, car in top_cars.iterrows():
                        with st.expander(f"**{car['description']}** - RM {car['list_price']:,.0f} | Match: {car['match_score']:.0f}%"):
                            col1, col2, col3 = st.columns(3)
                            col1.write(f"**Location:** {car['location']}")
                            col2.write(f"**Mileage:** {car['milleage']:,.0f} km")
                            col3.write(f"**Gear:** {car['gear_type']}")
                            col1.write(f"**Year:** {int(car['year'])}")
                            col2.write(f"**Model:** {car['model']}")
