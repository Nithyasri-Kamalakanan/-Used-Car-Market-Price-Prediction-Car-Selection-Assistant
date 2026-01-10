import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
import warnings

warnings.filterwarnings("ignore")

# optional: try to import LightGBM, otherwise fallback to sklearn only
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

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
                         encoding='utf-8'
                        )

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # Extract year from description
    df["year"] = df["description"].str.extract(r"(\d{4})")[0]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Extract brand from model
    df["brand"] = df["model"].str.split().str[0]

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
 
    # -------------------------------
    # CLEAN INVALID MODEL NAMES
    # -------------------------------
    df["model"] = df["model"].astype(str)
                    
    price_pattern = r"(rm\s?\d+|\d+\s*/\s*month|\d+\s*month)"
                    
    df = df[~df["model"].str.lower().str.contains(price_pattern, regex=True)]
    df = df[df["model"].str.contains("[A-Za-z]", regex=True)]
                    
    return df


def smooth_target_encoding(series, target, m=5):
    # series: pd.Series of categorical values
    # target: pd.Series numeric target (not log-transformed)
    agg = target.groupby(series).agg(["mean", "count"]).rename(columns={"mean": "mean", "count": "count"})
    global_mean = target.mean()
    smooth = (agg["count"] * agg["mean"] + m * global_mean) / (agg["count"] + m)
    return series.map(smooth).fillna(global_mean)

# -------------------------------
# Train Model
# -------------------------------
@st.cache_resource
def train_model(df, do_cv=True, retrain: bool = False):
    # Prepare features with smoothing encodings and label encoder for brand
    df_model = df.copy()

    # smooth encodings based on original price (not log) to preserve scale
    df_model["model_encoded"] = smooth_target_encoding(df_model["model"], df_model["list_price"], m=8)
    df_model["gear_encoded"] = smooth_target_encoding(df_model["gear_type"], df_model["list_price"], m=8)

    le_brand = LabelEncoder()
    df_model["brand_encoded"] = le_brand.fit_transform(df_model["brand"].astype(str))

    # target: log1p transform to reduce skew
    y = np.log1p(df_model["list_price"])

    X = df_model[["year", "milleage", "car_age", "brand_encoded", "model_encoded", "gear_encoded"]]

    # small train-test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Candidate models: RandomForest and LightGBM (if available)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    models = {}
    models["rf"] = rf
    if HAS_LGB:
        lgbm = lgb.LGBMRegressor(random_state=42)
        models["lgbm"] = lgbm

    best_models = {}

    # Quick randomized search spaces
    rf_params = {
        "n_estimators": [100, 200, 400],
        "max_depth": [10, 20, 30, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }

    lgb_params = {
        "n_estimators": [100, 300, 600],
        "num_leaves": [31, 50, 100],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 10, 20]
    }

    # Fit and tune each candidate with small RandomizedSearchCV
    for name, mdl in models.items():
        if name == "rf":
            rs = RandomizedSearchCV(mdl, rf_params, n_iter=2, cv=2, scoring="r2", n_jobs=1, random_state=42)
        else:
            rs = RandomizedSearchCV(mdl, lgb_params, n_iter=2, cv=2, scoring="r2", n_jobs=1, random_state=42)

        rs.fit(X_train, y_train)
        best_models[name] = rs.best_estimator_

    # Evaluate with 5-fold CV on training set for each best model
    cv_scores = {}
    for name, mdl in best_models.items():
        scores = cross_val_score(mdl, X_train, y_train, cv=3, scoring="r2", n_jobs=1)
        cv_scores[name] = np.mean(scores)

    # pick best by CV score
    # use items() to avoid typing issues with key= on some linters
    best_name = max(cv_scores.items(), key=lambda kv: kv[1])[0]
    best_model = best_models[best_name]

    # fit best_model on full train set
    best_model.fit(X_train, y_train)

    # Evaluate on holdout test (log-target R2)
    test_r2_log = best_model.score(X_test, y_test)

    # return a dict of helpers and model
    artifacts = {
        "model": best_model,
        "le_brand": le_brand,
        "model_encoding_map": df_model.set_index("model")["model_encoded"].to_dict(),
        "gear_encoding_map": df_model.set_index("gear_type")["gear_encoded"].to_dict(),
        "cv_scores": cv_scores,
        "chosen": best_name,
        "test_r2_log": test_r2_log
    }

    return artifacts

# -------------------------------
# Main App
# -------------------------------
df = load_data()

if df is not None and len(df) > 0:
    # allow manual retrain: clear cached resource and retrain only when user requests it
    if st.sidebar.button("Retrain model (slow)"):
        st.sidebar.info("Clearing cache and retraining — this may take some time...")
        try:
            st.cache_resource.clear()
        except Exception:
            # older Streamlit versions may not have clear(); ignore
            pass
        artifacts = train_model(df, retrain=True)
    else:
        artifacts = train_model(df)

    model = artifacts["model"]
    le_brand = artifacts["le_brand"]
    model_price_mean = artifacts["model_encoding_map"]
    gear_price_mean = artifacts["gear_encoding_map"]
    score = artifacts.get("test_r2_log", None)

    if model is not None:
        chosen = artifacts.get("chosen", "random forest")
        cv_scores = artifacts.get("cv_scores", {})
        st.sidebar.success(f"Model trained! | {chosen.upper()} | test R² (log-target): {score:.3f}")
        if cv_scores:
            st.sidebar.write("CV R² (training):")
            for k, v in cv_scores.items():
                st.sidebar.write(f" - {k}: {v:.3f}")

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
                brand = str(selected_model).split()[0] if selected_model else "Unknown"
            
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
                    
                    brand_encoded = safe_encode(le_brand, brand)

                    def _dict_mean(d):
                        try:
                            return np.mean(list(d.values()))
                        except Exception:
                            return 0.0

                    model_encoded = model_price_mean.get(selected_model, _dict_mean(model_price_mean))
                    gear_encoded = gear_price_mean.get(gear_type, _dict_mean(gear_price_mean))
                    
                    input_data = pd.DataFrame({
                        "year": [year],
                        "milleage": [mileage],
                        "car_age": [car_age],
                        "brand_encoded": [brand_encoded],
                        "model_encoded": [model_encoded],
                        "gear_encoded": [gear_encoded],
                    })
                    
                    # model predicts log1p(target), invert with expm1
                    predicted_log = model.predict(input_data)[0]
                    predicted_price = np.expm1(predicted_log)

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
                    st.success(f"Found {len(filtered)} cars. Showing top 10 recommendations:")
                    
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

                    for idx, car in top_cars.iterrows():
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{car['model']}** ({int(car['year'])})")
                                st.write(f"Mileage: {car['milleage']:,.0f} km | Gear: {car['gear_type']}")
                            with col2:
                                st.metric("Price", f"RM {car['list_price']:,.0f}")
                            with col3:
                                st.metric("Match Score", f"{car['match_score']:.1f}%")
                            st.divider()
