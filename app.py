# ============================================================
# EcoSense AI
# Intelligent Insights for Sustainable Resource Consumption
# 1M1B AI for Sustainability Virtual Internship
# ============================================================

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EcoSense AI",
    page_icon="🌱",
    layout="wide"
)

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "campus_energy_data.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "prediction_model.pkl")
FEATURE_FILE = os.path.join(MODEL_DIR, "features.pkl")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f8f6;
}

.block-container {
    padding-top: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #0f766e,
        #166534
    );
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}

.small-text {
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. DATASET GENERATION
# ============================================================

@st.cache_data
def generate_dataset():

    np.random.seed(42)

    num_days = 180

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=num_days * 24,
        freq="h"
    )

    buildings = [
        "Block A",
        "Block B",
        "Block C",
        "Block D",
        "Block E"
    ]

    rows = []

    for timestamp in timestamps:

        hour = timestamp.hour
        weekday = timestamp.weekday()

        weekend = 1 if weekday >= 5 else 0

        working_hours = (
            1
            if 8 <= hour <= 18 and weekend == 0
            else 0
        )

        holiday = (
            1
            if timestamp.day in [1, 15]
            else 0
        )

        for building in buildings:

            base_occupancy = {
                "Block A": 300,
                "Block B": 450,
                "Block C": 250,
                "Block D": 350,
                "Block E": 200
            }[building]

            if working_hours:

                occupancy = (
                    base_occupancy
                    + np.random.normal(0, 30)
                )

            else:

                occupancy = (
                    base_occupancy * 0.10
                    + np.random.normal(0, 10)
                )

            occupancy = max(0, occupancy)

            # Temperature
            temperature = (
                28
                + 5 * np.sin(
                    (hour - 6) / 24 * 2 * np.pi
                )
                + np.random.normal(0, 1.5)
            )

            humidity = (
                65
                + np.random.normal(0, 5)
            )

            # HVAC
            hvac = (
                20
                + occupancy * 0.025
                + max(temperature - 25, 0) * 8
                + np.random.normal(0, 5)
            )

            if not working_hours:
                hvac *= 0.35

            hvac = max(0, hvac)

            # Lighting
            lighting = (
                10
                + occupancy * 0.015
                + np.random.normal(0, 2)
            )

            if not working_hours:
                lighting *= 0.20

            lighting = max(0, lighting)

            # Equipment
            equipment = (
                25
                + occupancy * 0.02
                + np.random.normal(0, 5)
            )

            # Solar
            if 7 <= hour <= 17:

                solar = max(
                    0,
                    80
                    * np.sin(
                        (hour - 7) / 10 * np.pi
                    )
                    + np.random.normal(0, 5)
                )

            else:

                solar = 0

            # Electricity
            electricity = (
                hvac
                + lighting
                + equipment
                - solar * 0.25
                + np.random.normal(0, 10)
            )

            electricity = max(
                10,
                electricity
            )

            rows.append([
                timestamp,
                building,
                electricity,
                hvac,
                lighting,
                occupancy,
                temperature,
                humidity,
                working_hours,
                weekend,
                holiday,
                solar
            ])

    columns = [
        "timestamp",
        "building",
        "electricity_kwh",
        "hvac_kwh",
        "lighting_kwh",
        "occupancy",
        "temperature",
        "humidity",
        "working_hours",
        "weekend",
        "holiday",
        "solar_generation_kwh"
    ]

    df = pd.DataFrame(
        rows,
        columns=columns
    )

    # ========================================================
    # SYNTHETIC ANOMALIES
    # ========================================================

    condition = (
        (df["building"] == "Block B")
        &
        (df["working_hours"] == 0)
        &
        (df["timestamp"].dt.hour.isin(
            [20, 21, 22]
        ))
    )

    df.loc[
        condition,
        "hvac_kwh"
    ] *= 2.5

    df.loc[
        condition,
        "lighting_kwh"
    ] *= 2.2

    df.loc[
        condition,
        "electricity_kwh"
    ] = (
        df.loc[
            condition,
            "hvac_kwh"
        ]
        +
        df.loc[
            condition,
            "lighting_kwh"
        ]
        + 25
    )

    df.to_csv(
        DATA_FILE,
        index=False
    )

    return df


# ============================================================
# 2. PREPROCESSING
# ============================================================

@st.cache_data
def preprocess_data(df):

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data["hour"] = (
        data["timestamp"].dt.hour
    )

    data["day"] = (
        data["timestamp"].dt.day
    )

    data["month"] = (
        data["timestamp"].dt.month
    )

    data["weekday"] = (
        data["timestamp"].dt.weekday
    )

    # Building encoding
    building_mapping = {
        "Block A": 0,
        "Block B": 1,
        "Block C": 2,
        "Block D": 3,
        "Block E": 4
    }

    data["building_encoded"] = (
        data["building"].map(
            building_mapping
        )
    )

    # Lag features
    data = data.sort_values(
        ["building", "timestamp"]
    )

    data["previous_hour_consumption"] = (
        data.groupby("building")
        ["electricity_kwh"]
        .shift(1)
    )

    data["previous_day_consumption"] = (
        data.groupby("building")
        ["electricity_kwh"]
        .shift(24)
    )

    data["rolling_mean"] = (
        data.groupby("building")
        ["electricity_kwh"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(24)
            .mean()
        )
    )

    data = data.dropna()

    return data


# ============================================================
# 3. TRAIN PREDICTION MODEL
# ============================================================

@st.cache_resource
def train_prediction_model(data):

    features = [
        "building_encoded",
        "hour",
        "weekday",
        "weekend",
        "holiday",
        "occupancy",
        "temperature",
        "humidity",
        "hvac_kwh",
        "lighting_kwh",
        "solar_generation_kwh",
        "previous_hour_consumption",
        "previous_day_consumption",
        "rolling_mean"
    ]

    X = data[features]

    y = data["electricity_kwh"]

    # Chronological split
    split = int(
        len(data) * 0.8
    )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # XGBoost
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Generate predictions for entire dataset
    data = data.copy()

    data["expected_electricity_kwh"] = (
        model.predict(
            data[features]
        )
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        features,
        FEATURE_FILE
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return model, data, metrics, features


# ============================================================
# 4. ANOMALY DETECTION
# ============================================================

@st.cache_data
def detect_anomalies(data):

    result = data.copy()

    result["difference_kwh"] = (
        result["electricity_kwh"]
        -
        result["expected_electricity_kwh"]
    )

    result["deviation_percent"] = (
        result["difference_kwh"]
        /
        result["expected_electricity_kwh"]
        * 100
    )

    # Residual
    residuals = (
        result["difference_kwh"]
    )

    isolation = IsolationForest(
        contamination=0.03,
        random_state=42
    )

    isolation_result = isolation.fit_predict(
        residuals.values.reshape(-1, 1)
    )

    result["isolation_anomaly"] = (
        isolation_result == -1
    )

    # Main anomaly rule
    result["is_anomaly"] = (
        (
            result["deviation_percent"]
            > 20
        )
        &
        result["isolation_anomaly"]
    )

    result["severity"] = "Normal"

    result.loc[
        result["deviation_percent"] > 20,
        "severity"
    ] = "Warning"

    result.loc[
        result["deviation_percent"] > 40,
        "severity"
    ] = "Critical"

    return result


# ============================================================
# 5. ROOT CAUSE / CONTRIBUTING FACTORS
# ============================================================

def analyze_root_cause(row):

    factors = []

    if row["hvac_kwh"] > 60:
        factors.append(
            "Elevated HVAC load"
        )

    if row["lighting_kwh"] > 25:
        factors.append(
            "Elevated lighting load"
        )

    if row["temperature"] > 30:
        factors.append(
            "High temperature"
        )

    if row["occupancy"] < 50:
        factors.append(
            "Low occupancy despite high consumption"
        )

    if row["working_hours"] == 0:
        factors.append(
            "Consumption during non-working hours"
        )

    if not factors:

        factors.append(
            "No dominant contributing factor identified"
        )

    return factors


# ============================================================
# 6. DIGITAL TWIN
# ============================================================

def simulate_scenario(
    baseline,
    lighting_reduction,
    hvac_reduction,
    nonworking_reduction,
    solar_increase
):

    lighting_effect = (
        baseline
        * 0.20
        * (lighting_reduction / 100)
    )

    hvac_effect = (
        baseline
        * 0.45
        * (hvac_reduction / 100)
    )

    nonworking_effect = (
        baseline
        * 0.15
        * (nonworking_reduction / 100)
    )

    solar_effect = (
        baseline
        * 0.10
        * (solar_increase / 100)
    )

    total_saving = (
        lighting_effect
        + hvac_effect
        + nonworking_effect
        + solar_effect
    )

    simulated = max(
        baseline - total_saving,
        0
    )

    energy_saved = (
        baseline - simulated
    )

    percentage_saved = (
        energy_saved
        / baseline
        * 100
        if baseline > 0
        else 0
    )

    electricity_price = 8

    carbon_factor = 0.7

    cost_saved = (
        energy_saved
        * electricity_price
    )

    co2_saved = (
        energy_saved
        * carbon_factor
    )

    return {
        "baseline": baseline,
        "simulated": simulated,
        "energy_saved": energy_saved,
        "percentage_saved": percentage_saved,
        "cost_saved": cost_saved,
        "co2_saved": co2_saved
    }


# ============================================================
# 7. OPTIMIZATION ENGINE
# ============================================================

def optimize_strategy(
    baseline
):

    best = None

    lighting_values = [
        0,
        5,
        10,
        15,
        20
    ]

    hvac_values = [
        0,
        5,
        10,
        15
    ]

    nonworking_values = [
        0,
        5,
        10,
        15
    ]

    for lighting in lighting_values:

        for hvac in hvac_values:

            for nonworking in nonworking_values:

                result = simulate_scenario(
                    baseline,
                    lighting,
                    hvac,
                    nonworking,
                    0
                )

                # Constraint:
                # avoid excessive intervention

                if (
                    lighting <= 20
                    and hvac <= 15
                    and nonworking <= 15
                ):

                    if (
                        best is None
                        or
                        result["energy_saved"]
                        > best["result"]["energy_saved"]
                    ):

                        best = {
                            "lighting": lighting,
                            "hvac": hvac,
                            "nonworking": nonworking,
                            "result": result
                        }

    return best


# ============================================================
# 8. LLM-STYLE SUSTAINABILITY ADVISOR
# ============================================================

def generate_advice(
    anomaly_row=None,
    simulation=None,
    optimization=None
):

    response = ""

    if anomaly_row is not None:

        building = anomaly_row[
            "building"
        ]

        deviation = anomaly_row[
            "deviation_percent"
        ]

        working = anomaly_row[
            "working_hours"
        ]

        factors = analyze_root_cause(
            anomaly_row
        )

        response += (
            f"### Detected Issue\n"
            f"{building} is consuming "
            f"approximately "
            f"{deviation:.1f}% more electricity "
            f"than its expected baseline.\n\n"
        )

        if working == 0:

            response += (
                "### Observed Pattern\n"
                "The anomaly occurred during "
                "non-working hours.\n\n"
            )

        response += (
            "### Associated Factors\n"
        )

        for factor in factors:

            response += (
                f"- {factor}\n"
            )

        response += (
            "\n### Recommended Investigation\n"
            "Check HVAC schedules, lighting "
            "controls and equipment operation "
            "during the identified period.\n\n"
        )

    if simulation is not None:

        response += (
            "### Simulation Result\n"
            f"Energy saved: "
            f"{simulation['energy_saved']:.2f} kWh\n\n"
            f"Reduction: "
            f"{simulation['percentage_saved']:.2f}%\n\n"
            f"Estimated cost saving: "
            f"₹{simulation['cost_saved']:.2f}\n\n"
            f"Estimated CO₂ reduction: "
            f"{simulation['co2_saved']:.2f} kg\n\n"
        )

    if optimization is not None:

        response += (
            "### Suggested Strategy\n"
            f"Lighting reduction: "
            f"{optimization['lighting']}%\n\n"
            f"HVAC reduction: "
            f"{optimization['hvac']}%\n\n"
            f"Non-working-hour reduction: "
            f"{optimization['nonworking']}%\n\n"
            f"Estimated energy saving: "
            f"{optimization['result']['energy_saved']:.2f} kWh\n"
        )

    return response


# ============================================================
# LOAD / GENERATE DATA
# ============================================================

df = generate_dataset()

processed_df = preprocess_data(
    df
)

model, prediction_df, metrics, features = (
    train_prediction_model(
        processed_df
    )
)

analysis_df = detect_anomalies(
    prediction_df
)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🌱 EcoSense AI</h1>

<p>
Intelligent Insights for Sustainable Resource Consumption
</p>

<p>
AI-powered campus energy prediction, anomaly detection,
Digital Twin simulation and sustainability decision support.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "EcoSense AI"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Sustainability Dashboard",
        "🏢 Building Analysis",
        "🚨 Anomaly Detection",
        "🔬 Digital Twin",
        "⚙️ Optimization",
        "🤖 AI Sustainability Advisor",
        "📈 Model Performance"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header(
        "EcoSense AI"
    )

    st.write(
        """
        EcoSense AI is an AI-powered sustainability
        decision-support system designed for university
        campuses.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    total_energy = (
        analysis_df[
            "electricity_kwh"
        ].sum()
    )

    expected_energy = (
        analysis_df[
            "expected_electricity_kwh"
        ].sum()
    )

    anomalies = (
        analysis_df[
            "is_anomaly"
        ].sum()
    )

    deviation = (
        (
            total_energy
            -
            expected_energy
        )
        /
        expected_energy
        * 100
    )

    col1.metric(
        "Total Energy",
        f"{total_energy:,.0f} kWh"
    )

    col2.metric(
        "Expected Energy",
        f"{expected_energy:,.0f} kWh"
    )

    col3.metric(
        "Anomalies",
        int(anomalies)
    )

    col4.metric(
        "Deviation",
        f"{deviation:.1f}%"
    )

    st.markdown(
        "### AI Decision-Support Pipeline"
    )

    st.code(
        """
Campus Data
     ↓
Data Processing
     ↓
AI Prediction
     ↓
Anomaly Detection
     ↓
Contributing-Factor Analysis
     ↓
Digital Twin
     ↓
Optimization Engine
     ↓
AI Sustainability Advisor
        """,
        language="text"
    )

    st.info(
        "This prototype uses synthetic campus "
        "data for demonstration purposes."
    )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Sustainability Dashboard":

    st.header(
        "📊 Sustainability Dashboard"
    )

    col1, col2, col3, col4 = st.columns(4)

    total = analysis_df[
        "electricity_kwh"
    ].sum()

    expected = analysis_df[
        "expected_electricity_kwh"
    ].sum()

    savings = max(
        expected - total,
        0
    )

    co2 = total * 0.7

    col1.metric(
        "Electricity",
        f"{total:,.0f} kWh"
    )

    col2.metric(
        "Expected",
        f"{expected:,.0f} kWh"
    )

    col3.metric(
        "Potential Saving",
        f"{savings:,.0f} kWh"
    )

    col4.metric(
        "Estimated CO₂",
        f"{co2:,.0f} kg"
    )

    st.subheader(
        "Actual vs Expected Consumption"
    )

    daily = (
        analysis_df
        .set_index("timestamp")
        .resample("D")
        [
            [
                "electricity_kwh",
                "expected_electricity_kwh"
            ]
        ]
        .sum()
        .reset_index()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily["timestamp"],
            y=daily["electricity_kwh"],
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily["timestamp"],
            y=daily[
                "expected_electricity_kwh"
            ],
            name="Expected"
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    building_energy = (
        analysis_df
        .groupby("building")
        ["electricity_kwh"]
        .sum()
        .reset_index()
    )

    fig2 = px.bar(
        building_energy,
        x="building",
        y="electricity_kwh",
        title="Building-wise Energy Consumption"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# BUILDING ANALYSIS
# ============================================================

elif page == "🏢 Building Analysis":

    st.header(
        "🏢 Building Analysis"
    )

    building = st.selectbox(
        "Select Building",
        sorted(
            analysis_df[
                "building"
            ].unique()
        )
    )

    building_df = analysis_df[
        analysis_df[
            "building"
        ] == building
    ]

    actual = building_df[
        "electricity_kwh"
    ].sum()

    expected = building_df[
        "expected_electricity_kwh"
    ].sum()

    deviation = (
        (
            actual - expected
        )
        /
        expected
        * 100
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Actual",
        f"{actual:,.0f} kWh"
    )

    col2.metric(
        "Expected",
        f"{expected:,.0f} kWh"
    )

    col3.metric(
        "Deviation",
        f"{deviation:.2f}%"
    )

    st.subheader(
        f"{building} Consumption"
    )

    hourly = (
        building_df
        .set_index("timestamp")
        .resample("D")
        [
            [
                "electricity_kwh",
                "expected_electricity_kwh"
            ]
        ]
        .sum()
        .reset_index()
    )

    fig = px.line(
        hourly,
        x="timestamp",
        y=[
            "electricity_kwh",
            "expected_electricity_kwh"
        ],
        title=f"{building}: Actual vs Expected"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Consumption Factors"
    )

    factor_df = pd.DataFrame({
        "Factor": [
            "HVAC",
            "Lighting",
            "Occupancy"
        ],
        "Average": [
            building_df[
                "hvac_kwh"
            ].mean(),
            building_df[
                "lighting_kwh"
            ].mean(),
            building_df[
                "occupancy"
            ].mean()
        ]
    })

    fig2 = px.bar(
        factor_df,
        x="Factor",
        y="Average"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "🚨 Anomaly Detection":

    st.header(
        "🚨 Context-Aware Anomaly Detection"
    )

    anomalies = analysis_df[
        analysis_df[
            "is_anomaly"
        ]
    ].copy()

    st.metric(
        "Detected Anomalies",
        len(anomalies)
    )

    display_columns = [
        "timestamp",
        "building",
        "electricity_kwh",
        "expected_electricity_kwh",
        "deviation_percent",
        "severity"
    ]

    st.dataframe(
        anomalies[
            display_columns
        ].sort_values(
            "deviation_percent",
            ascending=False
        ).head(100),
        use_container_width=True
    )

    st.subheader(
        "Actual vs Expected"
    )

    sample = analysis_df[
        analysis_df["building"] == "Block B"
    ].tail(500)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sample["timestamp"],
            y=sample[
                "electricity_kwh"
            ],
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sample["timestamp"],
            y=sample[
                "expected_electricity_kwh"
            ],
            name="Expected"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if len(anomalies) > 0:

        selected_index = st.selectbox(
            "Select an anomaly",
            anomalies.index
        )

        selected = anomalies.loc[
            selected_index
        ]

        st.subheader(
            "Anomaly Explanation"
        )

        st.write(
            f"**Building:** "
            f"{selected['building']}"
        )

        st.write(
            f"**Actual:** "
            f"{selected['electricity_kwh']:.2f} kWh"
        )

        st.write(
            f"**Expected:** "
            f"{selected['expected_electricity_kwh']:.2f} kWh"
        )

        st.write(
            f"**Deviation:** "
            f"{selected['deviation_percent']:.2f}%"
        )

        factors = analyze_root_cause(
            selected
        )

        st.write(
            "**Possible contributing factors:**"
        )

        for factor in factors:

            st.write(
                f"• {factor}"
            )


# ============================================================
# DIGITAL TWIN
# ============================================================

elif page == "🔬 Digital Twin":

    st.header(
        "🔬 Digital Twin / What-If Simulation"
    )

    building = st.selectbox(
        "Select Building",
        sorted(
            analysis_df[
                "building"
            ].unique()
        )
    )

    building_df = analysis_df[
        analysis_df[
            "building"
        ] == building
    ]

    baseline = building_df[
        "electricity_kwh"
    ].mean()

    st.info(
        f"Current average baseline for "
        f"{building}: "
        f"{baseline:.2f} kWh"
    )

    lighting = st.slider(
        "Lighting Reduction (%)",
        0,
        30,
        15
    )

    hvac = st.slider(
        "HVAC Reduction (%)",
        0,
        30,
        10
    )

    nonworking = st.slider(
        "Non-working Hour Reduction (%)",
        0,
        30,
        10
    )

    solar = st.slider(
        "Solar Generation Increase (%)",
        0,
        50,
        0
    )

    if st.button(
        "🔬 Simulate Scenario"
    ):

        result = simulate_scenario(
            baseline,
            lighting,
            hvac,
            nonworking,
            solar
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Baseline",
            f"{result['baseline']:.2f} kWh"
        )

        col2.metric(
            "Simulated",
            f"{result['simulated']:.2f} kWh"
        )

        col3.metric(
            "Energy Saved",
            f"{result['energy_saved']:.2f} kWh"
        )

        col4.metric(
            "Reduction",
            f"{result['percentage_saved']:.2f}%"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Estimated Cost Saving",
            f"₹{result['cost_saved']:.2f}"
        )

        col2.metric(
            "Estimated CO₂ Reduction",
            f"{result['co2_saved']:.2f} kg"
        )

        comparison = pd.DataFrame({
            "Scenario": [
                "Baseline",
                "Simulation"
            ],
            "Energy": [
                result["baseline"],
                result["simulated"]
            ]
        })

        fig = px.bar(
            comparison,
            x="Scenario",
            y="Energy",
            title="Baseline vs Simulated Consumption"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# OPTIMIZATION
# ============================================================

elif page == "⚙️ Optimization":

    st.header(
        "⚙️ Sustainability Optimization Engine"
    )

    building = st.selectbox(
        "Select Building",
        sorted(
            analysis_df[
                "building"
            ].unique()
        )
    )

    building_df = analysis_df[
        analysis_df[
            "building"
        ] == building
    ]

    baseline = building_df[
        "electricity_kwh"
    ].mean()

    st.write(
        f"Baseline consumption: "
        f"**{baseline:.2f} kWh**"
    )

    if st.button(
        "⚙️ Find Sustainable Strategy"
    ):

        with st.spinner(
            "Evaluating sustainability scenarios..."
        ):

            best = optimize_strategy(
                baseline
            )

        st.success(
            "Optimization completed."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Lighting Reduction",
            f"{best['lighting']}%"
        )

        col2.metric(
            "HVAC Reduction",
            f"{best['hvac']}%"
        )

        col3.metric(
            "Non-working Reduction",
            f"{best['nonworking']}%"
        )

        result = best["result"]

        st.subheader(
            "Estimated Impact"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Energy Saved",
            f"{result['energy_saved']:.2f} kWh"
        )

        col2.metric(
            "Cost Saving",
            f"₹{result['cost_saved']:.2f}"
        )

        col3.metric(
            "CO₂ Reduction",
            f"{result['co2_saved']:.2f} kg"
        )

        st.info(
            "This recommendation is generated by "
            "evaluating feasible sustainability "
            "scenarios using the Digital Twin."
        )


# ============================================================
# AI SUSTAINABILITY ADVISOR
# ============================================================

elif page == "🤖 AI Sustainability Advisor":

    st.header(
        "🤖 AI Sustainability Advisor"
    )

    st.write(
        """
        Ask questions about campus energy consumption,
        anomalies and sustainability scenarios.
        """
    )

    question = st.text_input(
        "Ask EcoSense AI"
    )

    if st.button(
        "Ask Advisor"
    ):

        if question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            question_lower = (
                question.lower()
            )

            # Find relevant anomaly
            anomaly_data = analysis_df[
                analysis_df[
                    "is_anomaly"
                ]
            ]

            selected_anomaly = None

            if len(anomaly_data) > 0:

                selected_anomaly = (
                    anomaly_data
                    .sort_values(
                        "deviation_percent",
                        ascending=False
                    )
                    .iloc[0]
                )

            simulation = None
            optimization = None

            # Detect lighting question
            if (
                "lighting" in question_lower
                or
                "15%" in question_lower
            ):

                if selected_anomaly is not None:

                    baseline = (
                        selected_anomaly[
                            "electricity_kwh"
                        ]
                    )

                    simulation = simulate_scenario(
                        baseline,
                        15,
                        0,
                        0,
                        0
                    )

            # Optimization question
            if (
                "optimiz" in question_lower
                or
                "recommend" in question_lower
                or
                "strategy" in question_lower
            ):

                if selected_anomaly is not None:

                    optimization = (
                        optimize_strategy(
                            selected_anomaly[
                                "electricity_kwh"
                            ]
                        )
                    )

            answer = generate_advice(
                selected_anomaly,
                simulation,
                optimization
            )

            if answer == "":

                answer = """
### EcoSense AI Advisor

I can analyze campus sustainability information.

Try asking:

- Why is Block B consuming more electricity?
- What happens if lighting is reduced by 15%?
- Recommend a sustainable strategy.
- Explain the current anomaly.
"""

            st.markdown(
                answer
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.header(
        "📈 AI Prediction Model Performance"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{metrics['MAE']:.2f}"
    )

    col2.metric(
        "RMSE",
        f"{metrics['RMSE']:.2f}"
    )

    col3.metric(
        "R²",
        f"{metrics['R2']:.4f}"
    )

    st.subheader(
        "Actual vs Predicted"
    )

    test_sample = prediction_df.tail(
        500
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=test_sample[
                "electricity_kwh"
            ].values,
            name="Actual"
        )
    )

    fig.add_trace(
        go.Scatter(
            y=test_sample[
                "expected_electricity_kwh"
            ].values,
            name="Predicted"
        )
    )

    fig.update_layout(
        xaxis_title="Sample",
        yaxis_title="Electricity (kWh)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Model Features"
    )

    st.write(
        features
    )

    st.success(
        "The prediction model is trained using "
        "historical and contextual campus data."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "EcoSense AI | 1M1B AI for Sustainability "
    "Virtual Internship | IBM SkillsBuild & AICTE"
)
