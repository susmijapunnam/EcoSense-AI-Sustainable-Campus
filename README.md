# 🌱 EcoSense AI: Intelligent Insights for Sustainable Resource Consumption

> **AI-powered campus sustainability monitoring, prediction, anomaly detection, simulation, and optimization platform.**

EcoSense AI is an intelligent sustainability platform designed to help educational institutions monitor and optimize electricity and resource consumption.

The system combines **Machine Learning, anomaly detection, contributing-factor analysis, Digital Twin simulation, optimization, and an AI Sustainability Advisor** into an interactive web application.

The project is developed as part of the **1M1B AI for Sustainability Virtual Internship**, in collaboration with **IBM SkillsBuild and AICTE**.

---

## 🚀 Project Overview

Traditional energy dashboards mainly display historical consumption data. EcoSense AI goes beyond visualization by providing intelligent insights and decision-support capabilities.

The system can:

* 📊 Monitor campus electricity consumption
* 🤖 Predict expected energy consumption
* 🚨 Detect unusual consumption patterns
* 🔍 Identify contributing factors behind anomalies
* 🏢 Analyze individual campus buildings
* 🧪 Simulate "What-If" sustainability scenarios
* ⚡ Optimize energy-saving strategies
* 🌍 Estimate CO₂ emission reductions
* 💬 Provide sustainability recommendations through an AI advisor

### Example

If the system detects that:

> **Block B is consuming significantly more electricity than expected during non-working hours**

EcoSense AI can investigate contributing factors such as:

* HVAC consumption
* Lighting consumption
* Temperature
* Occupancy
* Non-working-hour usage

It can then simulate a scenario such as:

> **What happens if lighting consumption is reduced by 15%?**

The system estimates:

* Energy saved
* Percentage reduction
* Cost savings
* CO₂ reduction

---

# 🎯 Problem Statement

Educational campuses consume significant amounts of electricity through:

* HVAC systems
* Lighting
* Computer laboratories
* Classrooms
* Administrative buildings
* Other electrical equipment

Conventional monitoring systems often focus on displaying consumption rather than explaining **why abnormal consumption occurs** or **what actions could reduce it**.

EcoSense AI addresses this problem by combining prediction, anomaly detection, analysis, simulation, and optimization into a single decision-support platform.

---

# 💡 Key Features

## 1. 📊 Sustainability Dashboard

Provides an interactive overview of campus energy consumption.

The dashboard displays:

* Total electricity consumption
* Building-wise consumption
* Expected vs actual consumption
* Energy-saving opportunities
* CO₂ impact
* Interactive charts

---

## 2. 🤖 AI Energy Prediction

A Machine Learning model predicts expected electricity consumption based on factors such as:

* Building
* Hour
* Day
* Temperature
* Humidity
* Occupancy
* HVAC usage
* Lighting usage
* Historical consumption

The predicted value is compared with actual consumption to identify unusual behavior.

### Model

**XGBoost Regressor**

Evaluation metrics include:

* MAE
* RMSE
* R² Score

---

# 3. 🚨 Anomaly Detection

EcoSense AI identifies unusual energy consumption using Machine Learning.

The system compares:

```text
Actual Consumption
        ↓
Expected Consumption
        ↓
Deviation
        ↓
Anomaly Detection
```

Anomalies are categorized into:

* Normal
* Warning
* Critical

The system also calculates the percentage deviation between actual and expected consumption.

---

# 4. 🔍 Contributing-Factor Analysis

Instead of simply reporting an anomaly, EcoSense AI investigates possible contributing factors.

The analysis considers:

* High HVAC consumption
* High lighting consumption
* High temperature
* Low occupancy
* Non-working-hour consumption

### Example Insight

```text
Unusual consumption detected in Block B.

Possible contributing factors:
• Elevated HVAC usage
• Increased lighting load
• Consumption during non-working hours
• Low occupancy compared with energy usage
```

---

# 5. 🧪 Digital Twin / What-If Simulation

The Digital Twin component allows users to simulate possible sustainability strategies.

Users can modify parameters such as:

* Lighting reduction
* HVAC reduction
* Non-working-hour reduction
* Solar generation increase

The system calculates the potential impact.

### Example

```text
Lighting Reduction: 15%

Estimated Results:

Energy Saved:       XX kWh
Energy Reduction:   XX %
Cost Saved:         ₹XX
CO₂ Reduction:      XX kg
```

This allows decision-makers to evaluate sustainability strategies before implementing them.

---

# 6. ⚡ Energy Optimization

EcoSense AI searches different combinations of energy-saving strategies.

Possible strategies include:

```text
Lighting Reduction
HVAC Reduction
Non-working-hour Reduction
Solar Generation Increase
```

The system identifies combinations that can provide significant energy savings while respecting predefined constraints.

---

# 7. 💬 AI Sustainability Advisor

The platform includes an intelligent Sustainability Advisor that provides recommendations based on the system's findings.

Example questions:

```text
Which building has the highest anomaly?

Why is Block B consuming more energy?

How can I reduce energy consumption?

What happens if lighting is reduced by 15%?

How much CO₂ can be saved?
```

The advisor uses the available sustainability analysis to provide actionable recommendations.

> **Future integration:** IBM watsonx.ai / IBM Granite can be integrated to provide a generative AI-powered sustainability advisor.

---

# 🏗️ System Architecture

```text
                 CAMPUS DATA
                     │
                     ▼
              DATA PROCESSING
                     │
                     ▼
          ┌─────────────────────┐
          │ Feature Engineering │
          └──────────┬──────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   ENERGY PREDICTION      ANOMALY DETECTION
          │                     │
          │                     ▼
          │             CONTRIBUTING-FACTOR
          │                  ANALYSIS
          │                     │
          └──────────┬──────────┘
                     ▼
              DIGITAL TWIN
              WHAT-IF MODEL
                     │
                     ▼
               OPTIMIZATION
                     │
                     ▼
          AI SUSTAINABILITY ADVISOR
                     │
                     ▼
             STREAMLIT DASHBOARD
```

---

# 🛠️ Technology Stack

| Technology               | Purpose                           |
| ------------------------ | --------------------------------- |
| Python                   | Core development                  |
| Pandas                   | Data processing                   |
| NumPy                    | Numerical computation             |
| Scikit-learn             | Machine Learning                  |
| XGBoost                  | Energy prediction                 |
| Isolation Forest         | Anomaly detection                 |
| Plotly                   | Interactive visualization         |
| Streamlit                | Web application                   |
| Joblib                   | Model persistence                 |
| IBM watsonx.ai / Granite | Planned Generative AI integration |

---

# 📁 Project Structure

```text
EcoSense-AI-Sustainable-Campus/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│
├── models/
│
└── outputs/
```

### Main Files

**`app.py`**

Contains the Streamlit application and core EcoSense AI functionality.

**`requirements.txt`**

Contains the Python dependencies required to run the application.

**`data/`**

Stores datasets used by the application.

**`models/`**

Stores trained Machine Learning models when required.

**`outputs/`**

Stores generated analysis results and outputs.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/susmijapunnam/EcoSense-AI-Sustainable-Campus.git
```

```bash
cd EcoSense-AI-Sustainable-Campus
```

---

## 2. Install Dependencies

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

# 🖥️ Application Modules

The application contains several modules:

### 📊 Overview

Provides a high-level summary of campus sustainability performance.

### 🏢 Building Analysis

Allows users to examine individual campus buildings.

### 🚨 Anomaly Detection

Displays unusual energy consumption patterns.

### 🧪 Digital Twin

Simulates sustainability interventions.

### ⚡ Optimization

Identifies potential energy-saving strategies.

### 💬 AI Sustainability Advisor

Provides intelligent sustainability recommendations.

### 📈 Model Performance

Displays Machine Learning evaluation metrics.

---

# 🔬 Machine Learning Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
XGBoost Prediction Model
   ↓
Expected Consumption
   ↓
Residual Calculation
   ↓
Isolation Forest
   ↓
Anomaly Detection
   ↓
Contributing-Factor Analysis
```

---

# 🌍 Sustainability Impact

EcoSense AI is designed to support institutions in:

* Reducing unnecessary electricity consumption
* Identifying inefficient building operations
* Reducing energy costs
* Improving resource utilization
* Increasing renewable-energy utilization
* Supporting data-driven sustainability decisions
* Reducing potential carbon emissions

---

# 📊 Demonstration Scenario

A typical demonstration workflow is:

```text
1. Open Sustainability Dashboard
             ↓
2. Select Block B
             ↓
3. Detect abnormal consumption
             ↓
4. Analyze contributing factors
             ↓
5. Open Digital Twin
             ↓
6. Reduce lighting by 15%
             ↓
7. Calculate energy savings
             ↓
8. Calculate cost savings
             ↓
9. Calculate CO₂ reduction
             ↓
10. Ask AI Sustainability Advisor
```

---

# 📌 Current Data

The current demonstration version can generate **synthetic campus energy data** for development and demonstration purposes.

The synthetic dataset includes variables such as:

```text
Timestamp
Building
Electricity Consumption
HVAC Consumption
Lighting Consumption
Occupancy
Temperature
Humidity
Working Hours
Weekend
Holiday
Solar Generation
```

> **Note:** Synthetic data is used for demonstration and development. It should not be represented as actual institutional energy-consumption data.

---

# 🔮 Future Scope

Future versions can include:

* 🔗 Real-time IoT energy meters
* ☀️ Solar-panel monitoring
* 🏢 Real campus building data
* 🤖 IBM Granite-powered sustainability advisor
* 🌐 Cloud deployment
* 📱 Mobile-friendly interface
* 🔔 Real-time anomaly alerts
* 🗺️ Campus digital twin visualization
* 📈 Long-term energy forecasting
* 🌱 Carbon-footprint monitoring
* 💰 Automated cost optimization
* 🔌 Smart-device integration

---

# 🎓 Internship

This project is developed as part of:

**1M1B AI for Sustainability Virtual Internship**

In collaboration with:

* **IBM SkillsBuild**
* **AICTE**
* **1M1B**

The project focuses on applying Artificial Intelligence and Machine Learning to sustainability and resource optimization.

---

# 👩‍💻 Author

**Susmija Punnam**

B.Tech – Computer Science and Engineering
SR University, Telangana, India

GitHub:
https://github.com/susmijapunnam

---

# 📜 License

This project is intended for educational, academic, and demonstration purposes.

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub.

**EcoSense AI — Turning campus data into intelligent sustainability decisions.** 🌱
