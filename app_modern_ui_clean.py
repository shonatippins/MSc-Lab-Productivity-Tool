import streamlit as st
import pandas as pd
from together import Together
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Laboratory Productivity Diagnostic Tool",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1180px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 1.5rem 1.6rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12);
    margin-bottom: 0.9rem;
}
.hero h1 {
    color: white;
    margin: 0 0 0.35rem 0;
    font-size: 2rem;
}
.hero p {
    color: #dbeafe;
    margin: 0;
    font-size: 1rem;
    line-height: 1.55;
}
.prototype-note {
    padding: 0.7rem 0.9rem;
    margin-bottom: 1.1rem;
    border: 1px solid rgba(59,130,246,.22);
    border-radius: 12px;
    background: rgba(59,130,246,.07);
    font-size: .9rem;
}
div[data-testid="stMetric"] {
    border: 1px solid rgba(148,163,184,.25);
    border-radius: 14px;
    padding: .9rem 1rem;
    box-shadow: 0 4px 14px rgba(15,23,42,.04);
}
div[data-testid="stMetricValue"] {
    font-size: 1.5rem;
    font-weight: 650;
}
.stButton > button {
    border-radius: 10px;
    min-height: 2.6rem;
    font-weight: 600;
}
div[data-testid="stAlert"],
div[data-testid="stExpander"],
div[data-testid="stFileUploader"] section {
    border-radius: 12px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: .3rem;
    padding: .2rem;
    border-radius: 12px;
    background: rgba(148,163,184,.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding-left: .8rem;
    padding-right: .8rem;
}
</style>

<div class="hero">
    <h1>Laboratory Productivity</h1>
    <p>AI-supported workflow decision support for Life Sciences SMEs — diagnose issues, explore recommendations, predict outcomes, compare interventions and evaluate performance.</p>
</div>
<div class="prototype-note">
    <strong>Decision-support prototype:</strong> outputs support investigation and professional judgement rather than replacing laboratory decision-making.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "results_df" not in st.session_state:
    st.session_state.results_df = None

if "concerns_df" not in st.session_state:
    st.session_state.concerns_df = None

if "ai_recommendation" not in st.session_state:
    st.session_state.ai_recommendation = None

if "workflow_df" not in st.session_state:
    st.session_state.workflow_df = None
    
if "productivity_outcome" not in st.session_state:
    st.session_state.productivity_outcome = None
if "predictive_model" not in st.session_state:
    st.session_state.predictive_model = None

if "model_features" not in st.session_state:
    st.session_state.model_features = None

if "model_target" not in st.session_state:
    st.session_state.model_target = None

if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = None

if "optimisation_results" not in st.session_state:
    st.session_state.optimisation_results = None

if "best_intervention" not in st.session_state:
    st.session_state.best_intervention = None

if "baseline_prediction" not in st.session_state:
    st.session_state.baseline_prediction = None

if "productivity_outcome" not in st.session_state:
    st.session_state.productivity_outcome = None
# --------------------------------------------------
# PRODUCTIVITY DIAGNOSTIC FUNCTION
# --------------------------------------------------

def assess_kpi(name, current, target, higher_is_better=True):

    if target <= 0:
        return {
            "KPI": name,
            "Current": current,
            "Target": target,
            "Gap (%)": None,
            "Status": "Invalid target"
        }

    if higher_is_better:

        if current >= target:
            gap = 0
            status = "Target achieved"
        else:
            gap = ((target - current) / target) * 100
            status = "Below target"

    else:

        if current <= target:
            gap = 0
            status = "Target achieved"
        else:
            gap = ((current - target) / target) * 100
            status = "Above target"

    return {
        "KPI": name,
        "Current": current,
        "Target": target,
        "Gap (%)": round(gap, 1),
        "Status": status
    }


# --------------------------------------------------
# INTERVENTION KNOWLEDGE BASE
# --------------------------------------------------

def get_interventions(kpi):

    interventions = {

        "Sample Throughput": [
            "Workflow bottleneck analysis",
            "Workload and capacity scheduling",
            "Sample preparation optimisation",
            "Automation of repetitive workflow stages"
        ],

        "Turnaround Time": [
            "Workflow scheduling optimisation",
            "Queue and workload management",
            "Sample preparation optimisation",
            "Automated status monitoring"
        ],

        "Equipment Downtime": [
            "Preventive maintenance scheduling",
            "Predictive maintenance",
            "Equipment utilisation analysis",
            "Workflow rescheduling around equipment availability"
        ],

        "Repeat/Error Rate": [
            "Automated quality checks",
            "Anomaly detection",
            "Standardisation of procedures",
            "Staff training and workflow review"
        ],

        "Manual Processing Time": [
            "Workflow automation",
            "Automated data extraction",
            "AI-assisted reporting",
            "Integration of disconnected digital systems"
        ]
    }

    return interventions.get(kpi, [])


# --------------------------------------------------
# AI RECOMMENDATION FUNCTION
# --------------------------------------------------

def generate_ai_recommendation(priority_issue, concerns_df):

    priority_kpi = priority_issue["KPI"]

    possible_interventions = get_interventions(priority_kpi)

    other_concerns = concerns_df[
        concerns_df["KPI"] != priority_kpi
    ][["KPI", "Gap (%)"]].to_dict("records")

    prompt = f"""
You are an AI productivity decision-support assistant for a
life-science SME laboratory.

The laboratory productivity diagnostic has identified the following:

Highest-priority productivity issue:
KPI: {priority_kpi}
Current value: {priority_issue['Current']}
Target value: {priority_issue['Target']}
Relative performance gap: {priority_issue['Gap (%)']}%

Other identified productivity concerns:
{other_concerns}

Possible improvement approaches:
{possible_interventions}

Recommend ONE approach that should be investigated first.

Your response must contain these sections:

### Recommended Approach
State one recommended approach.

### Why This Approach
Explain why this option is relevant to the identified productivity problem.

### What to Investigate First
Identify what the laboratory should examine before implementation.

### Potential Productivity Benefit
Explain how the approach could potentially improve productivity.

### Limitation or Risk
Identify one important limitation or risk.

Important rules:
- Do not claim that the recommendation will definitely improve productivity.
- Do not claim to know the confirmed root cause.
- Do not provide medical or scientific diagnostic advice.
- Treat the output as decision support for laboratory professionals.
- Base the recommendation on the supplied KPI information and listed approaches.
- Keep the response concise and practical.
"""

    client = Together(
        api_key=st.secrets["TOGETHER_API_KEY"]
    )

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=700
    )

    return response.choices[0].message.content


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "01 Diagnose",
    "02 Recommend",
    "03 Predict",
    "04 Optimise",
    "05 Evaluate"
])


# ==================================================
# TAB 1 — PRODUCTIVITY DIAGNOSIS
# ==================================================

with tab1:

    st.caption("STAGE 01  •  DIAGNOSE")
    st.header("Productivity Diagnosis")

    st.write(
        "Enter current laboratory performance and target values "
        "for each productivity indicator."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Current Performance")

        current_throughput = st.number_input(
            "Current sample throughput",
            min_value=0.0,
            value=70.0
        )

        current_turnaround = st.number_input(
            "Current turnaround time (hours)",
            min_value=0.0,
            value=8.0
        )

        current_downtime = st.number_input(
            "Current equipment downtime (%)",
            min_value=0.0,
            value=12.0
        )

        current_error_rate = st.number_input(
            "Current repeat/error rate (%)",
            min_value=0.0,
            value=8.0
        )

        current_manual_time = st.number_input(
            "Current manual processing time (hours/day)",
            min_value=0.0,
            value=4.0
        )

    with col2:

        st.subheader("Target Performance")

        target_throughput = st.number_input(
            "Target sample throughput",
            min_value=0.1,
            value=100.0
        )

        target_turnaround = st.number_input(
            "Target turnaround time (hours)",
            min_value=0.1,
            value=6.0
        )

        target_downtime = st.number_input(
            "Target equipment downtime (%)",
            min_value=0.1,
            value=5.0
        )

        target_error_rate = st.number_input(
            "Target repeat/error rate (%)",
            min_value=0.1,
            value=3.0
        )

        target_manual_time = st.number_input(
            "Target manual processing time (hours/day)",
            min_value=0.1,
            value=2.0
        )

    if st.button("Analyse Productivity", key="analyse_productivity"):

        results = []

        results.append(
            assess_kpi(
                "Sample Throughput",
                current_throughput,
                target_throughput,
                True
            )
        )

        results.append(
            assess_kpi(
                "Turnaround Time",
                current_turnaround,
                target_turnaround,
                False
            )
        )

        results.append(
            assess_kpi(
                "Equipment Downtime",
                current_downtime,
                target_downtime,
                False
            )
        )

        results.append(
            assess_kpi(
                "Repeat/Error Rate",
                current_error_rate,
                target_error_rate,
                False
            )
        )

        results.append(
            assess_kpi(
                "Manual Processing Time",
                current_manual_time,
                target_manual_time,
                False
            )
        )

        results_df = pd.DataFrame(results)

        concerns_df = results_df[
            results_df["Status"] != "Target achieved"
        ].copy()

        concerns_df = concerns_df.sort_values(
            by="Gap (%)",
            ascending=False
        )

        st.session_state.results_df = results_df
        st.session_state.concerns_df = concerns_df
        st.session_state.ai_recommendation = None

    if st.session_state.results_df is not None:

        results_df = st.session_state.results_df
        concerns_df = st.session_state.concerns_df

        total_kpis = len(results_df)
        total_concerns = len(concerns_df)
        targets_achieved = total_kpis - total_concerns

        if total_concerns > 0:
            priority_issue = concerns_df.iloc[0]
            priority_name = priority_issue["KPI"]
        else:
            priority_issue = None
            priority_name = "None"

        st.subheader("At a Glance")

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        with summary_col1:
            st.metric("KPIs Assessed", total_kpis)

        with summary_col2:
            st.metric("Targets Achieved", targets_achieved)

        with summary_col3:
            st.metric(
                "Areas Requiring Investigation",
                total_concerns
            )

        with summary_col4:
            st.metric(
                "Highest Priority Area",
                priority_name
            )

        st.subheader("Detailed KPI Results")

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Performance Gaps")

        if total_concerns > 0:

            chart_data = concerns_df[
                ["KPI", "Gap (%)"]
            ].set_index("KPI")

            st.bar_chart(chart_data)

            st.subheader(
                "Highest Priority Area for Investigation"
            )

            st.warning(
                f"""
                **{priority_issue['KPI']}**

                Current performance: **{priority_issue['Current']}**

                Target performance: **{priority_issue['Target']}**

                Relative performance gap: **{priority_issue['Gap (%)']}%**

                This indicator currently shows the largest relative
                deviation from its target and may warrant further investigation.
                """
            )

        else:

            st.success(
                "All selected productivity indicators are meeting their targets."
            )


# ==================================================
# TAB 2 — AI RECOMMENDATION
# ==================================================

with tab2:

    st.caption("STAGE 02  •  RECOMMEND")
    st.header("AI-Supported Recommendation")

    if (
        st.session_state.results_df is None
        or st.session_state.concerns_df is None
    ):

        st.info(
            "Run the productivity diagnosis first before generating "
            "an AI-supported recommendation."
        )

    elif len(st.session_state.concerns_df) == 0:

        st.success(
            "No productivity concerns were identified, so an "
            "intervention recommendation is not currently required."
        )

    else:

        concerns_df = st.session_state.concerns_df
        priority_issue = concerns_df.iloc[0]

        st.subheader("Priority Issue")

        st.warning(
            f"""
            **{priority_issue['KPI']}**

            Current value: **{priority_issue['Current']}**

            Target value: **{priority_issue['Target']}**

            Performance gap: **{priority_issue['Gap (%)']}%**
            """
        )

        st.subheader("Approaches to Consider")

        possible_interventions = get_interventions(
            priority_issue["KPI"]
        )

        for intervention in possible_interventions:
            st.write(f"• {intervention}")

        st.write(
            "The hosted Llama model considers the diagnostic "
            "results and available intervention approaches before "
            "recommending which option should be investigated first."
        )

        if st.button(
            "Generate AI Recommendation",
            key="generate_ai_recommendation"
        ):

            with st.spinner(
                "Generating AI-supported recommendation..."
            ):

                try:

                    recommendation = generate_ai_recommendation(
                        priority_issue,
                        concerns_df
                    )

                    st.session_state.ai_recommendation = recommendation

                except Exception as e:

                    st.error(
                        "The AI recommendation could not be generated."
                    )

                    st.write(
                        "Check that the Together AI API key is configured "
                        "correctly and that API access is available."
                    )

                    st.write(e)

        if st.session_state.ai_recommendation is not None:

            st.markdown("---")

            st.markdown(
                st.session_state.ai_recommendation
            )

            st.info(
                "This recommendation is intended as decision support. "
                "It does not establish the root cause of the productivity "
                "problem and should be reviewed by appropriate laboratory "
                "personnel before operational changes are made."
            )


# ==================================================
# TAB 3 — PREDICTIVE WORKFLOW ANALYSIS
# ==================================================

with tab3:

    st.caption("STAGE 03  •  PREDICT")
    st.header("Predictive Workflow Analysis")

    st.write(
        "Upload historical laboratory workflow data to train a predictive "
        "machine-learning model and estimate future productivity performance."
    )

    uploaded_file = st.file_uploader(
        "Upload laboratory workflow data",
        type=["csv"],
        key="workflow_upload"
    )

    if uploaded_file is not None:

        try:

            workflow_df = pd.read_csv(uploaded_file)

            st.session_state.workflow_df = workflow_df

            st.success("Dataset uploaded successfully.")

            # ------------------------------------------
            # DATASET PREVIEW
            # ------------------------------------------

            st.subheader("Dataset Preview")

            st.dataframe(
                workflow_df.head(10),
                use_container_width=True
            )

            # ------------------------------------------
            # DATASET INFORMATION
            # ------------------------------------------

            col_a, col_b = st.columns(2)

            with col_a:
                st.metric(
                    "Number of Rows",
                    workflow_df.shape[0]
                )

            with col_b:
                st.metric(
                    "Number of Columns",
                    workflow_df.shape[1]
                )

            # ------------------------------------------
            # NUMERIC COLUMNS
            # ------------------------------------------

            numeric_columns = workflow_df.select_dtypes(
                include=["number"]
            ).columns.tolist()

            st.subheader("Configure Prediction")

            if len(numeric_columns) < 2:

                st.warning(
                    "The uploaded dataset does not contain enough numeric "
                    "columns for predictive modelling."
                )

            else:

                # --------------------------------------
                # TARGET SELECTION
                # --------------------------------------

                default_target = (
                    numeric_columns.index("Turnaround_Time_Hours")
                    if "Turnaround_Time_Hours" in numeric_columns
                    else 0
                )

                target_column = st.selectbox(
                    "Select the productivity outcome to predict",
                    options=numeric_columns,
                    index=default_target
                )

                # --------------------------------------
                # FEATURE SELECTION
                # --------------------------------------

                available_features = [
                    column
                    for column in numeric_columns
                    if column != target_column
                ]

                # Exclude Day by default
                default_features = [
                    column
                    for column in available_features
                    if column != "Day"
                ]

                selected_features = st.multiselect(
                    "Select the workflow factors to use as predictors",
                    options=available_features,
                    default=default_features
                )

                # --------------------------------------
                # MODEL TRAINING
                # --------------------------------------

                if st.button(
                    "Train Predictive Model",
                    key="train_predictive_model"
                ):

                    if len(selected_features) == 0:

                        st.warning(
                            "Select at least one predictor before training."
                        )

                    else:

                        model_data = workflow_df[
                            selected_features + [target_column]
                        ].dropna()

                        X = model_data[selected_features]
                        y = model_data[target_column]

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42
                        )

                        model = RandomForestRegressor(
                            n_estimators=200,
                            random_state=42
                        )

                        model.fit(X_train, y_train)

                        predictions = model.predict(X_test)

                        mae = mean_absolute_error(
                            y_test,
                            predictions
                        )

                        rmse = mean_squared_error(
                            y_test,
                            predictions
                        ) ** 0.5

                        r2 = r2_score(
                            y_test,
                            predictions
                        )

                        st.success(
                            "Predictive model trained successfully."
                        )

                        # ----------------------------------
                        # MODEL PERFORMANCE
                        # ----------------------------------

                        st.subheader("Model Performance")

                        metric1, metric2, metric3 = st.columns(3)

                        with metric1:
                            st.metric(
                                "MAE",
                                f"{mae:.2f}"
                            )

                        with metric2:
                            st.metric(
                                "RMSE",
                                f"{rmse:.2f}"
                            )

                        with metric3:
                            st.metric(
                                "R²",
                                f"{r2:.3f}"
                            )

                        # ----------------------------------
                        # FEATURE IMPORTANCE
                        # ----------------------------------

                        st.subheader("Feature Importance")

                        importance_df = pd.DataFrame({
                            "Feature": selected_features,
                            "Importance": model.feature_importances_
                        }).sort_values(
                            by="Importance",
                            ascending=False
                        )

                        st.dataframe(
                            importance_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.bar_chart(
                            importance_df.set_index("Feature")
                        )

                        # Store model information
                        st.session_state.predictive_model = model
                        st.session_state.model_features = selected_features
                        st.session_state.model_target = target_column
                        st.session_state.model_metrics = {
                            "MAE": mae,
                            "RMSE": rmse,
                            "R2": r2
                        }

        except Exception as e:

            st.error(
                "The dataset could not be processed."
            )

            st.write(e)

    else:

        st.info(
            "Upload a CSV dataset to begin predictive workflow analysis."
        )

    st.subheader("Predict a New Workflow Scenario")

    if (
        st.session_state.get("predictive_model") is not None
        and st.session_state.get("workflow_df") is not None
    ):

        model = st.session_state.predictive_model
        model_features = st.session_state.model_features
        workflow_df = st.session_state.workflow_df

        st.write(
            "Enter a new set of laboratory workflow conditions to estimate "
            "the predicted productivity outcome."
        )

        prediction_inputs = {}

        for feature in model_features:

            prediction_inputs[feature] = st.number_input(
                f"Enter {feature}",
                value=float(workflow_df[feature].median()),
                key=f"predict_{feature}"
            )

        if st.button(
            "Predict Workflow Performance",
            key="predict_workflow"
        ):

            new_data = pd.DataFrame(
                [prediction_inputs]
            )

            predicted_value = model.predict(new_data)[0]

            st.success(
                f"Predicted {st.session_state.model_target}: "
                f"{predicted_value:.2f}"
            )

    else:

        st.info(
            "Train a predictive model first to enable new workflow predictions."
        )


# ==================================================
# TAB 4 — WORKFLOW OPTIMISATION
# ==================================================

with tab4:

    st.caption("STAGE 04  •  OPTIMISE")
    st.header("AI-Supported Workflow Optimisation")

    st.write(
        "This section uses the trained predictive model to compare "
        "different productivity improvement scenarios and identify "
        "which intervention could produce the greatest predicted "
        "improvement in workflow performance."
    )

    # Check that a predictive model has been trained
    if "predictive_model" not in st.session_state:

        st.info(
            "Train a predictive model in the Predictive Workflow Analysis "
            "tab before using workflow optimisation."
        )

    else:

        model = st.session_state.predictive_model
        model_features = st.session_state.model_features
        target = st.session_state.model_target

        # We currently optimise turnaround time
        if target != "Turnaround_Time_Hours":

            st.warning(
                "The current optimisation prototype is designed for "
                "Turnaround_Time_Hours as the prediction target."
            )

        else:

            st.subheader("Baseline Scenario")

            st.write(
                "Enter the current workflow conditions. These values "
                "will be used as the baseline against which different "
                "interventions are compared."
            )

            workflow_df = st.session_state.workflow_df

            baseline_values = {}

            for feature in model_features:

                default_value = float(
                    workflow_df[feature].median()
                )

                baseline_values[feature] = st.number_input(
                    f"Baseline {feature}",
                    value=default_value,
                    key=f"baseline_{feature}"
                )

            baseline_df = pd.DataFrame(
                [baseline_values]
            )

            baseline_prediction = model.predict(
                baseline_df
            )[0]

            st.metric(
                "Predicted Baseline Turnaround Time",
                f"{baseline_prediction:.2f} hours"
            )


            # --------------------------------------------------
            # INTERVENTION SETTINGS
            # --------------------------------------------------

            st.subheader("Compare Intervention Scenarios")

            st.write(
                "Select the level of improvement that could potentially "
                "be achieved for each workflow factor."
            )

            downtime_reduction = st.slider(
                "Reduce equipment downtime by (%)",
                min_value=0,
                max_value=80,
                value=30,
                step=5
            )

            error_reduction = st.slider(
                "Reduce repeat/error rate by (%)",
                min_value=0,
                max_value=80,
                value=30,
                step=5
            )

            manual_reduction = st.slider(
                "Reduce manual processing time by (%)",
                min_value=0,
                max_value=80,
                value=30,
                step=5
            )


            # --------------------------------------------------
            # RUN OPTIMISATION
            # --------------------------------------------------

            if st.button(
                "Compare Workflow Interventions",
                key="compare_interventions"
            ):

                scenarios = []

                # ----------------------------------------------
                # DOWNTIME SCENARIO
                # ----------------------------------------------

                if "Equipment_Downtime_Percent" in model_features:

                    downtime_scenario = baseline_values.copy()

                    downtime_scenario[
                        "Equipment_Downtime_Percent"
                    ] = (
                        baseline_values[
                            "Equipment_Downtime_Percent"
                        ]
                        * (1 - downtime_reduction / 100)
                    )

                    downtime_df = pd.DataFrame(
                        [downtime_scenario]
                    )

                    downtime_prediction = model.predict(
                        downtime_df
                    )[0]

                    scenarios.append({
                        "Intervention":
                            "Reduce Equipment Downtime",
                        "Predicted Turnaround":
                            downtime_prediction,
                        "Improvement":
                            baseline_prediction
                            - downtime_prediction
                    })


                # ----------------------------------------------
                # ERROR RATE SCENARIO
                # ----------------------------------------------

                if "Repeat_Error_Rate_Percent" in model_features:

                    error_scenario = baseline_values.copy()

                    error_scenario[
                        "Repeat_Error_Rate_Percent"
                    ] = (
                        baseline_values[
                            "Repeat_Error_Rate_Percent"
                        ]
                        * (1 - error_reduction / 100)
                    )

                    error_df = pd.DataFrame(
                        [error_scenario]
                    )

                    error_prediction = model.predict(
                        error_df
                    )[0]

                    scenarios.append({
                        "Intervention":
                            "Reduce Repeat/Error Rate",
                        "Predicted Turnaround":
                            error_prediction,
                        "Improvement":
                            baseline_prediction
                            - error_prediction
                    })


                # ----------------------------------------------
                # MANUAL PROCESSING SCENARIO
                # ----------------------------------------------

                if "Manual_Processing_Hours" in model_features:

                    manual_scenario = baseline_values.copy()

                    manual_scenario[
                        "Manual_Processing_Hours"
                    ] = (
                        baseline_values[
                            "Manual_Processing_Hours"
                        ]
                        * (1 - manual_reduction / 100)
                    )

                    manual_df = pd.DataFrame(
                        [manual_scenario]
                    )

                    manual_prediction = model.predict(
                        manual_df
                    )[0]

                    scenarios.append({
                        "Intervention":
                            "Reduce Manual Processing Time",
                        "Predicted Turnaround":
                            manual_prediction,
                        "Improvement":
                            baseline_prediction
                            - manual_prediction
                    })


                # --------------------------------------------------
                # RESULTS
                # --------------------------------------------------

                if len(scenarios) > 0:

                    optimisation_df = pd.DataFrame(
                        scenarios
                    )

                    optimisation_df[
                        "Predicted Turnaround"
                    ] = optimisation_df[
                        "Predicted Turnaround"
                    ].round(2)

                    optimisation_df[
                        "Improvement"
                    ] = optimisation_df[
                        "Improvement"
                    ].round(2)

                    optimisation_df = optimisation_df.sort_values(
                        by="Predicted Turnaround",
                        ascending=True
                    )

                    st.subheader(
                        "Predicted Intervention Outcomes"
                    )

                    st.dataframe(
                        optimisation_df,
                        use_container_width=True,
                        hide_index=True
                    )


                    # --------------------------------------------------
                    # BEST INTERVENTION
                    # --------------------------------------------------

                    best_intervention = (
                        optimisation_df.iloc[0]
                    )

                    st.subheader(
                        "Best Predicted Intervention"
                    )

                    st.success(
                        f"""
                        **{best_intervention['Intervention']}**

                        Baseline predicted turnaround:
                        **{baseline_prediction:.2f} hours**

                        Predicted turnaround after intervention:
                        **{best_intervention['Predicted Turnaround']:.2f} hours**

                        Predicted time improvement:
                        **{best_intervention['Improvement']:.2f} hours**
                        """
                    )


                    # --------------------------------------------------
                    # PREDICTED PRODUCTIVITY OUTCOME
                    # --------------------------------------------------

                    predicted_turnaround = float(
                        best_intervention["Predicted Turnaround"]
                    )

                    time_saved = (
                        baseline_prediction - predicted_turnaround
                    )

                    if baseline_prediction > 0:

                        improvement_percentage = (
                            time_saved / baseline_prediction
                        ) * 100

                    else:

                        improvement_percentage = 0


                    st.subheader("Productivity Outcome")

                    productivity_col1, productivity_col2, productivity_col3 = st.columns(3)

                    with productivity_col1:

                        st.metric(
                            "Baseline Turnaround",
                            f"{baseline_prediction:.2f} hours"
                        )

                    with productivity_col2:

                        st.metric(
                            "Predicted Turnaround",
                            f"{predicted_turnaround:.2f} hours"
                        )

                    with productivity_col3:

                        st.metric(
                            "Turnaround-Time Improvement",
                            f"{improvement_percentage:.1f}%"
                        )


                    # --------------------------------------------------
                    # INTERPRET PRODUCTIVITY RESULT
                    # --------------------------------------------------

                    if improvement_percentage > 0:

                        st.success(
                            f"""
                            ### Potential Productivity Improvement

                            The **{best_intervention['Intervention']}** scenario
                            produced the strongest predicted workflow outcome.

                            The model predicts that turnaround time could decrease
                            from **{baseline_prediction:.2f} hours** to
                            **{predicted_turnaround:.2f} hours**.

                            This represents a predicted time saving of
                            **{time_saved:.2f} hours**, equivalent to a
                            **{improvement_percentage:.1f}% reduction in turnaround time**
                            under the modelled scenario.

                            This indicates a potential improvement in laboratory
                            workflow productivity and should be investigated
                            further before implementation.
                            """
                        )

                    else:

                        st.warning(
                            """
                            The tested intervention scenarios did not produce
                            a predicted improvement in turnaround time.
                            Further workflow investigation or alternative
                            interventions may therefore be required.
                            """
                        )


                    # --------------------------------------------------
                    # VISUAL COMPARISON
                    # --------------------------------------------------

                    chart_df = optimisation_df[
                        [
                            "Intervention",
                            "Predicted Turnaround"
                        ]
                    ].set_index(
                        "Intervention"
                    )

                    st.subheader("Intervention Comparison")

                    st.bar_chart(
                        chart_df
                    )


                    # --------------------------------------------------
                    # STORE RESULTS
                    # --------------------------------------------------

                    st.session_state.optimisation_results = optimisation_df
                    st.session_state.best_intervention = best_intervention
                    st.session_state.baseline_prediction = baseline_prediction

                    st.session_state.productivity_outcome = {
                        "intervention": best_intervention["Intervention"],
                        "baseline_turnaround": baseline_prediction,
                        "predicted_turnaround": predicted_turnaround,
                        "time_saved": time_saved,
                        "improvement_percentage": improvement_percentage
                    }

                    st.info(
                        "These results are modelled scenarios rather than "
                        "guaranteed real-world improvements. Actual outcomes "
                        "would need to be validated using observed laboratory data."
                    )

                else:

                    st.warning(
                        "The selected predictive model does not contain "
                        "the workflow variables required for the current "
                        "optimisation scenarios."
                    )
# ==================================================
# TAB 5 — OUTCOME EVALUATION
# ==================================================

# ==================================================
# TAB 5 — OUTCOME EVALUATION
# ==================================================

with tab5:

    st.caption("STAGE 05  •  EVALUATE")
    st.header("Outcome Evaluation")

    st.write(
        "This section evaluates whether the selected intervention "
        "produced a measurable improvement in laboratory workflow "
        "productivity by comparing baseline, predicted and observed results."
    )

    # --------------------------------------------------
    # CHECK FOR OPTIMISATION RESULT
    # --------------------------------------------------

    if st.session_state.productivity_outcome is None:

        st.info(
            "Run the Workflow Optimisation analysis first before "
            "evaluating an intervention outcome."
        )

    else:

        outcome = st.session_state.productivity_outcome

        baseline_turnaround = outcome["baseline_turnaround"]
        predicted_turnaround = outcome["predicted_turnaround"]
        predicted_improvement = outcome["improvement_percentage"]
        intervention = outcome["intervention"]

        # --------------------------------------------------
        # MODELLED OUTCOME
        # --------------------------------------------------

        st.subheader("Modelled Outcome")

        st.write(
            f"Selected intervention: **{intervention}**"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Baseline Turnaround",
                f"{baseline_turnaround:.2f} hours"
            )

        with col2:

            st.metric(
                "Predicted Turnaround",
                f"{predicted_turnaround:.2f} hours"
            )

        with col3:

            st.metric(
                "Predicted Improvement",
                f"{predicted_improvement:.1f}%"
            )


        # --------------------------------------------------
        # OBSERVED POST-INTERVENTION RESULT
        # --------------------------------------------------

        st.subheader("Observed Outcome")

        st.write(
            "Enter the turnaround time measured after the intervention "
            "was implemented or tested."
        )

        observed_turnaround = st.number_input(
            "Observed turnaround time after intervention (hours)",
            min_value=0.1,
            value=float(predicted_turnaround),
            step=0.1
        )


        # --------------------------------------------------
        # EVALUATE OUTCOME
        # --------------------------------------------------

        if st.button(
            "Evaluate Productivity Outcome",
            key="evaluate_outcome"
        ):

            actual_time_saved = (
                baseline_turnaround - observed_turnaround
            )

            if baseline_turnaround > 0:

                actual_improvement_percentage = (
                    actual_time_saved / baseline_turnaround
                ) * 100

            else:

                actual_improvement_percentage = 0


            prediction_error = abs(
                observed_turnaround - predicted_turnaround
            )


            # --------------------------------------------------
            # RESULTS
            # --------------------------------------------------

            st.subheader("Evaluation Results")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Observed Turnaround",
                    f"{observed_turnaround:.2f} hours"
                )

            with result_col2:

                st.metric(
                    "Actual Turnaround Improvement",
                    f"{actual_improvement_percentage:.1f}%"
                )

            with result_col3:

                st.metric(
                    "Prediction Error",
                    f"{prediction_error:.2f} hours"
                )


            # --------------------------------------------------
            # INTERPRET RESULT
            # --------------------------------------------------

            if actual_improvement_percentage > 0:

                st.success(
                    f"""
                    ### Productivity Improvement Observed

                    Following the **{intervention}** intervention,
                    turnaround time changed from
                    **{baseline_turnaround:.2f} hours** to
                    **{observed_turnaround:.2f} hours**.

                    This represents an observed reduction of
                    **{actual_time_saved:.2f} hours**, equivalent to a
                    **{actual_improvement_percentage:.1f}% improvement
                    in turnaround time**.

                    The predictive model originally estimated a
                    **{predicted_improvement:.1f}% improvement**.

                    The difference between the predicted and observed
                    turnaround time was **{prediction_error:.2f} hours**.

                    These results indicate that the intervention was
                    associated with improved workflow performance in
                    this evaluation scenario.
                    """
                )

            elif actual_improvement_percentage == 0:

                st.warning(
                    """
                    No measurable turnaround-time improvement was observed.
                    The intervention did not change the productivity outcome
                    within this evaluation scenario.
                    """
                )

            else:

                st.error(
                    f"""
                    ### Productivity Performance Deteriorated

                    Turnaround time increased following the intervention.

                    The observed change was
                    **{actual_improvement_percentage:.1f}%**.

                    This suggests that the intervention did not improve
                    turnaround-time productivity under the evaluated
                    conditions and further investigation would be required.
                    """
                )


            # --------------------------------------------------
            # COMPARISON TABLE
            # --------------------------------------------------

            comparison_df = pd.DataFrame({
                "Stage": [
                    "Baseline",
                    "Predicted After Intervention",
                    "Observed After Intervention"
                ],

                "Turnaround Time (Hours)": [
                    baseline_turnaround,
                    predicted_turnaround,
                    observed_turnaround
                ]
            })

            st.subheader("Outcome Comparison")

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                comparison_df.set_index("Stage")
            )


            # --------------------------------------------------
            # IMPORTANT LIMITATION
            # --------------------------------------------------

            st.info(
                "A reduction in turnaround time represents improvement "
                "in one dimension of laboratory productivity and should "
                "not be interpreted as a percentage increase in total "
                "organisational productivity. Real-world validation would "
                "require observed laboratory data and consideration of "
                "additional productivity indicators."
            )