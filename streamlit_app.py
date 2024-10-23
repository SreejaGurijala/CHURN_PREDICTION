import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut


if 'GROQ_API_KEY' in os.environ:
  api_key = os.environ['GROQ_API_KEY']
else:
  api_key = st.secrets['GROQ_API_KEY']

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY"))


def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


xgboost_model = load_model("xgb_model_V2.pkl")
naive_bayes_model = load_model("nb_model_V2.pkl")
random_forest_model = load_model("rf_model_V2.pkl")
decision_tree_model = load_model("dt_model_V2.pkl")
svm_model = load_model("svm_model_V2.pkl")
knn_model = load_model("knn_model_V2.pkl")
voting_classifier_model = load_model("voting_clf_V2.pkl")
xgboost_SMOTE_model = load_model("xgboost-SMOTE_V2.pkl")
xgboost_featureEngineering_model = load_model(
    "xgboost-featureEngineered_V2.pkl")
stacking_model = load_model("stacking_model_V2.pkl")
gradient_boosting_model = load_model("gb_model_V2.pkl")


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_memeber,
                  estimated_salary):

    credit_score_quantiles = {
        "Low": 584,  # Change based on the dataset's 25th percentile value
        "Medium": 652,  # Example: Adjust based on 50th percentile value
        "Good": 718,  # Example: Adjust based on 75th percentile value
        "Excellent": 850  # Example: Adjust based on dataset max value
    }

    if credit_score <= credit_score_quantiles['Low']:
        credit_score_category = 'Low'
    elif credit_score <= credit_score_quantiles['Medium']:
        credit_score_category = 'Medium'
    elif credit_score <= credit_score_quantiles['Good']:
        credit_score_category = 'Good'
    else:
        credit_score_category = 'Excellent'

    input_dict = {
        "CreditScore": [credit_score],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [int(has_credit_card)],
        "IsActiveMember": [int(is_active_memeber)],
        "EstimatedSalary": [estimated_salary],
        "Geography_France": [1 if location == "France" else 0],
        "Geography_Germany": [1 if location == "Germany" else 0],
        "Geography_Spain": [1 if location == "Spain" else 0],
        "Gender_Male": [1 if gender == "Male" else 0],
        "Gender_Female": [1 if gender == "Female" else 0],
        "CLV": [(balance * estimated_salary) / 100000],
        "TenureAgeRatio": [tenure / age],
        "AgeGroup_MiddleAge": [1 if age > 30 and age <= 45 else 0],
        "AgeGroup_Senior": [1 if age > 45 and age <= 60 else 0],
        "AgeGroup_Elderly": [1 if age > 60 and age <= 100 else 0],
        "BalanceSalaryRatio": [balance / estimated_salary],
        "ActivityLevelScore": [num_products * is_active_memeber],
        "TenureAgeInteraction": [tenure * age],
        "AgeEstimatedSalaryInteraction": [age * estimated_salary],
        "AgeCreditScoreInteraction": [age * credit_score],
        "CreditScoreBalanceInteraction": [credit_score * balance],
        "CreditScoreCategory_Low":
        [1 if credit_score_category == 'Low' else 0],
        "CreditScoreCategory_Medium":
        [1 if credit_score_category == 'Medium' else 0],
        "CreditScoreCategory_Good":
        [1 if credit_score_category == 'Good' else 0],
        "CreditScoreCategory_Excellent":
        [1 if credit_score_category == 'Excellent' else 0],
        "TenureGroup_New": [1 if tenure > 0 and tenure <= 3 else 0],
        "TenureGroup_Medium": [1 if tenure > 3 and tenure <= 6 else 0],
        "TenureGroup_Long-term": [1 if tenure > 6 and tenure <= 10 else 0],
    }

    input_df = pd.DataFrame(input_dict)
    return input_df, input_dict


def make_predictions(input_df, input_dict):

    probabilities = {
        "XGBoost": xgboost_model.predict_proba(input_df)[0][1],
        "Random_Forest": random_forest_model.predict_proba(input_df)[0][1],
        "K-Nearest Neighbors": knn_model.predict_proba(input_df)[0][1],
        "Naive Bayes": naive_bayes_model.predict_proba(input_df)[0][1],
        "Decison Tree": decision_tree_model.predict_proba(input_df)[0][1],
        "SVM": svm_model.predict_proba(input_df)[0][1],
        "Voting Classifier":
        voting_classifier_model.predict_proba(input_df)[0][1],
        "XGBoost-SMOTE": xgboost_SMOTE_model.predict_proba(input_df)[0][1],
        "Stacking": stacking_model.predict_proba(input_df)[0][1],
        "Gradient Boosting":
        gradient_boosting_model.predict_proba(input_df)[0][1]
    }
    avg_probability = np.mean(list(probabilities.values()))

    # st.markdown("### Model Probabilities")
    # for model, prob in probabilities.items():
    #   st.write(f"{model} {prob} ")
    # st.write(f"Average Probability: {avg_probability}")

    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"The customer has a {avg_probability:.2%} chance of churning.")

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    return avg_probability


def explain_prediction(probability, input_dict, surname):

    prompt = f"""You are a customer relationship expert at a bank, tasked with interpreting data-driven insights about customer churn risk.

A customer named {surname} has been identified as having a {round(probability * 100, 1)}% likelihood of leaving the bank. Here's their relevant information:

{input_dict}

Key factors influencing customer retention, in order of importance:
 0	CreditScore	0.008054
1	Age	0.032190
2	Tenure	0.007498
3	Balance	0.011675
4	NumOfProducts	0.073608
5	HasCrCard	0.007066
6	IsActiveMember	0.042533
7	EstimatedSalary	0.007581
8	Geography_France	0.010124
9	Geography_Germany	0.025802
10	Geography_Spain	0.008856
11	Gender_Female	0.012987
12	Gender_Male	0.000000
13	CLV	0.010273
14	TenureAgeRatio	0.007633
15	AgeGroup_MiddleAge	0.009635
16	AgeGroup_Senior	0.634856
17	AgeGroup_Elderly	0.000000
18	BalanceSalaryRatio	0.012130
19	ActivityLevelScore	0.014399
20	TenureAgeInteraction	0.007686
21	AgeEstimatedSalaryInteraction	0.008808
22	AgeCreditScoreInteraction	0.007108
23	CreditScoreBalanceInteraction	0.011112
24	CreditScoreCategory_Low	0.000000
25	CreditScoreCategory_Medium	0.007671
26	CreditScoreCategory_Good	0.009362
27	CreditScoreCategory_Excellent	0.000000
28	TenureGroup_New	0.005650
29	TenureGroup_Medium	0.005703
30	TenureGroup_Long-term	0.000000

Average profiles:
Churned customers: {df[df['Exited'] == 1].describe()}
Loyal customers: {df[df['Exited'] == 0].describe()}

Your task:
1. Analyze the customer's data in relation to these factors and averages.
2. Identify 3-4 key reasons why this customer might be considering leaving the bank.
3. For each reason, suggest a personalized retention strategy or product that could address the customer's potential concerns.

Craft your response as 3-4 brief paragraphs, each addressing one key factor and its corresponding retention strategy. Keep the tone professional and empathetic, focusing on how the bank can better serve the customer's needs.

Do not mention churn probabilities, data analysis, or machine learning in your response. Frame your insights as observations about the customer's banking habits and needs.
    """

    #     prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

    #   Your machine learning model has predicted that a custome named {surname} has a probability of {round(probability * 100, 1)}% of churning, based on the information provided below.

    #   Here is the customer's information:
    #   {input_dict}

    #   Here are the machine learning model;s top 10 mmost important features for predicting churn in descending order:

    #   feature	importance and their associated importances -
    #   0	CreditScore	0.008054
    # 1	Age	0.032190
    # 2	Tenure	0.007498
    # 3	Balance	0.011675
    # 4	NumOfProducts	0.073608
    # 5	HasCrCard	0.007066
    # 6	IsActiveMember	0.042533
    # 7	EstimatedSalary	0.007581
    # 8	Geography_France	0.010124
    # 9	Geography_Germany	0.025802
    # 10	Geography_Spain	0.008856
    # 11	Gender_Female	0.012987
    # 12	Gender_Male	0.000000
    # 13	CLV	0.010273
    # 14	TenureAgeRatio	0.007633
    # 15	AgeGroup_MiddleAge	0.009635
    # 16	AgeGroup_Senior	0.634856
    # 17	AgeGroup_Elderly	0.000000
    # 18	BalanceSalaryRatio	0.012130
    # 19	ActivityLevelScore	0.014399
    # 20	TenureAgeInteraction	0.007686
    # 21	AgeEstimatedSalaryInteraction	0.008808
    # 22	AgeCreditScoreInteraction	0.007108
    # 23	CreditScoreBalanceInteraction	0.011112
    # 24	CreditScoreCategory_Low	0.000000
    # 25	CreditScoreCategory_Medium	0.007671
    # 26	CreditScoreCategory_Good	0.009362
    # 27	CreditScoreCategory_Excellent	0.000000
    # 28	TenureGroup_New	0.005650
    # 29	TenureGroup_Medium	0.005703
    # 30	TenureGroup_Long-term	0.000000

    # {pd.set_option('display.max_columns', None)}

    # Here are summary statistices for churned customers:
    # {df[df['Exited'] == 1].describe()}

    # Here are summary statistics for non-churned customers:
    # {df[df['Exited'] == 0].describe()}

    # -If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at a risk of churning.
    # - If the customer has less than 40% risk of churning, generate a 3 senetence explanation of why they might not be ata risk of churning.
    # - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.
    #   Dont mention the probability of churning or the maching learning model, or say anything like "Based on the machine learning model's prediction and top10 most important features", just explain the prediction. """

    raw_response = client.chat.completions.create(
        #model="llama-3.2-3b-preview",
        model="llama3-groq-70b-8192-tool-use-preview",
        messages=[{
            "role":
            "system",
            "content":
            "You are a helpful assistant that explains customer churn predictions."
        }, {
            "role": "user",
            "content": prompt
        }])
    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
    #   prompt = f"""You are a manager at HS Bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.

    # You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

    # Here is a some explanation as to why the customer might be a risk of churning:
    # {explanation}

    # Generate an email to the customer based on their information, asking them to stay if they are at a risk of churning, or offering them incentives so that they become more loyal to the bank.

    # Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning or the machine learning model to the customer.
    # """

    prompt = f"""As a manager at HS Bank, your priority is to retain customers and offer incentives to enhance their loyalty.
  A customer named {surname} may be at risk of leaving the bank, and here are some insights into why:
  {explanation}
  Write an email to the customer, either encouraging them to stay with the bank or presenting personalized incentives that would increase their loyalty.
  Be sure to include a list of relevant offers, tailored to their specific situation, without mentioning the probability of them leaving or referencing any predictive models.
      """

    raw_response = client.chat.completions.create(
        #model="llama-3.1-8b-instant",
        model="llama-3.1-70b-versatile",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    print("\n\n EMAIL PROMPT", prompt)
    return raw_response.choices[0].message.content


st.title("Customer Churn Predictiion")

df = pd.read_csv("churn.csv")

customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:

    selected_customer_id = int(selected_customer_option.split(" - ")[0])

    print("Selected Customer ID", selected_customer_id)

    selected_surname = selected_customer_option.split(" - ")[1]
    print("Surname", selected_surname)

    selected_customer = df.loc[df['CustomerId'] ==
                               selected_customer_id].iloc[0]

    print("Selected Customer", selected_customer)

    col1, col2 = st.columns(2)

    with col1:

        credit_score = st.number_input("Credit Score",
                                       min_value=360,
                                       max_value=850,
                                       value=int(
                                           selected_customer["CreditScore"]))

        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index=["Spain", "France", "Germany"
                                       ].index(selected_customer['Geography']))

        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer['Gender'] == 'Male' else 1)

        age = st.number_input("Age",
                              min_value=18,
                              max_value=100,
                              value=int(selected_customer["Age"]))

        tenure = st.number_input("Tenure (years)",
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer["Tenure"]))

    with col2:

        balance = st.number_input("Balance",
                                  min_value=0.0,
                                  value=float(selected_customer['Balance']))

        num_products = st.number_input("Number of Products",
                                       min_value=1,
                                       max_value=10,
                                       value=int(
                                           selected_customer["NumOfProducts"]))

        has_credit_card = st.checkbox("Has Credit Card",
                                      value=bool(
                                          selected_customer["HasCrCard"]))

        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember']))

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer["EstimatedSalary"]))

    input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                         tenure, balance, num_products,
                                         has_credit_card, is_active_member,
                                         estimated_salary)

    avg_probability = make_predictions(input_df, input_dict)

    explanation = explain_prediction(avg_probability, input_dict,
                                     selected_customer["Surname"])

    st.markdown("---")

    st.subheader("Explanation of Prediction")

    st.markdown(explanation)

    email = generate_email(avg_probability, input_dict, explanation,
                           selected_customer["Surname"])

    st.markdown("---")
    st.subheader("Personalized Email")
    st.markdown(email)
