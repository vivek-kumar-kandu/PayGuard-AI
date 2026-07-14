import pandas as pd
import numpy as np

df=pd.read_csv('dataset/cleaned_onlinefraud.csv')

# print(df.shape)
# print(df.columns)
# print(df.head())
# print(df.dtypes)


# Target Variable ( isFraud )
# print(df["isFraud"].value_counts())  

''' 0 => Legitimate (Genuine) Transaction which is 6354407  => 99.87%
    1=> Fraudulent Transaction which is 8213  => 0.13%
Observation:
    The dataset is highly imbalanced because genuine transactions
    significantly outnumber than fraudulent transactions.
'''

# Features Variables ( Types  of Transactions(type) )

# print(df['type'].value_counts())

"""
Observation:

The dataset contains five types of financial transactions:

• CASH_OUT  : 2,237,500 transactions (35.17%)
• PAYMENT   : 2,151,495 transactions (33.81%)
• CASH_IN   : 1,399,284 transactions (21.99%)
• TRANSFER  :   532,909 transactions (8.38%)
• DEBIT     :    41,432 transactions (0.65%)

Insights:
- CASH_OUT is the most frequent transaction type (35.17%).
- PAYMENT is the second most common transaction type (33.81%).
- CASH_IN accounts for approximately 22% of all transactions.
- TRANSFER transactions represent 8.38% of the dataset.
- DEBIT is the least frequent transaction type (0.65%).
- Most transactions belong to CASH_OUT and PAYMENT, which together account for nearly 69% of the dataset.
"""

# Features Variables ( Transaction Amounts (amount))
# print(df['amount'].value_counts())
# print(df["amount"].describe())
# print("Minimum Amount:", df["amount"].min())

"""
Observation:

Transaction Amount Analysis:

• Total Transactions : 6,362,620
• Average Transaction Amount : 179,861.90
• Minimum Transaction Amount : 0.00
• Maximum Transaction Amount : 92,445,516.64
• Median Transaction Amount : 74,871.94

Insights:
- Transaction amounts vary significantly across the dataset.
- The large difference between the mean and maximum indicates the presence of high-value transactions (outliers).
- 50% of transactions have an amount less than or equal to 74,871.94.
- The dataset contains transactions ranging from 0 to over 92 million, showing high variability.
"""


# Balance Analysis

# print(df["amount"].describe())
# print(df[[
#     "oldbalanceOrg",
#     "newbalanceOrig",
#     "oldbalanceDest",
#     "newbalanceDest"
# ]].describe())


"""
Observation:

Sender Balance Before Transaction (oldbalanceOrg)
• Mean Balance        : 833,883.10
• Median Balance      : 14,208.00
• Minimum Balance     : 0.00
• 25th Percentile     : 0.00
• 75th Percentile     : 107,315.20
• Maximum Balance     : 59,585,040.00
• Standard Deviation  : 2,888,243.00

Sender Balance After Transaction (newbalanceOrig)
• Mean Balance        : 855,113.70
• Median Balance      : 0.00
• Minimum Balance     : 0.00
• 25th Percentile     : 0.00
• 75th Percentile     : 144,258.40
• Maximum Balance     : 49,585,040.00
• Standard Deviation  : 2,924,049.00

Receiver Balance Before Transaction (oldbalanceDest)
• Mean Balance        : 1,100,702.21
• Median Balance      : 132,705.70
• Minimum Balance     : 0.00
• 25th Percentile     : 0.00
• 75th Percentile     : 943,036.70
• Maximum Balance     : 356,015,900.00
• Standard Deviation  : 3,399,180.00

Receiver Balance After Transaction (newbalanceDest)
• Mean Balance        : 1,224,996.40
• Median Balance      : 214,661.40
• Minimum Balance     : 0.00
• 25th Percentile     : 0.00
• 75th Percentile     : 1,111,909.00
• Maximum Balance     : 356,179,300.00
• Standard Deviation  : 3,674,129.00

Observations:
• Receiver accounts generally maintain higher balances than sender accounts.
• Many accounts have zero balance before or after transactions, indicating inactive or newly created accounts.
• The large difference between the mean and median across all balance features indicates highly skewed distributions.
• The high standard deviation and maximum balance values suggest the presence of significant outliers and high-value transactions.
• These balance-related features are expected to play an important role in distinguishing fraudulent and legitimate transactions during model training.
"""


#  Corelation Analysis

correlation = df.corr(numeric_only=True)
print(correlation)

"""
Observation:

Correlation Analysis

• oldbalanceOrg and newbalanceOrig have a very strong positive correlation (0.9988), indicating that the sender's balance before and after a transaction is highly related.

• oldbalanceDest and newbalanceDest also show a very strong positive correlation (0.9766), suggesting that the receiver's balance changes consistently after transactions.

• amount has a moderate positive correlation with newbalanceDest (0.4593) and oldbalanceDest (0.2941), indicating that larger transactions generally result in higher receiver account balances.

• isFraud has a weak positive correlation with amount (0.0767) and step (0.0316), while its correlation with other numerical features is close to zero.

Insights:
• Most numerical features have weak linear relationships with the target variable (isFraud).
• Fraud detection is unlikely to rely on a single feature; instead, it depends on the combined patterns of multiple features.
• The strong correlations between sender and receiver balance features indicate consistent balance updates during transactions.
• Advanced Machine Learning models such as Random Forest and XGBoost are better suited for capturing these complex relationships than simple linear models.
"""

# Generating Report in the form of Chart by Given DataFrame

import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)

fraud_counts = df["isFraud"].value_counts()

plt.figure(figsize=(6,5))
plt.bar(["Legitimate", "Fraud"], fraud_counts.values)
plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.savefig("reports/fraud_distribution.png", dpi=300)
plt.show()

"""
Observation:

Fraud Distribution Chart

• Legitimate Transactions : 6,354,407 (99.87%)
• Fraud Transactions      : 8,213 (0.13%)

Insights:
• The dataset is extremely imbalanced.
• Genuine transactions dominate the dataset.
• Machine learning models may become biased toward the majority class.
• Class balancing techniques such as SMOTE are required before training.
"""

transaction_counts = df["type"].value_counts()
plt.figure(figsize=(8,5))
plt.bar(transaction_counts.index, transaction_counts.values)
plt.title("Transaction Type Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.savefig("reports/transaction_type_distribution.png", dpi=300)
plt.show()

"""
Observation:

Transaction Type Distribution

• CASH_OUT is the most common transaction type.
• PAYMENT is the second most frequent transaction.
• CASH_IN contributes around one-fifth of all transactions.
• TRANSFER transactions are comparatively fewer.
• DEBIT transactions are the least common.

Insights:
• Nearly 69% of all transactions belong to CASH_OUT and PAYMENT.
• Understanding transaction frequency helps identify high-risk transaction categories.
"""


fraud_by_type = df.groupby("type")["isFraud"].sum()
plt.figure(figsize=(8,5))
plt.bar(fraud_by_type.index, fraud_by_type.values)
plt.title("Fraudulent Transactions by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Count")
plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.savefig("reports/fraud_by_transaction_type.png", dpi=300)
plt.show()

"""
Observation:

Fraud by Transaction Type

• Fraudulent transactions occur mainly in:
    - CASH_OUT
    - TRANSFER

• No fraud cases were observed in:
    - CASH_IN
    - PAYMENT
    - DEBIT

Insights:
• Fraudsters mainly exploit money withdrawal and money transfer operations.
• CASH_OUT and TRANSFER become the most important transaction types for fraud detection.
• Transaction type is expected to be a highly informative feature during model training.
"""


plt.figure(figsize=(8,5))
plt.hist(df["amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.grid(alpha=0.4)
plt.savefig("reports/amount_distribution.png", dpi=300)
plt.show()

"""
Observation:

Transaction Amount Distribution

• Most transactions involve relatively small amounts.
• Only a small number of transactions have very high values.
• The distribution is highly right-skewed.

Insights:
• The presence of high-value transactions creates significant outliers.
• Large transaction amounts alone do not indicate fraud.
• Machine learning models should learn patterns from multiple features rather than relying only on transaction amount.
"""

import seaborn as sns

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.savefig("reports/correlation_heatmap.png", dpi=300)
plt.show()

"""
Observation:

Correlation Heatmap

• Sender balances before and after transactions are highly correlated.
• Receiver balances before and after transactions are also strongly correlated.
• Transaction amount has moderate correlation with receiver balances.
• Fraud has only weak correlation with individual numerical features.

Insights:
• Fraud detection depends on multiple feature interactions.
• Tree-based algorithms are more suitable than linear models for capturing these complex relationships.
"""

plt.figure(figsize=(8,5))

plt.boxplot(df["amount"], orientation="horizontal", showfliers=False)

plt.title("Boxplot of Transaction Amount")
plt.xlabel("Amount")
plt.savefig("reports/amount_boxplot.png", dpi=300)
plt.show()

"""
Observation:

Transaction Amount Boxplot

• The median transaction amount is much lower than the maximum transaction amount.
• Most transactions lie within a relatively small range.
• Several extremely high-value transactions exist.

Insights:
• The transaction amount contains significant outliers.
• High-value transactions should not be removed because they may represent genuine or fraudulent financial activities.
• Decision Tree and Random Forest models are naturally robust to outliers.
"""

plt.figure(figsize=(10,6))

plt.boxplot(
    [
        df["oldbalanceOrg"],
        df["newbalanceOrig"],
        df["oldbalanceDest"],
        df["newbalanceDest"]
    ],
    tick_labels=[
        "Old Sender",
        "New Sender",
        "Old Receiver",
        "New Receiver"
    ]
)

plt.title("Balance Distribution")
plt.ylabel("Balance")
plt.xticks(rotation=20)
plt.savefig("reports/balance_boxplot.png", dpi=300)
plt.show()

"""
Observation:

Balance Distribution

• Receiver account balances are generally higher than sender balances.
• All balance-related features contain numerous extreme values.
• Large variability exists across account balances.

Insights:
• Balance features exhibit strong skewness and many outliers.
• These balance variables are valuable indicators for fraud detection.
• Machine learning algorithms can utilize these balance patterns to distinguish fraudulent and legitimate transactions.
"""