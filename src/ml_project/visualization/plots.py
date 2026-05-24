from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_fraud_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="isFraud", ax=ax)
    ax.set_title("Distribution de la cible fraude")
    ax.set_xlabel("isFraud")
    ax.set_ylabel("Nombre de transactions")
    return fig


def plot_transaction_amounts(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="isFraud", y="amount", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Montants des transactions par classe")
    return fig


def plot_customer_segments(df: pd.DataFrame, x: str = "Total_Spend", y: str = "Income"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, hue="segment", palette="tab10", ax=ax)
    ax.set_title("Segments clients")
    return fig
