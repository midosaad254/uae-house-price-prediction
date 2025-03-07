# utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # إضافة استيراد pandas
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def plot_price_trend(data, x_col, y_col, title):
    """رسم اتجاه الأسعار باستخدام خطوط تفاعلية."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    sns.lineplot(x=x_col, y=y_col, data=data, ax=ax, color='#2ECC71', linewidth=2.5)
    ax.set_title(title, fontsize=14, color='#D4AF37')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    plt.tight_layout()
    return fig

def plot_heatmap(data, index, columns, values, title):
    """رسم خريطة حرارية لتوزيع الأسعار."""
    heatmap_data = data.pivot_table(values=values, index=index, columns=columns, aggfunc='mean').fillna(0)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.1f', ax=ax, cbar_kws={'label': values})
    ax.set_title(title, fontsize=14, color='#D4AF37')
    plt.tight_layout()
    return fig

def plot_feature_importance(model, features, title):
    """رسم أهمية الميزات."""
    importance = pd.Series(model.feature_importances_, index=features).nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    importance.plot(kind='barh', ax=ax, color='#D4AF37')
    ax.set_title(title, fontsize=14, color='#2ECC71')
    ax.set_xlabel('Importance', fontsize=12)
    plt.tight_layout()
    return fig

def export_to_pdf(result_dict, filename="prediction_result.pdf"):
    """تصدير النتائج إلى PDF."""
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    for key, value in result_dict.items():
        c.drawString(100, y, f"{key}: {value}")
        y -= 20
    c.save()
    return filename