# src/predictor.py
import pandas as pd
import joblib

def predict_price(input_dict, processed_data_path, scaler_path, model_path):
    """
    التنبؤ بسعر العقار بناءً على المدخلات.
    
    Args:
        input_dict (dict): بيانات الإدخال من المستخدم
        processed_data_path (str): مسار البيانات المعالجة
        scaler_path (str): مسار المقياس
        model_path (str): مسار النموذج
    
    Returns:
        tuple: السعر الإجمالي وسعر القدم المربع
    """
    processed_data = pd.read_csv(processed_data_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # إنشاء DataFrame للمدخلات
    input_data = pd.DataFrame([input_dict])
    input_data['Beds_Baths_Interaction'] = input_data['Beds'] * input_data['Baths']
    input_data['Rooms_per_sqft'] = (input_data['Beds'] + input_data['Baths']) / input_data['Area_in_sqft']
    input_data['Area_category'] = pd.cut(input_data['Area_in_sqft'], bins=[0, 500, 1000, 2000, 5000, float('inf')], 
                                         labels=['صغيرة جدًا', 'صغيرة', 'متوسطة', 'كبيرة', 'كبيرة جدًا'], include_lowest=True).cat.add_categories('N/A')

    # تحويل الفئويات إلى دمية
    cat_cols = ['Type', 'Furnishing', 'Purpose', 'Season', 'Area_category', 'Location', 'City']
    input_data = pd.get_dummies(input_data, columns=cat_cols)

    # محاذاة الأعمدة مع البيانات المعالجة
    for col in processed_data.columns:
        if col not in input_data.columns and col != 'Rent_per_sqft':
            input_data[col] = 0
    input_data = input_data[processed_data.drop(columns='Rent_per_sqft').columns]

    # تطبيع البيانات
    numeric_cols = ['Beds', 'Baths', 'Area_in_sqft', 'Beds_Baths_Interaction', 'Rooms_per_sqft', 'Location_encoded', 'City_encoded', 'Year', 'Month']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # التنبؤ
    predicted_rent_per_sqft = model.predict(input_data)[0]
    total_price = predicted_rent_per_sqft * input_data['Area_in_sqft'].iloc[0]

    return total_price, predicted_rent_per_sqft

# src/utils.py
import matplotlib.pyplot as plt
import seaborn as sns
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
