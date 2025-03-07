# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import joblib
import os

def process_data(input_file, output_dir, results_dir):
    """
    معالجة البيانات الأولية وتحضيرها للتدريب والتنبؤ.
    
    Args:
        input_file (str): مسار ملف البيانات الأصلي
        output_dir (str): مجلد حفظ المقياس والنماذج
        results_dir (str): مجلد حفظ البيانات المعالجة
    
    Returns:
        pd.DataFrame: البيانات المعالجة النهائية
    """
    # تحميل البيانات واختيار الأعمدة المهمة
    data = pd.read_csv(input_file)
    cols = ['Beds', 'Baths', 'Area_in_sqft', 'Rent_per_sqft', 'Type', 'Furnishing', 'Purpose', 'Location', 'City', 'Posted_date']
    df = data[cols].copy()

    # معالجة القيم المفقودة باستخدام KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    numeric_cols = ['Beds', 'Baths', 'Area_in_sqft', 'Rent_per_sqft']
    df.loc[:, numeric_cols] = imputer.fit_transform(df[numeric_cols])
    df[numeric_cols] = df[numeric_cols].astype(float)

    # إزالة القيم المتطرفة باستخدام IQR
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # هندسة الميزات
    df['Beds_Baths_Interaction'] = df['Beds'] * df['Baths']
    df['Rooms_per_sqft'] = (df['Beds'] + df['Baths']) / df['Area_in_sqft']
    df['Posted_date'] = pd.to_datetime(df['Posted_date'], errors='coerce')
    df['Year'] = df['Posted_date'].dt.year
    df['Month'] = df['Posted_date'].dt.month
    df['Season'] = df['Month'].map({1: 'شتاء', 2: 'شتاء', 3: 'ربيع', 4: 'ربيع', 5: 'ربيع', 
                                    6: 'صيف', 7: 'صيف', 8: 'صيف', 9: 'خريف', 10: 'خريف', 
                                    11: 'خريف', 12: 'شتاء'})
    df['Area_category'] = pd.cut(df['Area_in_sqft'], bins=[0, 500, 1000, 2000, 5000, float('inf')], 
                                 labels=['صغيرة جدًا', 'صغيرة', 'متوسطة', 'كبيرة', 'كبيرة جدًا'], include_lowest=True).cat.add_categories('N/A')

    # ترميز الفئويات
    cat_cols = ['Type', 'Furnishing', 'Purpose', 'Location', 'City', 'Season', 'Area_category']
    for col in cat_cols:
        df[col] = df[col].fillna('N/A')
    for col in ['Location', 'City']:
        df[col + '_encoded'] = df.groupby(col)['Rent_per_sqft'].transform('median')

    # حفظ البيانات الخام
    df.to_csv(os.path.join(results_dir, 'raw_processed_data.csv'), index=False)

    # تحويل الفئويات إلى دمية
    df.drop(columns=['Posted_date'], inplace=True)
    df.columns = df.columns.str.replace(' ', '_')
    df = pd.get_dummies(df, columns=cat_cols)

    # تطبيع البيانات (بدون Rent_per_sqft)
    scaler = StandardScaler()
    numeric_cols_for_scaling = ['Beds', 'Baths', 'Area_in_sqft', 'Beds_Baths_Interaction', 'Rooms_per_sqft', 'Location_encoded', 'City_encoded', 'Year', 'Month']
    df.loc[:, numeric_cols_for_scaling] = scaler.fit_transform(df[numeric_cols_for_scaling])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # حفظ البيانات المعالجة
    df.to_csv(os.path.join(results_dir, 'processed_data.csv'), index=False)

    return df

if __name__ == "__main__":
    base_path = r"D:\trans\ai\machine learning 3\projects\1\uae House Price Prediction\dubai_properties"
    input_file = os.path.join(base_path, "dubai_properties.csv")
    output_dir = os.path.join(base_path, "models")
    results_dir = os.path.join(base_path, "results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    process_data(input_file, output_dir, results_dir)