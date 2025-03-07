# uae_house_price_prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import sys
import arabic_reshaper
from bidi.algorithm import get_display

sys.stdout.reconfigure(encoding='utf-8')

# تحديد المسار
base_path = r"D:\trans\ai\machine learning 3\projects\1\uae House Price Prediction\dubai_properties"
file_path = os.path.join(base_path, "dubai_properties.csv")
output_dir = os.path.join(base_path, "plots")
results_dir = os.path.join(base_path, "results")
models_dir = os.path.join(base_path, "models")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# تحميل البيانات
data = pd.read_csv(file_path)
print(data.shape)

# اختيار الأعمدة
cols = ['Beds', 'Baths', 'Area_in_sqft', 'Rent_per_sqft', 'Type', 'Furnishing', 'Purpose', 'Location', 'City', 'Posted_date']
df = data[cols].copy()

# --- تحسين جودة البيانات ---
imputer = KNNImputer(n_neighbors=5)
numeric_cols = ['Beds', 'Baths', 'Area_in_sqft', 'Rent_per_sqft']
df.loc[:, numeric_cols] = imputer.fit_transform(df[numeric_cols])
df[numeric_cols] = df[numeric_cols].astype(float)

# معالجة القيم المتطرفة باستخدام IQR
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
print(f"حجم البيانات بعد معالجة القيم المتطرفة: {df.shape}")

# تسريع: أخذ عينة عشوائية (10,000 سطر)
df = df.sample(n=10000, random_state=0)
print(f"حجم العينة العشوائية: {df.shape}")

# --- التحليل الاستكشافي ---
print(f"نسبة الصفوف التي تحتوي على قيم مفقودة: {(df.isna().sum(axis=1) > 0).mean() * 100:.2f}%")
print(df.isna().sum() / df.shape[0] * 100)
print(df.dtypes)

# رسومات مع تشكيل النصوص العربية
plt.rcParams['font.family'] = 'Arial'
reshaped_title = get_display(arabic_reshaper.reshape('توزيع الإيجار لكل قدم مربع'))
df['Rent_per_sqft'].plot(kind='box', vert=False)
plt.title(reshaped_title)
plt.savefig(os.path.join(output_dir, 'rent_per_sqft_boxplot.png'))
plt.close()

# رسم إضافي: توزيع الإيجار حسب المدينة
plt.figure(figsize=(10, 6))
reshaped_title = get_display(arabic_reshaper.reshape('توزيع الإيجار حسب المدينة'))
sns.boxplot(x='City', y='Rent_per_sqft', data=df)
plt.title(reshaped_title)
plt.xticks(rotation=45)
plt.xlabel(get_display(arabic_reshaper.reshape('المدينة')))
plt.ylabel(get_display(arabic_reshaper.reshape('الإيجار لكل قدم مربع')))
plt.savefig(os.path.join(output_dir, 'rent_per_city_boxplot.png'))
plt.close()

# رسم إضافي: الاتجاه الزمني
df['Posted_date'] = pd.to_datetime(df['Posted_date'], errors='coerce')
df['Month'] = df['Posted_date'].dt.month
plt.figure(figsize=(10, 6))
reshaped_title = get_display(arabic_reshaper.reshape('اتجاه الإيجار حسب الشهر'))
sns.lineplot(x='Month', y='Rent_per_sqft', data=df)
plt.title(reshaped_title)
plt.xlabel(get_display(arabic_reshaper.reshape('الشهر')))
plt.ylabel(get_display(arabic_reshaper.reshape('الإيجار لكل قدم مربع')))
plt.savefig(os.path.join(output_dir, 'rent_per_month_lineplot.png'))
plt.close()

# --- هندسة الميزات ---
df['Beds_Baths_Interaction'] = df['Beds'] * df['Baths']
df['Rooms_per_sqft'] = (df['Beds'] + df['Baths']) / df['Area_in_sqft']
df['Season'] = df['Month'].map({1: 'شتاء', 2: 'شتاء', 3: 'ربيع', 4: 'ربيع', 5: 'ربيع', 
                                6: 'صيف', 7: 'صيف', 8: 'صيف', 9: 'خريف', 10: 'خريف', 
                                11: 'خريف', 12: 'شتاء'})
df['Area_category'] = pd.cut(df['Area_in_sqft'], bins=[0, 1000, 2000, float('inf')], 
                             labels=['صغيرة', 'متوسطة', 'كبيرة'], include_lowest=True).cat.add_categories('N/A')

cat_cols = ['Type', 'Furnishing', 'Purpose', 'Location', 'City', 'Season', 'Area_category']
for col in cat_cols:
    df.loc[:, col] = df[col].fillna('N/A')

for col in ['Location', 'City']:
    means = df.groupby(col)['Rent_per_sqft'].mean()
    df.loc[:, col + '_encoded'] = df[col].map(means)
df.drop(columns=['Location', 'City', 'Posted_date', 'Month'], inplace=True)

df.columns = df.columns.str.replace(' ', '_')
df = pd.get_dummies(df, columns=[c for c in cat_cols if c not in ['Location', 'City']])

# تطبيع البيانات وحفظ المقياس
scaler = StandardScaler()
numeric_cols_extended = numeric_cols + ['Beds_Baths_Interaction', 'Rooms_per_sqft']
df.loc[:, numeric_cols_extended] = scaler.fit_transform(df[numeric_cols_extended])
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))  # حفظ المقياس للتنبؤ

X = df.drop(columns='Rent_per_sqft')
y = df['Rent_per_sqft']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, X_test.shape)

# --- تدريب النماذج ---
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=0, min_samples_leaf=5)
rf = RandomForestRegressor(random_state=0)
xgb_model = xgb.XGBRegressor(random_state=0)
lasso = Lasso(alpha=0.1, random_state=0)
lgbm = lgb.LGBMRegressor(random_state=0)

models = {'LR': lr, 'DT': dt, 'RF': rf, 'XGBoost': xgb_model, 'Lasso': lasso, 'LightGBM': lgbm}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(models_dir, f'{name}_model.pkl'))

def evaluate_models(models):
    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        metrics.append([r2, rmse, mae])
    return pd.DataFrame(metrics, columns=['R2', 'RMSE', 'MAE'], index=models.keys()).round(3)

metrics_df = evaluate_models(models)
metrics_df.to_csv(os.path.join(results_dir, 'model_metrics.csv'))
print(metrics_df)

# --- تحسين النماذج ---
rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
rf_cv = RandomizedSearchCV(RandomForestRegressor(random_state=0), rf_param_grid, n_iter=5, cv=3, scoring='r2', n_jobs=-1, random_state=0)
rf_cv.fit(X_train, y_train)
print("أفضل معلمات RF:", rf_cv.best_params_)

xgb_param_grid = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
xgb_cv = RandomizedSearchCV(xgb.XGBRegressor(random_state=0), xgb_param_grid, n_iter=5, cv=3, scoring='r2', n_jobs=-1, random_state=0)
xgb_cv.fit(X_train, y_train)
print("أفضل معلمات XGBoost:", xgb_cv.best_params_)

lgbm_param_grid = {'num_leaves': [31, 50], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
lgbm_cv = RandomizedSearchCV(lgb.LGBMRegressor(random_state=0), lgbm_param_grid, n_iter=5, cv=3, scoring='r2', n_jobs=-1, random_state=0)
lgbm_cv.fit(X_train, y_train)
print("أفضل معلمات LightGBM:", lgbm_cv.best_params_)

models['RF_محسن'] = rf_cv.best_estimator_
models['XGBoost_محسن'] = xgb_cv.best_estimator_
models['LightGBM_محسن'] = lgbm_cv.best_estimator_

for name in ['RF_محسن', 'XGBoost_محسن', 'LightGBM_محسن']:
    joblib.dump(models[name], os.path.join(models_dir, f'{name}_model.pkl'))

metrics_df = evaluate_models(models)
metrics_df.to_csv(os.path.join(results_dir, 'model_metrics_optimized.csv'))
print(metrics_df)

# اختيار أفضل نموذج بناءً على R2 وحفظه للتنبؤ
best_model_name = metrics_df['R2'].idxmax()
best_model = models[best_model_name]
joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
print(f"أفضل نموذج: {best_model_name}")

def test_overfitting(models):
    r2_scores = []
    for name, model in models.items():
        r2_train = r2_score(y_train, model.predict(X_train))
        r2_test = r2_score(y_test, model.predict(X_test))
        r2_scores.append([r2_train, r2_test])
    df_r2 = pd.DataFrame(r2_scores, columns=['R2_التدريب', 'R2_الاختبار'], index=models.keys()).round(3)
    df_r2.to_csv(os.path.join(results_dir, 'train_test_r2.csv'))
    reshaped_title = get_display(arabic_reshaper.reshape('مقارنة R2 للتدريب والاختبار'))
    df_r2.plot(kind='bar', title=reshaped_title, color=['#66c2a5', '#fc8d62'])
    plt.legend(loc='center')
    plt.savefig(os.path.join(output_dir, 'train_test_r2_bar.png'))
    plt.close()
    return df_r2

df_r2 = test_overfitting(models)
print(df_r2)

def plot_importances(model, features, name):
    importances = pd.Series(model.feature_importances_, index=features).nlargest(10)
    importances.to_csv(os.path.join(results_dir, f'feature_importance_{name}.csv'))
    reshaped_title = get_display(arabic_reshaper.reshape(f'{name}: أهم 10 ميزات'))
    importances.sort_values().plot(kind='barh', color='#8da0cb')
    plt.title(reshaped_title)
    plt.xlabel(get_display(arabic_reshaper.reshape('الأهمية')))
    plt.savefig(os.path.join(output_dir, f'feature_importance_{name}.png'))
    plt.close()

plot_importances(dt, X.columns, 'شجرة_القرار')
plot_importances(rf_cv.best_estimator_, X.columns, 'الغابة_العشوائية_محسن')
plot_importances(xgb_cv.best_estimator_, X.columns, 'XGBoost_محسن')
plot_importances(lgbm_cv.best_estimator_, X.columns, 'LightGBM_محسن')

# حفظ البيانات المُعالجة للاستخدام في التنبؤ
df.to_csv(os.path.join(results_dir, 'processed_data.csv'), index=False)