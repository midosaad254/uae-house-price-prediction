# src/model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import numpy as np

def train_models(df, models_dir, results_dir):
    """
    تدريب نماذج التعلم الآلي وحفظ الأفضل منها.
    
    Args:
        df (pd.DataFrame): البيانات المعالجة
        models_dir (str): مجلد حفظ النماذج
        results_dir (str): مجلد حفظ النتائج
    
    Returns:
        tuple: بيانات التدريب والاختبار والنماذج
    """
    # تقسيم البيانات
    X = df.drop(columns='Rent_per_sqft')
    y = df['Rent_per_sqft']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # تعريف النماذج
    models = {
        'RF': RandomForestRegressor(random_state=0),
        'XGBoost': xgb.XGBRegressor(random_state=0),
        'LightGBM': lgb.LGBMRegressor(random_state=0)
    }

    # تدريب النماذج الأساسية
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(models_dir, f'{name}_model.pkl'))

    # تحسين النماذج باستخدام RandomizedSearchCV
    rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
    rf_cv = RandomizedSearchCV(RandomForestRegressor(random_state=0), rf_param_grid, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=0)
    rf_cv.fit(X_train, y_train)

    xgb_param_grid = {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 200, 300]}
    xgb_cv = RandomizedSearchCV(xgb.XGBRegressor(random_state=0), xgb_param_grid, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=0)
    xgb_cv.fit(X_train, y_train)

    lgbm_param_grid = {'num_leaves': [31, 50, 70], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 200, 300]}
    lgbm_cv = RandomizedSearchCV(lgb.LGBMRegressor(random_state=0), lgbm_param_grid, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=0)
    lgbm_cv.fit(X_train, y_train)

    models['RF_optimized'] = rf_cv.best_estimator_
    models['XGBoost_optimized'] = xgb_cv.best_estimator_
    models['LightGBM_optimized'] = lgbm_cv.best_estimator_

    # حفظ النماذج المحسنة
    for name in ['RF_optimized', 'XGBoost_optimized', 'LightGBM_optimized']:
        joblib.dump(models[name], os.path.join(models_dir, f'{name}_model.pkl'))

    # تقييم النماذج
    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics.append([r2_score(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred)])
    metrics_df = pd.DataFrame(metrics, columns=['R2', 'RMSE', 'MAE'], index=models.keys()).round(3)
    metrics_df.to_csv(os.path.join(results_dir, 'model_metrics_optimized.csv'))

    # اختيار أفضل نموذج
    best_model_name = metrics_df['R2'].idxmax()
    best_model = models[best_model_name]
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    joblib.dump(metrics_df.loc[best_model_name, 'MAE'], os.path.join(models_dir, 'best_model_mae.pkl'))

    return X_train, X_test, y_train, y_test, models

if __name__ == "__main__":
    base_path = r"D:\trans\ai\machine learning 3\projects\1\uae House Price Prediction\dubai_properties"
    results_dir = os.path.join(base_path, "results")
    models_dir = os.path.join(base_path, "models")
    df = pd.read_csv(os.path.join(results_dir, 'processed_data.csv'))
    train_models(df, models_dir, results_dir)
    