# dash_app.py
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from predictor import predict_price
import joblib
import os
from functools import lru_cache

# تحديد المسارات
base_path = r"D:\trans\ai\machine learning 3\projects\1\uae House Price Prediction\dubai_properties"
raw_data = pd.read_csv(os.path.join(base_path, "results", "raw_processed_data.csv")).copy()
processed_data = pd.read_csv(os.path.join(base_path, "results", "processed_data.csv")).copy()
scaler_path = os.path.join(base_path, "models", "scaler.pkl")
model_path = os.path.join(base_path, "models", "best_model.pkl")
mae = joblib.load(os.path.join(base_path, "models", "best_model_mae.pkl"))

# التحقق من وجود 'Rent_per_sqft' و'Year'
if 'Rent_per_sqft' not in raw_data.columns:
    raise ValueError("'Rent_per_sqft' غير موجود في raw_data")
if 'Rent_per_sqft' not in processed_data.columns:
    raise ValueError("'Rent_per_sqft' غير موجود في processed_data")
if 'Year' not in raw_data.columns:
    raw_data['Year'] = 2025
if 'Year' not in processed_data.columns:
    processed_data['Year'] = 2025

# تعريف البيانات الجغرافية
locations = raw_data['Location'].unique()[:4]
uae_map_data = pd.DataFrame({
    'Location': locations,
    'Rent_per_sqft': raw_data[raw_data['Location'].isin(locations)].groupby('Location')['Rent_per_sqft'].mean().reindex(locations),
    'lat': [25.276987, 25.204849, 25.077250, 25.197197],
    'lon': [55.296249, 55.270783, 55.301500, 55.171280]
}).dropna()

# إعداد تطبيق Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css",
        "https://fonts.googleapis.com/css2?family=Dubai:wght@300;400;700&display=swap"
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# النصوص بلغتين
texts = {
    'title': {'ar': '🏠 تحليل العقارات الذكي', 'en': '🏠 Smart Property Analytics'},
    'prediction': {'ar': 'توقع السعر', 'en': 'Price Prediction'},
    'city': {'ar': 'المدينة', 'en': 'City'},
    'location': {'ar': 'الموقع', 'en': 'Location'},
    'type': {'ar': 'نوع العقار', 'en': 'Property Type'},
    'beds': {'ar': 'غرف النوم', 'en': 'Beds'},
    'baths': {'ar': 'الحمامات', 'en': 'Baths'},
    'area': {'ar': 'المساحة (قدم مربع)', 'en': 'Area (sqft)'},
    'year': {'ar': 'السنة (YYYY)', 'en': 'Year (YYYY)'},
    'predict': {'ar': 'توقع', 'en': 'Predict'},
    'price': {'ar': 'السعر المتوقع', 'en': 'Predicted Price'},
    'range': {'ar': 'النطاق', 'en': 'Range'},
    'trend_title': {'ar': 'اتجاه الأسعار', 'en': 'Price Trend'},
    'heatmap_title': {'ar': 'توزيع الأسعار', 'en': 'Price Distribution'},
    'features_title': {'ar': 'أهم العوامل', 'en': 'Key Factors'},
    'seasonal_title': {'ar': 'التغيرات الموسمية', 'en': 'Seasonal Trends'},
    'roi_title': {'ar': 'العائد على الاستثمار', 'en': 'Return on Investment'}
}

# الواجهة الاحترافية
app.layout = html.Div(
    className="dashboard",
    style={'fontFamily': 'Dubai, Arial', 'backgroundColor': '#F8F9FA', 'minHeight': '100vh'},
    children=[
        html.Div(
            className="navbar shadow-sm",
            style={'backgroundColor': '#FFFFFF', 'padding': '15px'},
            children=[
                html.Div(
                    className="container d-flex justify-content-between align-items-center",
                    children=[
                        html.Div(
                            className="logo d-flex align-items-center",
                            children=[
                                html.Img(src="/assets/logo.png", height="40px"),
                                html.H4(id='title', style={'marginLeft': '15px', 'color': '#006633'})
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Button("🇦🇪", id="lang-ar", className="btn btn-sm btn-outline-dark mx-1"),
                                html.Button("🇬🇧", id="lang-en", className="btn btn-sm btn-outline-dark mx-1"),
                                html.Button("🌙/☀️", id="theme-toggle", className="btn btn-sm btn-outline-dark mx-1")
                            ]
                        )
                    ]
                )
            ]
        ),
        html.Div(
            className="container py-4",
            children=[
                html.Div(className="row", children=[
                    html.Div(className="col-md-4", children=[
                        html.Div(className="card shadow-sm", id='prediction-card', children=[
                            html.H3(id='prediction-title', className="text-center", style={'color': '#006633'}),
                            dcc.Dropdown(id='city', options=[{'label': c, 'value': c} for c in raw_data['City'].unique()], value=raw_data['City'].unique()[0]),
                            dcc.Dropdown(id='location', options=[{'label': l, 'value': l} for l in raw_data['Location'].unique()], value=raw_data['Location'].unique()[0]),
                            dcc.Dropdown(id='type', options=[{'label': t, 'value': t} for t in raw_data['Type'].unique()], value=raw_data['Type'].unique()[0]),
                            dcc.Input(id='beds', type='number', placeholder='', value=2, min=1),
                            dcc.Input(id='baths', type='number', placeholder='', value=2, min=1),
                            dcc.Input(id='area', type='number', placeholder='', value=1000, min=100),
                            dcc.Input(id='year', type='number', placeholder='YYYY', value=2025, min=2000, max=2050),
                            html.Button(id='predict-btn', n_clicks=0, className="btn btn-success w-100", style={'backgroundColor': '#006633'}),
                            html.Div(id='prediction-output', className="mt-3 text-center")
                        ])
                    ]),
                    html.Div(className="col-md-8", children=[
                        html.Div(className="card shadow-sm mb-4", children=[dcc.Graph(id='price-trend'), html.P(id='trend-desc')]),
                        html.Div(className="card shadow-sm mb-4", children=[dcc.Graph(id='heatmap'), html.P(id='heatmap-desc')]),
                        html.Div(className="card shadow-sm mb-4", children=[dcc.Graph(id='feature-importance'), html.P(id='features-desc')]),
                        html.Div(className="card shadow-sm mb-4", children=[dcc.Graph(id='seasonal-trend'), html.P(id='seasonal-desc')]),
                        html.Div(className="card shadow-sm mb-4", children=[dcc.Graph(id='roi-analysis'), html.P(id='roi-desc')])
                    ])
                ])
            ]
        ),
        dcc.Store(id='lang-store', storage_type='local'),
        dcc.Store(id='theme-store', storage_type='local'),
        dcc.Store(id='data-store', storage_type='memory'),
        dcc.Interval(id='interval-component', interval=3600*1000, n_intervals=0)
    ]
)

# تعريف الدوال
def validate_input(input_dict):
    errors = []
    if input_dict['Beds'] <= 0 or input_dict['Beds'] > 20:
        errors.append("عدد غرف النوم يجب أن يكون بين 1 و20")
    if input_dict['Area_in_sqft'] <= 0 or input_dict['Area_in_sqft'] > 50000:
        errors.append("المساحة يجب أن تكون بين 1 و50000 قدم مربع")
    if input_dict['Location'] not in raw_data['Location'].unique():
        errors.append("الموقع غير موجود في قاعدة البيانات")
    if input_dict['Year'] < 2000 or input_dict['Year'] > 2050:
        errors.append("السنة يجب أن تكون بين 2000 و2050")
    return errors

def ensemble_prediction(input_dict):
    models = {'rf': joblib.load(model_path)}
    predictions = {}
    try:
        for name, model in models.items():
            predictions[name] = predict_price(input_dict, os.path.join(base_path, "results", "processed_data.csv"), 
                                            scaler_path, model_path if name == 'rf' else os.path.join(base_path, "models", f"{name}_model.pkl"))[0]
        weights = {'rf': 0.5}
        weighted_pred = sum(predictions[m] * weights.get(m, 1/len(models)) for m in predictions)
        return weighted_pred, predictions
    except Exception as e:
        raise Exception(f"Error in ensemble prediction: {str(e)}")

@lru_cache(maxsize=32)
def create_price_trend(data_tuple, lang):
    data = pd.DataFrame(data_tuple, columns=raw_data.columns)  # التأكد من مطابقة الأعمدة
    if 'Rent_per_sqft' not in data.columns:
        raise ValueError("'Rent_per_sqft' غير موجود في البيانات الممررة إلى create_price_trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Year'], y=data['Rent_per_sqft'],
        mode='lines+markers',
        line=dict(color='#006633', width=2.5),
        marker=dict(size=10, color='#D4AF37', line=dict(width=2, color='#FFFFFF')),
        hovertemplate='السنة: %{x}<br>الإيجار: %{y:.2f} درهم<extra></extra>' if lang == 'ar' else 'Year: %{x}<br>Rent: %{y:.2f} AED<extra></extra>'
    ))
    fig.update_layout(
        title={'text': texts['trend_title'][lang], 'font': {'size': 20, 'color': '#006633'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='السنة' if lang == 'ar' else 'Year',
        yaxis_title='الإيجار (درهم)' if lang == 'ar' else 'Rent (AED)',
        template='plotly_white',
        xaxis=dict(gridcolor='#E8ECEF', gridwidth=1, showgrid=True),
        yaxis=dict(gridcolor='#E8ECEF', gridwidth=1, showgrid=True),
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='#F8F9FA'
    )
    return fig

@lru_cache(maxsize=32)
def create_advanced_heatmap(data_tuple, lang):
    data = pd.DataFrame(data_tuple, columns=uae_map_data.columns)  # التأكد من مطابقة الأعمدة
    if 'Rent_per_sqft' not in data.columns:
        raise ValueError("'Rent_per_sqft' غير موجود في البيانات الممررة إلى create_advanced_heatmap")
    fig = go.Figure()
    fig.add_trace(go.Densitymapbox(
        lat=data['lat'], lon=data['lon'], z=data['Rent_per_sqft'],
        radius=15, colorscale='YlOrRd', opacity=0.8,
        colorbar_title='الإيجار (درهم)' if lang == 'ar' else 'Rent (AED)',
        hovertemplate='<b>%{text}</b><br>الإيجار: %{z:.2f} درهم<extra></extra>' if lang == 'ar' else '<b>%{text}</b><br>Rent: %{z:.2f} AED<extra></extra>',
        text=data['Location']
    ))
    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=9, mapbox_center={"lat": 25.2, "lon": 55.3},
        title={'text': texts['heatmap_title'][lang], 'font': {'size': 20, 'color': '#006633'}, 'x': 0.5, 'xanchor': 'center'},
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor='#F8F9FA'
    )
    return fig

@lru_cache(maxsize=32)
def create_feature_importance(lang):
    model = joblib.load(model_path)
    importance = pd.Series(model.feature_importances_, index=processed_data.drop(columns='Rent_per_sqft').columns).nlargest(10)
    fig = px.bar(
        importance, orientation='h',
        color=importance.index, color_discrete_sequence=px.colors.sequential.YlOrRd,
        text=importance.apply(lambda x: f'{x:.2f}')
    )
    fig.update_traces(textposition='auto')
    fig.update_layout(
        title={'text': texts['features_title'][lang], 'font': {'size': 20, 'color': '#006633'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='الأهمية' if lang == 'ar' else 'Importance',
        yaxis_title='العوامل' if lang == 'ar' else 'Features',
        template='plotly_white',
        showlegend=False,
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='#F8F9FA'
    )
    return fig

@lru_cache(maxsize=32)
def seasonal_analysis(data_tuple, lang):
    data = pd.DataFrame(data_tuple, columns=raw_data.columns)  # التأكد من مطابقة الأعمدة
    if 'Rent_per_sqft' not in data.columns:
        raise ValueError("'Rent_per_sqft' غير موجود في البيانات الممررة إلى seasonal_analysis")
    monthly_data = data.groupby(['Month'])['Rent_per_sqft'].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_data['Month'], y=monthly_data['Rent_per_sqft'],
        mode='lines+markers',
        line=dict(color='#D4AF37', width=2.5),
        marker=dict(size=10, color='#006633', line=dict(width=2, color='#FFFFFF')),
        hovertemplate='الشهر: %{x}<br>الإيجار: %{y:.2f} درهم<extra></extra>' if lang == 'ar' else 'Month: %{x}<br>Rent: %{y:.2f} AED<extra></extra>'
    ))
    fig.update_layout(
        title={'text': texts['seasonal_title'][lang], 'font': {'size': 20, 'color': '#006633'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='الشهر' if lang == 'ar' else 'Month',
        yaxis_title='الإيجار (درهم)' if lang == 'ar' else 'Rent (AED)',
        template='plotly_white',
        xaxis=dict(gridcolor='#E8ECEF', gridwidth=1, showgrid=True),
        yaxis=dict(gridcolor='#E8ECEF', gridwidth=1, showgrid=True),
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='#F8F9FA'
    )
    return fig

@lru_cache(maxsize=32)
def investment_roi_analysis(data_tuple, lang):
    data = pd.DataFrame(data_tuple, columns=raw_data.columns)  # التأكد من مطابقة الأعمدة
    if 'Rent_per_sqft' not in data.columns:
        raise ValueError("'Rent_per_sqft' غير موجود في البيانات الممررة إلى investment_roi_analysis")
    data['purchase_price'] = data['Rent_per_sqft'] * data['Area_in_sqft'] * 12 * 15
    roi_rows = []
    for location in data['Location'].unique()[:5]:
        loc_data = data[data['Location'] == location]
        avg_price = loc_data['purchase_price'].mean()
        avg_rent = loc_data['Rent_per_sqft'].mean() * loc_data['Area_in_sqft'].mean() * 12
        for year in range(1, 6):
            yearly_rent = avg_rent * (1.03 ** (year-1))
            cumulative_income = sum(avg_rent * (1.03 ** i) for i in range(year))
            roi = (cumulative_income / avg_price) * 100
            roi_rows.append({'Location': location, 'Year': year + 2024, 'ROI': roi})
    roi_data = pd.DataFrame(roi_rows)
    fig = px.line(
        roi_data, x='Year', y='ROI', color='Location',
        line_shape='spline', color_discrete_sequence=px.colors.sequential.YlOrRd,
        hover_data={'ROI': ':.2f'}
    )
    fig.update_traces(line_width=2.5)
    fig.update_layout(
        title={'text': texts['roi_title'][lang], 'font': {'size': 20, 'color': '#006633'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='السنة' if lang == 'ar' else 'Year',
        yaxis_title='ROI (%)',
        template='plotly_white',
        xaxis=dict(gridcolor='#E8ECEF', gridwidth=1, showgrid=True),
        yaxis=dict(gridcolor='#E8ECEF', gridwidth=1, showgrid=True),
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='#F8F9FA',
        legend_title_text='الموقع' if lang == 'ar' else 'Location'
    )
    return fig

# الـ Callback الرئيسي
@app.callback(
    [Output('title', 'children'), Output('prediction-title', 'children'),
     Output('beds', 'placeholder'), Output('baths', 'placeholder'), Output('area', 'placeholder'),
     Output('year', 'placeholder'), Output('predict-btn', 'children'), Output('prediction-output', 'children'),
     Output('price-trend', 'figure'), Output('heatmap', 'figure'), Output('feature-importance', 'figure'),
     Output('seasonal-trend', 'figure'), Output('roi-analysis', 'figure'),
     Output('trend-desc', 'children'), Output('heatmap-desc', 'children'), Output('features-desc', 'children'),
     Output('seasonal-desc', 'children'), Output('roi-desc', 'children'), Output('lang-store', 'data')],
    [Input('lang-ar', 'n_clicks'), Input('lang-en', 'n_clicks'), Input('predict-btn', 'n_clicks'), Input('theme-toggle', 'n_clicks')],
    [State('city', 'value'), State('location', 'value'), State('type', 'value'), State('beds', 'value'),
     State('baths', 'value'), State('area', 'value'), State('year', 'value'), State('lang-store', 'data'), State('data-store', 'data')]
)
def update_app(lang_ar, lang_en, predict_clicks, theme_clicks, city, location, prop_type, beds, baths, area, year, lang_data, stored_data):
    ctx = dash.callback_context
    lang = lang_data or 'en'
    if ctx.triggered_id == 'lang-ar':
        lang = 'ar'
    elif ctx.triggered_id == 'lang-en':
        lang = 'en'

    try:
        if stored_data:
            data = pd.read_json(stored_data, orient='split').copy()
        else:
            data = raw_data.copy()
    except Exception as e:
        print(f"Error processing stored data: {e}")
        data = raw_data.copy()

    prediction = ""
    if predict_clicks > 0:
        input_dict = {
            'Beds': beds, 'Baths': baths, 'Area_in_sqft': area, 'Type': prop_type,
            'Furnishing': 'Furnished', 'Purpose': 'Rent', 'City': city, 'Location': location,
            'Season': 'صيف', 'Year': year, 'Month': 3,
            'Location_encoded': raw_data.groupby('Location')['Rent_per_sqft'].median().get(location, raw_data['Rent_per_sqft'].median()),
            'City_encoded': raw_data.groupby('City')['Rent_per_sqft'].median().get(city, raw_data['Rent_per_sqft'].median())
        }
        errors = validate_input(input_dict)
        if errors:
            prediction = html.Div([html.P(f"خطأ: {e}", style={'color': 'red'}) for e in errors])
        else:
            try:
                total_price, _ = ensemble_prediction(input_dict)
                price_range = f"{total_price - mae * area:,.2f} - {total_price + mae * area:,.2f} AED"
                prediction = html.Div([
                    html.P(f"{texts['price'][lang]}: {total_price:,.2f} AED", style={'fontSize': '18px', 'fontWeight': 'bold'}),
                    html.P(f"{texts['range'][lang]}: {price_range}", style={'fontSize': '16px'})
                ])
            except Exception as e:
                prediction = html.P(f"حدث خطأ أثناء التنبؤ: {str(e)}", style={'color': 'red'})

    try:
        data_tuple = tuple(map(tuple, data.to_records(index=False)))
        uae_map_tuple = tuple(map(tuple, uae_map_data.to_records(index=False)))
        trend_fig = create_price_trend(data_tuple, lang)
        heatmap_fig = create_advanced_heatmap(uae_map_tuple, lang)
        importance_fig = create_feature_importance(lang)
        seasonal_fig = seasonal_analysis(data_tuple, lang)
        roi_fig = investment_roi_analysis(data_tuple, lang)
    except Exception as e:
        print(f"Error generating figures: {e}")
        return [dash.no_update] * 18 + [lang]

    trend_desc = html.P("يُظهر هذا الرسم البياني التغير في متوسط الإيجار لكل قدم مربع عبر السنوات." if lang == 'ar' else "This chart shows the change in average rent per square foot over the years.", style={'fontSize': '14px', 'color': '#666'})
    heatmap_desc = html.P("تعرض الخريطة الحرارية توزيع أسعار الإيجار في مواقع مختارة بدبي." if lang == 'ar' else "The heatmap displays the distribution of rental prices across selected locations in Dubai.", style={'fontSize': '14px', 'color': '#666'})
    features_desc = html.P("يُبرز هذا الرسم العوامل العشرة الأكثر تأثيرًا على توقع الأسعار." if lang == 'ar' else "This chart highlights the top 10 factors influencing price predictions.", style={'fontSize': '14px', 'color': '#666'})
    seasonal_desc = html.P("يوضح الرسم التغيرات الموسمية في متوسط الإيجار خلال الأشهر." if lang == 'ar' else "This chart illustrates seasonal changes in average rent across months.", style={'fontSize': '14px', 'color': '#666'})
    roi_desc = html.P("يحلل الرسم العائد المتوقع على الاستثمار لخمسة مواقع مختارة." if lang == 'ar' else "This chart analyzes the expected return on investment for five selected locations.", style={'fontSize': '14px', 'color': '#666'})

    return (texts['title'][lang], texts['prediction'][lang], texts['beds'][lang], texts['baths'][lang], texts['area'][lang],
            texts['year'][lang], texts['predict'][lang], prediction, trend_fig, heatmap_fig, importance_fig, seasonal_fig, roi_fig,
            trend_desc, heatmap_desc, features_desc, seasonal_desc, roi_desc, lang)

if __name__ == "__main__":
    app.run_server(debug=True)