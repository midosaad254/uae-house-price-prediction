# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from predictor import predict_price
import joblib
import os
import requests
import time
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# تحديد المسارات
base_path = r"D:\trans\ai\machine learning 3\projects\1\uae House Price Prediction\dubai_properties"
raw_data = pd.read_csv(os.path.join(base_path, "results", "raw_processed_data.csv"))
processed_data = pd.read_csv(os.path.join(base_path, "results", "processed_data.csv"))
scaler_path = os.path.join(base_path, "models", "scaler.pkl")
model_path = os.path.join(base_path, "models", "best_model.pkl")
mae = joblib.load(os.path.join(base_path, "models", "best_model_mae.pkl"))

# تعريف البيانات الجغرافية
locations = raw_data['Location'].unique()[:4]
uae_map_data = pd.DataFrame({
    'Location': locations,
    'Rent_per_sqft': raw_data[raw_data['Location'].isin(locations)].groupby('Location')['Rent_per_sqft'].mean().reindex(locations),
    'lat': [25.276987, 25.204849, 25.077250, 25.197197],
    'lon': [55.296249, 55.270783, 55.301500, 55.171280]
}).dropna()

# تعريف الدوال
def validate_input(input_dict, lang_code='en'):
    errors = []
    if input_dict['Beds'] <= 0 or input_dict['Beds'] > 20:
        errors.append("عدد غرف النوم يجب أن يكون بين 1 و20" if lang_code == 'ar' else "Bedrooms must be between 1 and 20")
    if input_dict['Area_in_sqft'] <= 0 or input_dict['Area_in_sqft'] > 50000:
        errors.append("المساحة يجب أن تكون بين 1 و50000 قدم مربع" if lang_code == 'ar' else "Area must be between 1 and 50000 sqft")
    if input_dict['Location'] not in raw_data['Location'].unique():
        errors.append("الموقع غير موجود في قاعدة البيانات" if lang_code == 'ar' else "Location not found in database")
    if input_dict['Year'] < 2000 or input_dict['Year'] > 2050:
        errors.append("السنة يجب أن تكون بين 2000 و2050" if lang_code == 'ar' else "Year must be between 2000 and 2050")
    return errors

def create_enhanced_pdf_report(property_data, prediction, lang_code='en'):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("تقرير تحليل العقار" if lang_code == 'ar' else "Property Analysis Report", styles['Heading1']),
        Spacer(1, 12)
    ]
    data = [["المدخلات" if lang_code == 'ar' else "Input", "القيمة" if lang_code == 'ar' else "Value"],
            ["الموقع" if lang_code == 'ar' else "Location", property_data['Location']],
            ["المساحة" if lang_code == 'ar' else "Area", f"{property_data['Area_in_sqft']} قدم مربع" if lang_code == 'ar' else f"{property_data['Area_in_sqft']} sqft"],
            ["غرف النوم" if lang_code == 'ar' else "Bedrooms", property_data['Beds']],
            ["الحمامات" if lang_code == 'ar' else "Bathrooms", property_data['Baths']],
            ["السنة" if lang_code == 'ar' else "Year", property_data['Year']]]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.green),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.extend([table, Spacer(1, 20), Paragraph(f"السعر المتوقع: {prediction:,.2f} AED" if lang_code == 'ar' else f"Predicted Price: {prediction:,.2f} AED", styles['Heading2']),
                     Spacer(1, 12), Paragraph("تحليل السوق:" if lang_code == 'ar' else "Market Analysis:", styles['Heading3']),
                     Paragraph("يظهر تحليل السوق الحالي اتجاهاً تصاعدياً بنسبة 4.2% سنوياً..." if lang_code == 'ar' else "Current market analysis shows an upward trend of 4.2% annually...", styles['Normal'])])
    doc.build(elements)
    return buffer.getvalue()

# CSS احترافي
st.markdown("""
    <style>
    body { font-family: 'Dubai', Arial; background-color: #F8F9FA; color: #2C3E50; }
    .stButton>button { background-color: #006633; color: white; border-radius: 5px; padding: 10px; }
    .card { background-color: #FFFFFF; padding: 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-left: 5px solid #D4AF37; }
    .title { color: #006633; text-align: center; }
    .rtl { direction: rtl; text-align: right; }
    .ltr { direction: ltr; text-align: left; }
    </style>
""", unsafe_allow_html=True)
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Dubai:wght@300;400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# النصوص
texts = {
    'title': {'ar': '🏠 تحليل العقارات الذكي', 'en': '🏠 Smart Property Analytics'},
    'tabs': {'ar': ["توقع السعر", "تحليل البيانات", "المستقبل"], 'en': ["Prediction", "Analysis", "Future"]},
    'beds': {'ar': "عدد غرف النوم", 'en': "Bedrooms"},
    'baths': {'ar': "عدد الحمامات", 'en': "Bathrooms"},
    'area': {'ar': "المساحة (قدم مربع)", 'en': "Area (sqft)"},
    'year': {'ar': "السنة (YYYY)", 'en': "Year (YYYY)"},  # إضافة حقل السنة
    'type': {'ar': "نوع العقار", 'en': "Property Type"},
    'city': {'ar': "المدينة", 'en': "City"},
    'location': {'ar': "الموقع", 'en': "Location"},
    'calculate': {'ar': "احسب", 'en': "Calculate"},
    'price': {'ar': "السعر المتوقع", 'en': "Predicted Price"},
    'range': {'ar': "النطاق", 'en': "Range"},
    'download': {'ar': "تحميل تقرير PDF", 'en': "Download PDF Report"},
    'market_title': {'ar': 'مؤشرات السوق', 'en': 'Market Indicators'},
    'heatmap_title': {'ar': 'توزيع الأسعار', 'en': 'Price Distribution'},
    'seasonal_title': {'ar': 'التغيرات الموسمية', 'en': 'Seasonal Trends'},
    'roi_title': {'ar': 'العائد على الاستثمار', 'en': 'Return on Investment'}
}

# الشريط الجانبي
with st.sidebar:
    lang = st.selectbox("اختر اللغة / Select Language", ["العربية", "English"])
    lang_code = 'ar' if lang == "العربية" else 'en'
    
    st.subheader("البحث السريع" if lang_code == 'ar' else "Quick Search")
    search_query = st.text_input("أدخل اسم المنطقة" if lang_code == 'ar' else "Enter location name")
    if search_query:
        filtered_locations = [loc for loc in raw_data['Location'].unique() if search_query.lower() in loc.lower()]
        if filtered_locations:
            st.success(f"تم العثور على {len(filtered_locations)} منطقة" if lang_code == 'ar' else f"Found {len(filtered_locations)} locations")
            selected_loc = st.selectbox("اختر منطقة:" if lang_code == 'ar' else "Select location:", filtered_locations)
        else:
            st.error("لم يتم العثور على نتائج" if lang_code == 'ar' else "No results found")
    
    st.markdown(f"<h3>{texts['market_title'][lang_code]}</h3>", unsafe_allow_html=True)
    indicators = {'نمو السوق': '4.2%', 'مدة بقاء العقار': '45 يوم', 'معدل الشغور': '7.8%'}
    cols = st.columns(3)
    for i, (k, v) in enumerate(indicators.items()):
        with cols[i]:
            st.metric(k if lang_code == 'ar' else k.replace(' ', '_'), v)

# العنوان الرئيسي وعلامات التبويب
st.markdown(f"<h1 class='title'>{texts['title'][lang_code]}</h1>", unsafe_allow_html=True)
tabs = st.tabs(texts['tabs'][lang_code])

# تبويب التوقع
with tabs[0]:
    st.markdown(f"<div class='card {'rtl' if lang_code == 'ar' else 'ltr'}'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        beds = st.number_input(texts['beds'][lang_code], min_value=1, max_value=15, value=2)
        baths = st.number_input(texts['baths'][lang_code], min_value=1, max_value=15, value=2)
        area = st.number_input(texts['area'][lang_code], min_value=100, max_value=20000, value=1000, step=50)
        year = st.number_input(texts['year'][lang_code], min_value=2000, max_value=2050, value=2025)  # إضافة حقل السنة
    with col2:
        type_input = st.selectbox(texts['type'][lang_code], raw_data['Type'].unique())
        city = st.selectbox(texts['city'][lang_code], raw_data['City'].unique())
        location = st.selectbox(texts['location'][lang_code], raw_data['Location'].unique())
    
    if st.button(texts['calculate'][lang_code]):
        input_dict = {
            'Beds': beds, 'Baths': baths, 'Area_in_sqft': area, 'Type': type_input,
            'Furnishing': 'Furnished', 'Purpose': 'Rent', 'City': city, 'Location': location,
            'Season': 'صيف', 'Year': year, 'Month': 3,  # إضافة السنة من الإدخال
            'Location_encoded': raw_data.groupby('Location')['Rent_per_sqft'].median().get(location, raw_data['Rent_per_sqft'].median()),
            'City_encoded': raw_data.groupby('City')['Rent_per_sqft'].median().get(city, raw_data['Rent_per_sqft'].median())
        }
        errors = validate_input(input_dict, lang_code)
        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner('جاري حساب التوقعات...' if lang_code == 'ar' else 'Calculating predictions...'):
                time.sleep(1)
                try:
                    total_price, _ = predict_price(input_dict, os.path.join(base_path, "results", "processed_data.csv"), scaler_path, model_path)
                    price_range = f"{total_price - mae * area:,.2f} - {total_price + mae * area:,.2f} AED"
                    st.success('تم الانتهاء من التوقع!' if lang_code == 'ar' else 'Prediction completed!')
                    st.markdown(f"<div class='card'><b>{texts['price'][lang_code]}:</b> {total_price:,.2f} AED<br><b>{texts['range'][lang_code]}:</b> {price_range}</div>", unsafe_allow_html=True)

                    market_avg = raw_data[raw_data['Location'] == location]['Rent_per_sqft'].mean() * area
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=total_price,
                        delta={'reference': market_avg, 'relative': True, 'valueformat': '.1%'},
                        title={'text': f"مقارنة مع متوسط سوق {location}" if lang_code == 'ar' else f"Comparison with {location} Market Average"},
                        gauge={
                            'axis': {'range': [None, market_avg * 2]},
                            'bar': {'color': "#D4AF37"},
                            'steps': [
                                {'range': [0, market_avg * 0.9], 'color': "#009900"},
                                {'range': [market_avg * 0.9, market_avg * 1.1], 'color': "#FFDD00"},
                                {'range': [market_avg * 1.1, market_avg * 2], 'color': "#CC0000"}
                            ],
                            'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': market_avg}
                        }
                    ))
                    st.plotly_chart(fig)

                    pdf_data = create_enhanced_pdf_report(input_dict, total_price, lang_code)
                    st.download_button(texts['download'][lang_code], pdf_data, file_name="property_report.pdf")
                except Exception as e:
                    st.error(f"حدث خطأ أثناء التنبؤ: {str(e)}" if lang_code == 'ar' else f"Error during prediction: {str(e)}")
                    st.info("يرجى التحقق من المدخلات والمحاولة مرة أخرى" if lang_code == 'ar' else "Please check inputs and try again")
    st.markdown("</div>", unsafe_allow_html=True)

# تبويب التحليل
with tabs[1]:
    st.markdown(f"<div class='card {'rtl' if lang_code == 'ar' else 'ltr'}'>", unsafe_allow_html=True)
    heatmap_fig = go.Figure(go.Densitymapbox(
        lat=uae_map_data['lat'], lon=uae_map_data['lon'], z=uae_map_data['Rent_per_sqft'],
        radius=15, colorscale='YlOrRd', opacity=0.8,
        colorbar_title='الإيجار (درهم)' if lang_code == 'ar' else 'Rent (AED)',
        hovertemplate='<b>%{text}</b><br>الإيجار: %{z:.2f} درهم<extra></extra>' if lang_code == 'ar' else '<b>%{text}</b><br>Rent: %{z:.2f} AED<extra></extra>',
        text=uae_map_data['Location']
    ))
    heatmap_fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=9, mapbox_center={"lat": 25.2, "lon": 55.3},
        title={'text': texts['heatmap_title'][lang_code], 'font': {'size': 20, 'color': '#006633'}, 'x': 0.5, 'xanchor': 'center'},
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor='#F8F9FA'
    )
    st.plotly_chart(heatmap_fig)
    st.markdown("تعرض هذه الخريطة توزيع أسعار الإيجار في دبي." if lang_code == 'ar' else "This map shows the distribution of rental prices in Dubai.", unsafe_allow_html=True)
    
    st.subheader("مقارنة بين المناطق" if lang_code == 'ar' else "Location Comparison")
    selected_locations = st.multiselect("اختر المناطق للمقارنة" if lang_code == 'ar' else "Select locations to compare", raw_data['Location'].unique())
    if selected_locations:
        comp_fig = px.bar(
            raw_data[raw_data['Location'].isin(selected_locations)].groupby('Location')['Rent_per_sqft'].mean().reset_index(),
            x='Location', y='Rent_per_sqft', color='Location', color_discrete_sequence=px.colors.sequential.YlOrRd,
            text_auto='.2f', title="مقارنة متوسط الإيجار بين المناطق" if lang_code == 'ar' else "Comparison of Average Rent Across Locations"
        )
        comp_fig.update_traces(textposition='auto')
        comp_fig.update_layout(showlegend=False, plot_bgcolor='#F8F9FA', paper_bgcolor='#F8F9FA')
        st.plotly_chart(comp_fig)
        st.markdown("يُظهر هذا الرسم متوسط الإيجار لكل قدم مربع في المناطق المختارة." if lang_code == 'ar' else "This chart shows the average rent per square foot for selected locations.", unsafe_allow_html=True)
    
    st.subheader("توزيع الأسعار حسب الغرف" if lang_code == 'ar' else "Price Distribution by Bedrooms")
    dist_fig = px.box(
        raw_data, x="Beds", y="Rent_per_sqft", color="City",
        title="توزيع الأسعار حسب عدد غرف النوم والمدينة" if lang_code == 'ar' else "Price Distribution by Bedrooms and City",
        color_discrete_sequence=px.colors.sequential.YlOrRd
    )
    dist_fig.update_layout(
        xaxis_title='غرف النوم' if lang_code == 'ar' else 'Bedrooms',
        yaxis_title='الإيجار (درهم)' if lang_code == 'ar' else 'Rent (AED)',
        template='plotly_white', plot_bgcolor='#F8F9FA', paper_bgcolor='#F8F9FA'
    )
    st.plotly_chart(dist_fig)
    st.markdown("يوضح هذا الرسم نطاق أسعار الإيجار حسب عدد غرف النوم في كل مدينة." if lang_code == 'ar' else "This chart illustrates the range of rental prices by bedroom count across cities.", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# تبويب المستقبل
with tabs[2]:
    st.markdown(f"<div class='card {'rtl' if lang_code == 'ar' else 'ltr'}'>", unsafe_allow_html=True)
    seasonal_fig = go.Figure(go.Scatter(
        x=raw_data.groupby('Month')['Rent_per_sqft'].mean().index,
        y=raw_data.groupby('Month')['Rent_per_sqft'].mean(),
        mode='lines+markers',
        line=dict(color='#D4AF37', width=2.5),
        marker=dict(size=10, color='#006633', line=dict(width=2, color='#FFFFFF')),
        hovertemplate='الشهر: %{x}<br>الإيجار: %{y:.2f} درهم<extra></extra>' if lang_code == 'ar' else 'Month: %{x}<br>Rent: %{y:.2f} AED<extra></extra>'
    ))
    seasonal_fig.update_layout(
        title={'text': texts['seasonal_title'][lang_code], 'font': {'size': 20, 'color': '#006633'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='الشهر' if lang_code == 'ar' else 'Month',
        yaxis_title='الإيجار (درهم)' if lang_code == 'ar' else 'Rent (AED)',
        template='plotly_white',
        xaxis=dict(gridcolor='#E8ECEF', gridwidth=1, showgrid=True),
        yaxis=dict(gridcolor='#E8ECEF', gridwidth=1, showgrid=True),
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='#F8F9FA'
    )
    st.plotly_chart(seasonal_fig)
    st.markdown("يُظهر هذا الرسم المتوسط الشهري للإيجار." if lang_code == 'ar' else "This chart displays the monthly average rent.", unsafe_allow_html=True)

    roi_data = pd.DataFrame({
        'Year': [2025 + i for i in range(5)],
        'ROI': [raw_data['Rent_per_sqft'].mean() * 12 * (1.03 ** i) / (raw_data['Rent_per_sqft'].mean() * 12 * 15) * 100 for i in range(5)]
    })
    roi_fig = px.line(
        roi_data, x='Year', y='ROI',
        line_shape='spline', color_discrete_sequence=['#D4AF37'],
        hover_data={'ROI': ':.2f'}
    )
    roi_fig.update_traces(line_width=2.5)
    roi_fig.update_layout(
        title={'text': texts['roi_title'][lang_code], 'font': {'size': 20, 'color': '#006633'}, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='السنة' if lang_code == 'ar' else 'Year',
        yaxis_title='ROI (%)',
        template='plotly_white',
        xaxis=dict(gridcolor='#E8ECEF', gridwidth=1, showgrid=True),
        yaxis=dict(gridcolor='#E8ECEF', gridwidth=1, showgrid=True),
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='#F8F9FA'
    )
    st.plotly_chart(roi_fig)
    st.markdown("يحسب هذا الرسم العائد المتوقع على الاستثمار بناءً على متوسط الإيجار." if lang_code == 'ar' else "This chart calculates the expected return on investment based on average rent.", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)