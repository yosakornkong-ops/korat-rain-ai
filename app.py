import streamlit as st
import pandas as pd
import requests
from xgboost import XGBRegressor
from datetime import datetime, timedelta

st.set_page_config(page_title="Korat Rain AI", page_icon="🌦️", layout="centered")
st.title("🌦️ AI พยากรณ์ฝน โคราช (Real-Time)")
st.markdown("ระบบดึงข้อมูลสภาพอากาศอัตโนมัติ และทำนายด้วย XGBoost (รันผ่าน Google Colab)")

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model('korat_rain_ultimate_model.json')
    return model

try:
    model = load_model()
    st.success("✅ โหลดโมเดล AI พร้อมใช้งาน!")
except:
    st.error("❌ หาไฟล์โมเดลไม่เจอ อย่าลืมอัปโหลด korat_rain_ultimate_model.json ขึ้น Colab ก่อนนะครับ")
    st.stop()

if st.button("🔮 กดเพื่อพยากรณ์ฝนชั่วโมงนี้ (Auto Fetch)", use_container_width=True):
    with st.spinner("⏳ กำลังดึงข้อมูลสภาพอากาศโคราชและชัยภูมิ..."):
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=12)
            
            url_korat = f"https://api.open-meteo.com/v1/forecast?latitude=14.97&longitude=102.10&hourly=precipitation,temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover&start_hour={start_time.strftime('%Y-%m-%dT%H:00')}&end_hour={end_time.strftime('%Y-%m-%dT%H:00')}"
            df = pd.DataFrame(requests.get(url_korat).json()['hourly'])
            
            url_cp = f"https://api.open-meteo.com/v1/forecast?latitude=15.80&longitude=102.03&hourly=surface_pressure&start_hour={start_time.strftime('%Y-%m-%dT%H:00')}&end_hour={end_time.strftime('%Y-%m-%dT%H:00')}"
            df['pressure_chaiyaphum'] = requests.get(url_cp).json()['hourly']['surface_pressure']

            df['datetime'] = pd.to_datetime(df['time'])
            df['month'] = df['datetime'].dt.month
            df['hour'] = df['datetime'].dt.hour
            df['hourlyPrecipRate'] = df['precipitation'] 
            df['temperature'] = df['temperature_2m']
            df['humidity'] = df['relative_humidity_2m']
            df['pressure'] = df['surface_pressure']
            df['wind_speed'] = df['wind_speed_10m']
            
            df['rain_lag_1'] = df['precipitation'].shift(1)
            df['rain_lag_2'] = df['precipitation'].shift(2)
            df['rain_sum_3h'] = df['precipitation'].shift(1).rolling(3).sum()
            df['rain_sum_6h'] = df['precipitation'].shift(1).rolling(6).sum()
            df['humidity_mean_3h'] = df['humidity'].shift(1).rolling(3).mean()
            df['pressure_diff_3h'] = df['pressure'].shift(1) - df['pressure'].shift(4)
            df['enso_index'] = 0.5 
            
            current_data = df.dropna().iloc[-1:]
            features = ['hourlyPrecipRate', 'temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover', 'month', 'hour', 'rain_lag_1', 'rain_lag_2', 'rain_sum_3h', 'rain_sum_6h', 'humidity_mean_3h', 'pressure_diff_3h', 'enso_index', 'pressure_chaiyaphum']
            
            # สั่ง AI ทำนาย
            prediction = max(0, model.predict(current_data[features])[0])
            
            # 🛑 กฎปัดเศษ: ถ้าน้อยกว่า 0.5 มม. ให้ถือว่าฝนไม่ตก (บังคับให้เป็น 0)
            if prediction < 0.5:
                prediction = 0.0
            
            st.markdown("---")
            st.subheader("📊 ผลการวิเคราะห์ ณ เวลาปัจจุบัน")
            
            st.info(f"🌡️ อุณหภูมิ: {current_data['temperature'].values[0]:.1f} °C  |  💧 ความชื้น: {current_data['humidity'].values[0]:.0f} %  |  ☁️ เมฆ: {current_data['cloud_cover'].values[0]:.0f} %")
            
            st.markdown("### 🤖 AI ทำนายปริมาณฝนชั่วโมงนี้:")
            if prediction > 5.0: st.error(f"🚨 {prediction:.2f} มิลลิเมตร (ฝนตกหนัก เตรียมรับมือ!)")
            elif prediction > 1.0: st.warning(f"🌂 {prediction:.2f} มิลลิเมตร (ฝนตกปานกลาง พกร่มด้วยครับ)")
            elif prediction > 0.0: st.info(f"☁️ {prediction:.2f} มิลลิเมตร (ฝนตกปรอยๆ เล็กน้อย)")
            else: st.success(f"☀️ 0.00 มิลลิเมตร (อากาศแจ่มใส ฝนไม่ตก)")
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
