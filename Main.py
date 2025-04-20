# Gerekli kütüphaneleri içe aktarma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Veri setini yükleyelim
dataset = pd.read_csv("generated_data_10000.csv")

# "Date" ve "Day" sütunlarını sayısal verilere dönüştürme
le = LabelEncoder()
dataset['Day'] = le.fit_transform(dataset['Day'])  # "Day" sütunu sayısal hale getirildi
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%d-%m-%y')  # "Date" tarih formatına dönüştürüldü

# "Year", "Month", "DayOfWeek" gibi zaman serisi özelliklerini ekleyelim
dataset['Year'] = dataset['Date'].dt.year
dataset['Month'] = dataset['Date'].dt.month
dataset['DayOfWeek'] = dataset['Date'].dt.dayofweek  # Haftanın günü

# **Veri sayısını artırmak için mevcut veri setini çoğaltalım**
# Burada, örneklerin bir kısmını rastgele çoğaltıyoruz
augmentation_factor = 2  # Veri sayısını iki katına çıkarma
dataset = pd.concat([dataset] * augmentation_factor, ignore_index=True)

# Bağımsız (X) ve bağımlı (y) değişkenleri seçelim
X = dataset[['CodedDay', 'Zone', 'Weather', 'Temperature', 'Day', 'Year', 'Month', 'DayOfWeek']].values
y = dataset['Traffic'].values

# Veriyi eğitim ve test verisine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Özellik ölçekleme (Feature Scaling)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Random Forest modelini kurma
regressor = RandomForestRegressor(n_estimators=500, random_state=0, max_depth=15)

# Modeli eğitme
regressor.fit(X_train, y_train)

# Kullanıcıdan veri girmesini isteyelim
print("Trafik tahmini yapmak için verileri girin:")

# Kullanıcıdan girdi al
coded_day = int(input("CodedDay (Pazartesi = 1, Salı = 2, ... Pazar = 7): "))
zone = int(input("Bölge (Zone): "))
weather = int(input("Hava durumu kodu (örneğin, 35: Sisli, 36: Yağmurlu vb.): "))
temperature = int(input("Sıcaklık (Temperature): "))
day = int(input("Gün (Day): "))
year = int(input("Yıl (Year): "))
month = int(input("Ay (Month): "))
day_of_week = int(input("Haftanın günü (0 = Pazartesi, 1 = Salı, ... 6 = Pazar): "))

# Kullanıcıdan alınan veriyi 2D diziye dönüştürme
user_input = np.array([[coded_day, zone, weather, temperature, day, year, month, day_of_week]])

# Girdi verisini ölçekleme
user_input_scaled = sc_X.transform(user_input)

# Tahmin yapma
traffic_prediction = regressor.predict(user_input_scaled)

# Tahmin edilen trafik yoğunluğunu gösterme
print(f"Tahmin Edilen Trafik Yoğunluğu: {traffic_prediction[0]} (1: Az, 5: Çok yoğun)")

# Test seti üzerinde tahmin yapma
y_pred = regressor.predict(X_test)

# Yuvarlama işlemi
if (y_pred.all() < 2.5):
    y_pred = np.round(y_pred - 0.5)
else:
    y_pred = np.round(y_pred + 0.5)

# Hata oranını hesaplama
df1 = (y_pred - y_test) / y_test
df1 = round(df1.mean() * 100, 2)
print("Error = ", df1, "%")
accuracy = 100 - df1
print("Accuracy = ", accuracy, "%")

# Modelin doğruluğunu çapraz doğrulama ile kontrol edelim
cross_val_score_result = cross_val_score(regressor, X, y, cv=10)
print(f"Çapraz Doğrulama Doğruluğu: {cross_val_score_result.mean() * 100:.2f}%")

# Gerçek ve tahmin edilen değerleri karşılaştırmak için grafik çizme
plt.figure(figsize=(10,6))

# Gerçek değerler
plt.plot(y_test, color='blue', label='Gerçek Değerler')

# Tahmin edilen değerler
plt.plot(y_pred, color='red', label='Tahmin Edilen Değerler')

# Grafik başlıkları ve etiketler
plt.title('Gerçek ve Tahmin Edilen Değerler Karşılaştırması')
plt.xlabel('Veri Noktası')
plt.ylabel('Değer')
plt.legend()

# Grafiği gösterme
plt.show()
