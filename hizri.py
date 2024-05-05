import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd 
import pandas_datareader as web 
import datetime as dt
import ccxt

from sklearn.preprocessing import MinMaxScaler  
from keras.models import Sequential 
from keras.layers import Dense, Dropout, LSTM  


class DataLoader:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol  # инициализация данных для построенния даты
        self.timeframe = timeframe
        self.exchange = ccxt.binance()

    def load_data(self):  # метод, который загружает дату и меняет первый столбец индексов на временнОй столбец
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def preprocess_data(self, df):  # метод , который подготавливает наши данные для машинного обучения
        scaler = MinMaxScaler(
            feature_range=(0, 1))  # minmaxscaler ,можно сказать, преобразует наши данные в числовой диапозон 0-1
        scaled_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))
        return scaled_data, scaler


class LSTMModel:  # этот класс отвеает за создание , обучение и прогнозированние с использованием модели LSTM
    def __init__(self, scaled_data, prediction_days):
        self.scaled_data = scaled_data
        self.prediction_days = prediction_days  # количество дней исходя из которых будет обучаться нейросеть

    def create_model(self):  # этот метод создает саму модель LSTM
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.prediction_days, 1)))
        '''В строке выше создается слой памяти. Количество ячеек зависит от параметра units , параметр
        return sequences отвечает за возвращение выходных данных , параметр input_shape определает форму данных, в нашем случае
        это трехмерный тензор'''
        model.add(Dropout(
            0.2))  # dropout по сути "выкидывает" установленное нами количество данныъ(20%) для исключения зависимости модели от какой-то входной единицы
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # слой Dense служит для соединения нейронов из предыдущих слоев
        model.compile(optimizer='adam', loss='mean_squared_error')
        """используем оптимизатор adam, т.к. он эффективен и потребляет мало памяти,
        mean_squared_error используется для минимизации ошибок предсказания  """
        return model

    def train_model(self, model, x_train, y_train):  # этот метод обучает модель с использованием преобразованных данных
        model.fit(x_train, y_train, epochs=25, batch_size=32)

    def make_predictions(self, model, x_test):  # метод делающий прогноз с использованием обученной модели
        predicted_price = model.predict(x_test)
        return predicted_price


class Predictor:  # этот класс отвечает за построение прогнозов
    def __init__(self, data_loader, lstm_model):
        self.data_loader = data_loader
        self.lstm_model = lstm_model

    def plot_predictions(self, actual_prices, predicted_price,
                         prediction):  # этот метод отображает фактические и прогнозируемые цены
        plt.plot(actual_prices, color="black")
        plt.plot(predicted_price, color="green")
        plt.xlabel(f"Предсказанная цена на следующие n минут: {prediction[0][0]}")
        plt.show()

    def make_single_prediction(self, model, real_data):  # этот метод возвращает один прогноз по цене
        prediction = model.predict(real_data)
        return prediction


# загружаем данные

data_loader = DataLoader("XRP/USDT", '5m')
df = data_loader.load_data()

# преобразуем данные
scaled_data, scaler = data_loader.preprocess_data(df)

# создаем модель LSTM
lstm_model = LSTMModel(scaled_data, 60)
model = lstm_model.create_model()

# Обучаем модель
x_train = []
y_train = []

for x in range(lstm_model.prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - lstm_model.prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

lstm_model.train_model(model, x_train, y_train)

# Создаем прогозы
actual_prices = df['close'].values
total_dataset = pd.concat((df['close'], df['close'].head(500)), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(df['close'].head(500)) - lstm_model.prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)

model_inputs = scaler.transform(model_inputs)

x_test = []

for x in range(lstm_model.prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - lstm_model.prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_price = lstm_model.make_predictions(model, x_test)
predicted_price = scaler.inverse_transform(predicted_price)

# Выводим прогнозируеимую цену
real_data = [model_inputs[len(model_inputs) + 1 - lstm_model.prediction_days:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = lstm_model.make_predictions(model, real_data)
prediction = scaler.inverse_transform(prediction)

# Прогнозируем цену
predictor = Predictor(data_loader, lstm_model)
predictor.plot_predictions(actual_prices, predicted_price, prediction)