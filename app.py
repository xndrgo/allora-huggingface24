from flask import Flask, Response
import requests
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Создаем Flask приложение
app = Flask(__name__)

# Функция для получения данных с Coingecko
def get_coingecko_url(token):
    base_url = "https://api.coingecko.com/api/v3/coins/"
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum'
    }
    
    token = token.upper()
    if token in token_map:
        url = f"{base_url}{token_map[token]}/market_chart?vs_currency=usd&days=30&interval=daily"
        return url
    else:
        raise ValueError("Unsupported token")

# Предобработка данных
def preprocess_data(data):
    df = pd.DataFrame(data["prices"], columns=["date", "price"])
    df["date"] = pd.to_datetime(df["date"], unit='ms')
    df["price_change"] = df["price"].pct_change()  # добавляем столбец с процентным изменением цен
    df = df.dropna()  # убираем строки с NaN
    return df

# Обучение и предсказание модели
def train_and_predict(df, days_ahead=1):
    X = np.array(df.index).reshape(-1, 1)  # Даты как индекс
    y = df["price"].values

    # Разделение данных на обучающую и тестовую выборку
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Предсказание на следующий период
    last_index = np.array([[df.index[-1] + days_ahead]])
    predicted_price = model.predict(last_index)

    # Оценка модели
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return predicted_price[0]

# Определяем endpoint
@app.route("/inference/<string:token>")
def get_inference(token):
    """Генерация предсказания для заданного токена."""
    try:
        url = get_coingecko_url(token)
    except ValueError as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG-your_api_key" # Замените на ваш API ключ
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        df = preprocess_data(data)
        predicted_price = train_and_predict(df, days_ahead=1)  # Прогноз на следующий день
        return Response(json.dumps({"predicted_price": predicted_price}), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"error": "Failed to retrieve data from the API"}), 
                        status=response.status_code, 
                        mimetype='application/json')

# Запуск Flask приложения
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
