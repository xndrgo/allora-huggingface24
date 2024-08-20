from flask import Flask, Response
import requests
import json
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet

# create our Flask app
app = Flask(__name__)

# Define the path to your trained TFT model
model_path = "path_to_your_trained_tft_model"

# Load your trained TFT model
model = TemporalFusionTransformer.load_from_checkpoint(model_path)

# Define the parameters for your TFT model
max_encoder_length = 30  # Example length, adjust as needed
max_prediction_length = 1

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

@app.route("/inference/<string:token>")
def get_inference(token):
    """Generate inference for given token."""
    try:
        # Get the data from Coingecko
        url = get_coingecko_url(token)
    except ValueError as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG-your_api_key"  # Replace with your API key
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df = df.set_index("timestamp")
    else:
        return Response(json.dumps({"Failed to retrieve data from the API": str(response.text)}), 
                        status=response.status_code, 
                        mimetype='application/json')

    # Prepare data for the model
    df = df[['price']]
    df = df.reset_index()
    dataset = TimeSeriesDataSet(
        df,
        time_idx="timestamp",
        target="price",
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # Generate prediction
    try:
        model.eval()
        predictions = []
        for batch in dataloader:
            x, y = batch
            with torch.no_grad():
                pred = model(x)
            predictions.append(pred)
        
        # Assuming you want the mean of the predictions
        prediction_mean = torch.mean(torch.stack(predictions))
        return Response(str(prediction_mean.item()), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

# Run our Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
