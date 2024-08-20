import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_lightning import Trainer

# Пример данных
data = pd.DataFrame({
    'time_idx': range(100),
    'value': range(100),
    'group': [0] * 50 + [1] * 50
})

# Создаем датасет
training = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target="value",
    group_ids=["group"],
    min_encoder_length=10,
    max_encoder_length=30,
    min_prediction_length=1,
    max_prediction_length=1
)

# Определяем модель
model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    output_size=1
)

# Обучение модели
trainer = Trainer(max_epochs=10)
trainer.fit(model, train_dataloader=torch.utils.data.DataLoader(training, batch_size=64))

# Сохранение модели
model_path = "path_to_save_tft_model"
model.save_model(model_path)
