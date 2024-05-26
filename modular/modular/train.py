import torch
from torch import nn
import model_builder, data_setup, engine, utils

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEQUENCE_LENGTH = 180

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader = data_setup.create_dataloader(X_train=X_train, X_test=X_test, y_train=y_train,
                                                                y_test=y_test, batch_size=BATCH_SIZE)

model = model_builder.LSTMModel(input_size=1, hidden_size=8, num_layers=2, output_size=1)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

results = engine.train_model(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                             loss_fn=loss_fn, optimizer=optimizer, epochs=EPOCHS, device=device)

utils.save_model(model=model, model_name=city_name)