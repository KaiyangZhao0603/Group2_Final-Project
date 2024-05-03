from tensorflow.keras.models import Sequential
from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_model(input_shape):
    model = Sequential([
        TCN(input_shape=input_shape, nb_filters=64, kernel_size=6, nb_stacks=1,
            dilations=[1, 2, 4, 8, 16], padding='causal', use_skip_connections=True,
            dropout_rate=0.2, return_sequences=False, activation='relu', kernel_initializer='he_normal'),
        Dense(2, activation='linear', kernel_regularizer=l2(0.01))
    ])
    BatchNormalization(),
    optimizer = Adam(learning_rate=0.005)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model


