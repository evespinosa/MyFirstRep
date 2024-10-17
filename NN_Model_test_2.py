import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

data_base=f'D:/AFL_GRP/DataBase/AFLDataVault.db'
connection=sqlite3.connect(database=data_base)
query=f''' 
Select-- hg.year,hg.round,hg.team,hg.opponent ,sgd.Ground
        sgd.Team_Result
       ,sgs.Disposals_diff
       ,sgs.Kicks_diff
       ,sgs.Marks_diff
       ,sgs.Handballs_diff
       ,sgs.HitOuts_diff
       ,sgs.Tackles_diff
       ,sgs.Rebounds_diff
       ,sgs.Inside50_diff
       ,sgs.Clearances_diff
       ,sgs.Clangers_diff
       ,sgs.Frees_diff
       ,sgs.FreesAgainst_diff
       ,sgs.ContestedPossessions_diff
       ,sgs.UncontestedPossessions_diff
       ,sgs.ContestedMarks_diff
       ,sgs.MarksInside50_diff
       ,sgs.OnePercenters_diff
       ,sgs.Bounces_diff
       ,sgs.GoalAssists_diff
from h_game hg
join S_Game_Details sgd
on   hg.Game_hashkey=sgd.game_hashkey
join s_Game_statistics sgs
on   hg.game_hashkey=sgs.game_hashkey
'''
df=pd.read_sql(query,connection)
connection.close()

df['Team_Result'] = df['Team_Result'].replace({0: 0, 1: 1, 9: 2})
print(df.head())
# Features and target
X = df.drop('Team_Result', axis=1)  # Input features
y = df['Team_Result']               # Target variable

# Preprocessing: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features (mean=0, variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the target to one-hot encoding for multiclass classification
y_train = to_categorical(y_train, num_classes=3)  # 3 classes (0, 1, 9)
y_test = to_categorical(y_test, num_classes=3)

# Build the neural network model
model = Sequential()

# Input layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Hidden layers
model.add(Dense(32, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
# Output layer for 3 classes (softmax for multiclass classification)
model.add(Dense(3, activation='softmax'))  # 3 output units for 0, 1, 9

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))



# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
