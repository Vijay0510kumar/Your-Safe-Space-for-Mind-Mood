import os
import pandas as pd

USERS_FILE = 'data/users.csv'

def sign_up(username, password):
    if os.path.exists(USERS_FILE):
        df = pd.read_csv(USERS_FILE)
        if username in df['username'].values:
            return False, "Username already exists."
        new_row = pd.DataFrame({'username': [username], 'password': [password]})
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.DataFrame({'username': [username], 'password': [password]})
    df.to_csv(USERS_FILE, index=False)
    return True, "Account created."

def login(username, password):
    if os.path.exists(USERS_FILE):
        df = pd.read_csv(USERS_FILE)
        user_row = df[(df['username'] == username) & (df['password'] == password)]
        return not user_row.empty
    return False
