
import os
import datetime

def log(message, task):
    if task is None:
        return
    if (not os.path.exists(task)):
        os.makedirs('logs', exist_ok=True)
        with open(f'logs/{task}', 'w') as f:
            f.write(f"[{datetime.datetime.now()}] {message}\n")
    else:
        with open(f'logs/{task}', 'a') as f:
            f.write(f"[{datetime.datetime.now()}] {message}\n")
