#!/bin/bash

# Start a new tmux session named 'python_session' and run the Python script
tmux new-session -d -s python_session "python src/main.py"

echo "Started Python script in tmux session 'python_session'"

# Start another tmux session named 'ngrok_session' and run the ngrok command
tmux new-session -d -s ngrok_session "ngrok http --url=humbly-star-drake.ngrok-free.app 8001"

echo "Started ngrok in tmux session 'ngrok_session'"

