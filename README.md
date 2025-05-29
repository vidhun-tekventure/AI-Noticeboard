# Noticeboard AI Validator

This project trains a deep learning model to validate shop noticeboards against fixed visual rules (e.g., Arabic text, background color, logo presence). Includes a FastAPI interface.

## Run

```bash
## Create a Virtual Env
python -m venv venv

## Activate Virtual Env
.\venv\Scripts\activate

## Install Packages
pip install -r requirements.txt

## Train the Model
python train.py

## Run the python app
uvicorn main:app --reload
```

Go to `http://127.0.0.1:8000/docs` to try the API.
