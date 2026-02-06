# Employee Performance ML

Predict employee promotion using a **FastAPI backend** and **interactive frontend**.

## Features
- Train ML model with CSV data
- Test ML model with CSV data
- Predict single employee promotion via frontend form
- Frontend includes guidance for each input field

## Folder Structure
employee-performance-ml/
├── backend/
├── frontend/
├── dataset/
├── model/
├── requirements.txt
├── README.md
├── .gitignore
└── venv/


## Setup Locally

1. Clone repo:
```bash
git clone https://github.com/atharvananaware/employee-performance-ml.git
cd employee-performance-ml

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

uvicorn backend.main:app --reload
