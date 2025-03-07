# UAE House Price Prediction
A machine learning project to predict property rental prices in the UAE, featuring interactive dashboards built with Dash and Streamlit.
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project leverages machine learning to predict rental prices of properties in the UAE based on features such as location, area, number of bedrooms, and more. It includes two interactive web interfaces:
- **Dash**: A dynamic dashboard for real-time price predictions and advanced visualizations.
- **Streamlit**: A user-friendly interface with additional analysis and reporting tools.
## Features
- Accurate rental price predictions using a trained Random Forest model.
- Visualizations including price trends, heatmaps, seasonal trends, and ROI analysis.
- Bilingual support (Arabic and English) for broader accessibility.
- PDF report generation for detailed insights (Streamlit).

## Project Structure
uae-house-price-prediction/ ├── data/ # Raw and processed datasets │ ├── raw_data.csv │ └── processed_data.csv ├── models/ # Trained ML models and scalers │ ├── best_model.pkl │ ├── scaler.pkl │ └── best_model_mae.pkl ├── results/ # Processed data outputs │ ├── raw_processed_data.csv │ └── processed_data.csv ├── src/ # Source code │ ├── dash_app.py # Dash application │ ├── streamlit_app.py # Streamlit application │ ├── predictor.py # Prediction logic ├── assets/ # Static files (e.g., logo) │ └── logo.png ├── .gitignore # Files to ignore in Git ├── README.md # Project documentation ├── requirements.txt # Dependencies └── LICENSE # Project license

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/midosaad254/uae-house-price-prediction.git
   cd uae-house-price-prediction

   ## Usage
### Dash Application
Run the Dash app for an interactive dashboard:
```bash
cd src
python dash_app.py

Open your browser at http://127.0.0.1:8050.

Streamlit Application
Run the Streamlit app for a simplified interface with reporting
cd src
streamlit run streamlit_app.py
Open your browser at http://localhost:8501

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a Pull Request.
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
