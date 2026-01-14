# 🩺 Monkeypox Prediction System

A personalized federated learning system with evolutionary strategies for predicting monkeypox test outcomes across five states in southeastern Nigeria.

## 📋 Overview

This application uses state-specific AI models trained collaboratively through federated learning to predict monkeypox test results based on patient symptoms. Each state (Abia, Anambra, Enugu, Imo, and Ebonyi) has its own personalized model while benefiting from collective learning across all states.

## 🎯 Features

- **State-Specific Models**: Five personalized models, each optimized for local data patterns
- **Privacy-Preserving**: Federated learning ensures data never leaves its source
- **Evolutionary Strategy**: Adaptive hyperparameter optimization improves model performance over time
- **User-Friendly Interface**: Simple symptom checklist for clinical decision support
- **Real-Time Predictions**: Instant risk assessment with probability scores

## 🏗️ Architecture

### Personalized Federated Learning with Evolutionary Strategy (PFL-ES)

```
┌─────────────────────────────────────────────────────────────┐
│                    Federated Aggregation                     │
│         (Combines models from high-performing clients)       │
└─────────────────────────────────────────────────────────────┘
         ▲           ▲           ▲           ▲           ▲
         │           │           │           │           │
    ┌────┴────┐ ┌────┴────┐ ┌───┴─────┐ ┌───┴─────┐ ┌───┴─────┐
    │  Abia   │ │ Anambra │ │  Enugu  │ │   Imo   │ │ Ebonyi  │
    │  Model  │ │  Model  │ │  Model  │ │  Model  │ │  Model  │
    └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

### Model Architecture

- **Input Layer**: 24 features (23 symptoms + symptom_count)
- **Hidden Layer 1**: 64 neurons + ReLU + Dropout (0.3)
- **Hidden Layer 2**: 32 neurons + ReLU + Dropout (0.3)
- **Output Layer**: 1 neuron (binary classification)

## 📁 Project Structure

```
pfl_deployment/
├── app.py                      # Streamlit web application
├── model.py                    # MLP neural network architecture
├── saved_models/               # Trained model weights
│   ├── abia_best_model.pt
│   ├── anambra_best_model.pt
│   ├── enugu_best_model.pt
│   ├── imo_best_model.pt
│   └── ebonyi_best_model.pt
├── config/
│   └── feature_names.json      # Input features configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/aonyezewe/pfl_es.git
   cd pfl_deployment
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running Locally

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Select a State**: Choose the state model from the dropdown
2. **Input Symptoms**: Select "Yes" or "No" for each symptom
3. **Run Prediction**: Click the "Run Prediction" button
4. **Review Results**: View the prediction outcome and risk probability

## 📊 Input Features

The model accepts 23 clinical symptoms:

- Backache
- Chills or sweats
- Fatigue/general weakness
- Fever
- Headache
- Vesiculopustular rash
- Rash lesions deep and profound
- All rash lesions the same size
- All rash lesions in same state of development
- Rash that itches
- Enlarged lymph nodes (axillary, cervical, inguinal)
- Muscle pain
- Oral ulcers
- Painful lymphadenitis
- Purpuric rash
- Maculopapular rash
- Sore throat/pharyngitis
- Swollen glands
- Feverish feeling
- Increased fatigue

The `symptom_count` feature is calculated automatically based on the number of "Yes" responses.

## 🎓 Training Details

### Hyperparameters

- **Rounds**: 30 federated learning rounds
- **Epochs per round**: 5
- **Batch size**: 32
- **Learning rate**: 0.01 (with evolutionary adaptation)
- **Dropout rate**: 0.3
- **Optimizer**: Adam with class-weighted BCE loss

### Evolutionary Strategy

Starting from round 6:
- Models are ranked by F1 score performance
- High-performing clients (top 75%) share their learning rates
- Low-performing clients (bottom 25%) adopt evolved hyperparameters
- Only high-performing models contribute to federated aggregation

## ⚠️ Limitations & Disclaimers

### Known Limitations

1. **Class Imbalance**: Training datasets had more negative cases than positive cases, which may affect prediction calibration
2. **Data Quality**: Model performance is dependent on the quality and representativeness of the training data
3. **Generalization**: Models are optimized for southeastern Nigerian populations and may not generalize to other regions

### Medical Disclaimer

⚕️ **This tool is for clinical decision support only and should NOT replace professional medical diagnosis.** 

- Always consult qualified healthcare providers for final diagnosis
- Laboratory confirmation is required for definitive monkeypox diagnosis
- Use predictions as one data point among many clinical factors

## 🔬 Technical Notes

### Class Imbalance Handling

The training data exhibited significant class imbalance (more negative cases). This was addressed through:
- Weighted loss function (BCEWithLogitsLoss with pos_weight)
- Stratified train-test splits
- Dropout regularization to prevent overfitting

### Model Selection

Best models were selected based on F1 score during validation across 30 training rounds. The saved models represent the optimal performance for each state.

## 📦 Dependencies

```
streamlit>=1.28.0
torch>=2.0.0
numpy>=1.24.0
```

## 🤝 Contributing

This project was developed as part of research on personalized federated learning for medical diagnostics. For questions or collaboration inquiries, please contact the project team.

## 👥 Authors

Anozie Onyezewe

## 🙏 Acknowledgments

- Healthcare facilities in Abia, Anambra, Enugu, Imo, and Ebonyi states for data collection
- Thesis Supervisors: Prof. Lala O.G., Dr. Iyanda A.R
- Department of Computer Science, Faculty of Science, Adeleke University, Ede

## 📧 Contact

For technical support or questions:
- Email: aonyezewe@gmail.com
- GitHub: https://github.com/aonyezewe

---

**Version**: 1.0.0  
**Last Updated**: January 2026
