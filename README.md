Supervised Machine Learning & Deep Learning Portfolio

Welcome to my portfolio submission for the machine learning internship tasks. This repository contains complete implementations for Task 1 (Supervised Machine Learning Comparison), Task 2 (Deep Learning Image Classification), and Task 3 (Production API Model Deployment).

📁 Repository Directory Structure

To keep the deliverables highly organized for evaluation, the repository is split into three distinct directories corresponding to each task, along with global environment files:

Artificial Intelligence Internship Tasks/
│
├── README.md                      # This Master Documentation Guide
│
├── Internspark task1.ipynb       # Task 1 Notebook
│
├── internspark task2.ipynb       # Task 2 Notebook
│
└── Internspark task 3/
    ├── app.py                     # Production FastAPI Application
    ├── requirements.txt           # Task 3 Local Dependency Schema
    ├── Dockerfile                 # Container Build Configuration
    └── test_api.py                # Automated Local API Test Suite


🛠️ Global Environment Setup

The projects are fully configured to run inside a local Python environment. For maximum convenience and continuity, all tasks utilize the active python (moodmate) virtual environment kernel created in previous project phases.

Prerequisites & Dependencies

To ensure that all classification packages, mathematical plotters, deep learning runtimes, and web framework servers compile correctly on your local machine, open your Command Prompt (Windows) or Terminal (macOS/Linux) and run the following command to install the required libraries:

# Using %pip inside Jupyter or running pip directly inside your active terminal environment:
pip install numpy pandas matplotlib seaborn scikit-learn fastapi uvicorn python-multipart Pillow torch torchvision requests


🏃‍♂️ Step-by-Step Execution Instructions

Please follow the commands below to run each individual task from your computer:

📊 Task 1: Supervised ML Classification Notebook

This task compares Logistic Regression (Linear Classifier) and Random Forest (Non-linear Ensemble) using a curved "Moon Clusters" synthetic dataset to evaluate non-linear boundaries.

Open your Terminal or Command Prompt and navigate to your task folder:

cd "Path/To/Your/Task_1_Supervised_ML"


Launch the Jupyter Notebook interface:

jupyter notebook


Set your active kernel in the Jupyter interface to your custom python (moodmate) environment.

Run all code cells sequentially (Cells 0 through 6) to generate the data, preprocess, scale, execute 5-Fold Cross-Validation, and render the comparative Confusion Matrices and ROC-AUC Curves.

🧠 Task 2: Deep Learning Image Classification Notebook

This task implements Transfer Learning on a pre-trained ResNet18 Convolutional Neural Network (CNN) to perform binary image classification (Airplanes vs. Automobiles) using a subset of the CIFAR-10 dataset.

Navigate to the Task 2 directory:

cd "Path/To/Your/Task_2_Deep_Learning"


Launch the Jupyter interface:

jupyter notebook


Open deep_learning_classification_assignment.ipynb and ensure the python (moodmate) environment kernel is selected.

Run all cells sequentially (Cells 0 through 7). Cell 0 will verify that PyTorch/Torchvision are correctly loaded, train the model head for 5 epochs on your CPU (optimized to take <2 minutes), plot learning curves, and save the final trained weights as binary_image_classifier.pth.

🚀 Task 3: Production Model Deployment (API)

This task hosts your saved weights from Task 2 inside a high-performance FastAPI web service. It contains a GET endpoint for health checks and a POST endpoint that accepts image uploads, pre-processes them, and returns JSON-formatted predictions.

Before running, ensure that your saved weights file (binary_image_classifier.pth) generated in Task 2 is copied into the Task_3_Deployment folder.

Running the API Locally:

Open a Command Prompt or Terminal and navigate to the Task 3 directory:

cd "Path/To/Your/Task_3_Deployment"


Launch the FastAPI Uvicorn web server:

python -m uvicorn app:app --reload


Open your web browser and navigate to http://localhost:8000/ to verify the server status. You should see a successful health status payload on the screen:
{"status": "healthy", "model_loaded": true, "classes": ["Airplane", "Automobile"]}

Testing Image Predictions:

Leave your FastAPI server window running in your first command prompt.

Open a second Command Prompt window, navigate to your folder, and run the automated test suite script:

py test_api.py


The client test script will dynamically generate a mock image, upload it to the live server, and print a successful Response Status Code: 200 along with the model's classifications:

{
  "prediction": "Automobile",
  "confidence": 0.5176,
  "probabilities": {
    "Airplane": 0.4824,
    "Automobile": 0.5176
  }
}


Deploying via Docker (Optional Containerization):

If you have Docker Desktop installed, you can package the microservice into an isolated container:

Build the lightweight container image (using PyTorch CPU wheels to keep the footprint under ~700MB):

docker build -t ml-classifier-api .


Run the isolated container mapping port 8000:

docker run -p 8000:8000 ml-classifier-api


Run the client test script in a separate window to confirm successful execution:

python test_api.py


💾 Large Deliverables (Model Weights)

Due to the size limitations of hosting large trained neural network files directly inside a standard Git repository, the compiled model weights file (binary_image_classifier.pth) is securely hosted on Google Drive.

Public Google Drive Weights Link: https://drive.google.com/file/d/1LmIWK72KMUeMGv0BgZFivw3dWTBeF5jI/view?usp=sharing

This weights file has been verified, configured with read permissions for "Anyone with the link," and is used by the FastAPI backend to perform prediction mappings.
