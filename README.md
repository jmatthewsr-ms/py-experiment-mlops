# Data Science Interview Project

This repository contains a simple logistic regression model demonstration for a Principal Engineer data science interview.

## Setup Instructions

### Checking for pip

Before starting, ensure pip is installed:

1. Check if pip is installed:
   ```
   pip --version
   ```

2. If pip is not installed, install it:
   - On Windows:
     ```
     python -m ensurepip --default-pip
     ```
   - On macOS/Linux:
     ```
     python -m ensurepip --default-pip
     # Or use your system's package manager:
     # sudo apt-get install python3-pip  # For Ubuntu/Debian
     # sudo yum install python3-pip      # For CentOS/RHEL
     ```

### Using Python's Virtual Environment (venv)

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

## Running the Notebook

Open `principal_engineer_interview.ipynb` in Jupyter to view and run the logistic regression demonstration.

## Quick Setup

For convenience, you can use the setup scripts:
- Windows: Run `setup_venv.bat`
- macOS/Linux: Run `bash setup_venv.sh`
