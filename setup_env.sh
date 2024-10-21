#!/bin/bash

# Function to check if Python is installed
check_python() {
    if command -v python3 &>/dev/null; then
        echo "Python3 is installed."
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        echo "Python is installed."
        PYTHON_CMD="python"
    else
        echo "Python is not installed. Please install Python and try again."
        exit 1
    fi
}

# Function to check if pip is installed
check_pip() {
    if ! command -v pip &>/dev/null; then
        echo "pip is not installed. Attempting to install pip..."
        $PYTHON_CMD -m ensurepip --upgrade || {
            echo "Failed to install pip. Please install pip manually and try again."
            exit 1
        }
    else
        echo "pip is installed."
    fi
}

# Function to create and activate virtual environment
create_and_activate_venv() {
    # Create virtual environment
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv

    # Activate the virtual environment based on the OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Activating virtual environment on Linux..."
        source venv/bin/activate
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Activating virtual environment on macOS..."
        source venv/bin/activate
    elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" ]]; then
        echo "Activating virtual environment on Windows..."
        .\venv\Scripts\activate
    else
        echo "Unsupported OS. Cannot activate virtual environment."
        exit 1
    fi
}

# Function to install requirements
install_requirements() {
    if [ -f "requirements.txt" ]; then
        echo "Installing requirements from requirements.txt..."
        pip install -r requirements.txt
    else
        echo "No requirements.txt file found."
        exit 1
    fi
}

# Main script execution
check_python
check_pip
create_and_activate_venv
install_requirements

echo "Setup complete!"
