#!/bin/bash

# Configuration
PROJECT_NAME="omniprompt"
VENV_DIR=".venv-omniprompt"
PYTHON_BIN="$VENV_DIR/bin/python3"
PIP_BIN="$VENV_DIR/bin/pip"
PYTEST_BIN="$VENV_DIR/bin/pytest"

function setup() {
    echo "Setting up virtual environment..."
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install --upgrade pip
    pip install -e .
    pip install pytest ruff
}

function test() {
    echo "Running tests..."
    $PYTEST_BIN
}

function lint() {
    echo "Running linting (ruff)..."
    $PYTHON_BIN -m ruff check .
}

function start() {
    echo "Starting $PROJECT_NAME..."
    $PYTHON_BIN -m omniprompt.cli "$@"
}

function help() {
    echo "Usage: ./manage.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup   - Create venv and install dependencies"
    echo "  test    - Run tests"
    echo "  lint    - Run linting"
    echo "  start   - Run the CLI application"
    echo "  help    - Show this help message"
}

case "$1" in
    setup)
        setup
        ;;
    test)
        test
        ;;
    lint)
        lint
        ;;
    start)
        shift
        start "$@"
        ;;
    *)
        help
        ;;
esac
