#!/bin/bash
# Quick launcher for Study Assistant

cd "$(dirname "$0")"

echo "========================================"
echo "Study Assistant Launcher"
echo "========================================"
echo ""
echo "Choose an option:"
echo "  1) Ingest PDFs (run first, or after adding new PDFs)"
echo "  2) Start Web Interface (Streamlit)"
echo "  3) CLI Query Mode"
echo "  4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Running PDF ingestion..."
        python ingest.py
        ;;
    2)
        echo ""
        echo "Starting Streamlit web interface..."
        echo "Open http://localhost:8501 in your browser"
        streamlit run app.py
        ;;
    3)
        echo ""
        echo "Starting CLI query mode..."
        python query.py -i
        ;;
    4)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
