# Launch the dashboard without relying on streamlit.exe being on PATH.
Set-Location $PSScriptRoot
python -m streamlit run app.py
