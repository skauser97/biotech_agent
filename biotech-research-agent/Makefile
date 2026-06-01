.PHONY: setup app test help

help:
	@echo "Biotech Research Agent"
	@echo ""
	@echo "  make setup   Install dependencies + copy .env"
	@echo "  make app     Launch Streamlit demo"
	@echo "  make test    Test each tool individually"

setup:
	pip install -r requirements.txt
	@[ -f .env ] || cp .env.example .env
	@echo "Edit .env and add your GROQ_API_KEY"

app:
	streamlit run app/streamlit_app.py

test:
	jupyter notebook notebooks/01_Tool_Testing.ipynb
