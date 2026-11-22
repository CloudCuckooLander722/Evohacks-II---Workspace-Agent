FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt requirements-ui.txt ./
RUN pip install -r requirements.txt \
 && pip install -r requirements-ui.txt \
 && pip install "watchdog==7.0.0"

COPY agents ./agents
COPY tools ./tools
COPY workspace_agent ./workspace_agent
COPY adk_service ./adk_service

RUN useradd --create-home --uid 1000 app
USER app

ENV PORT=8080 \
    WORKSPACE_AGENT_LOAD_DOTENV=0
EXPOSE 8080

CMD ["sh", "-c", "exec streamlit run adk_service/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false"]
