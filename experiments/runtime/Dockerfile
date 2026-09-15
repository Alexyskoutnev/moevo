FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# Common local runtime. Model inference remains on the host's account-based
# Codex CLI; containers receive neither login credentials nor network access.
RUN python -m pip install --no-cache-dir \
    numpy==2.4.6 pandas==3.0.5 scipy==1.17.1 networkx==3.6.1 \
    pyomo==6.10.1 pyscipopt==6.2.1 \
    openpyxl==3.1.5 h5py==3.16.0 \
    sympy==1.14.0 math-verify==0.8.0 latex2sympy2_extended==1.10.2 \
    antlr4-python3-runtime==4.13.2 timeout-function-decorator==2.0.0 \
    openai==1.99.9 requests==2.32.5

ENV PYTHONDONTWRITEBYTECODE=1
ENV ORCLAW_SOLVE_TIME_LIMIT_SECONDS=300
ENV ORCLAW_SCIP_GAP=0.0005
WORKDIR /app
