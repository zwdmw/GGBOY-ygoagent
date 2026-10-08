FROM python:3.11-slim
WORKDIR /opt/sky
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[runtime]'
COPY configs ./configs
ENV YGO_SKY_HOME=/opt/sky JAX_PLATFORMS=cpu
EXPOSE 8765
CMD ["ygo-sky", "serve", "--host", "0.0.0.0", "--port", "8765"]
