FROM python:3.13-slim

# Hugging Face runs the container as uid 1000 and only /home/user and /tmp are
# writable, so both the install and the app have to sit under that user's home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Requirements first, app second: editing the page must not reinstall torch.
COPY --chown=user requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["python", "webapp/server.py", \
     "--res_dir", "results", "--fold", "1", "--samples", "samples", \
     "--device", "cpu", "--host", "0.0.0.0", "--port", "7860"]
