# Deployment Guide

## 🚀 Quick Deploy (Local)

### Prerequisites
- Python 3.7+
- Conda (optional but recommended)

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd AI_Coder_Metrics

# 2. Create virtual environment (optional)
conda create -n coder python=3.10
conda activate coder

# OR use venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the web application
python app.py

# 5. Open browser
# Navigate to: http://localhost:5000
```

## 🌐 Production Deployment

### Option 1: Heroku

1. **Create `Procfile`:**
```
web: gunicorn app:app
```

2. **Add gunicorn to requirements:**
```bash
echo "gunicorn>=20.1.0" >> requirements.txt
```

3. **Deploy:**
```bash
heroku create your-app-name
git push heroku main
```

### Option 2: AWS EC2

1. **Launch EC2 instance** (Ubuntu 20.04+)

2. **SSH into instance:**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

3. **Install dependencies:**
```bash
sudo apt update
sudo apt install python3-pip python3-venv nginx
```

4. **Clone and setup:**
```bash
git clone <repository-url>
cd AI_Coder_Metrics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
```

5. **Create systemd service** (`/etc/systemd/system/ai-coder.service`):
```ini
[Unit]
Description=AI Coder Evaluation
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/AI_Coder_Metrics
Environment="PATH=/home/ubuntu/AI_Coder_Metrics/venv/bin"
ExecStart=/home/ubuntu/AI_Coder_Metrics/venv/bin/gunicorn --workers 4 --bind 0.0.0.0:5000 app:app

[Install]
WantedBy=multi-user.target
```

6. **Start service:**
```bash
sudo systemctl start ai-coder
sudo systemctl enable ai-coder
```

7. **Configure Nginx** (`/etc/nginx/sites-available/ai-coder`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Increase upload size for large XML files
        client_max_body_size 100M;
    }
}
```

8. **Enable and restart Nginx:**
```bash
sudo ln -s /etc/nginx/sites-available/ai-coder /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Option 3: Docker

1. **Create `Dockerfile`:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

2. **Create `docker-compose.yml`:**
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./Benchmark:/app/Benchmark:ro
    restart: unless-stopped
```

3. **Build and run:**
```bash
docker-compose up -d
```

### Option 4: Google Cloud Run

1. **Create `app.yaml`:**
```yaml
runtime: python310
entrypoint: gunicorn --bind :$PORT app:app

instance_class: F2

env_variables:
  FLASK_ENV: production
```

2. **Deploy:**
```bash
gcloud app deploy
```

## ⚙️ Configuration

### Environment Variables

```bash
# Optional: Change host/port
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000

# Production mode
export FLASK_ENV=production

# Max upload size (default: 50MB)
export MAX_CONTENT_LENGTH=52428800
```

### Update `app.py` to use environment variables:

```python
import os

if __name__ == '__main__':
    app.run(
        debug=os.getenv('FLASK_ENV') != 'production',
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000))
    )
```

## 🔒 Security Considerations

### Production Checklist

- [ ] Set `debug=False` in production
- [ ] Use environment variables for sensitive config
- [ ] Implement rate limiting (Flask-Limiter)
- [ ] Add HTTPS (Let's Encrypt/CloudFlare)
- [ ] Configure CORS if needed
- [ ] Set appropriate file size limits
- [ ] Add authentication if needed (Flask-Login)
- [ ] Use gunicorn/uwsgi instead of Flask dev server
- [ ] Configure logging properly
- [ ] Set up monitoring (Sentry, DataDog)

### Optional: Add Authentication

```python
# Install: pip install flask-httpauth
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()

users = {
    "admin": "secure_password_here"
}

@auth.verify_password
def verify_password(username, password):
    if username in users and users[username] == password:
        return username

@app.route('/')
@auth.login_required
def index():
    return render_template('index.html')
```

## 📊 Performance Optimization

### For Large XML Files

1. **Increase worker timeouts:**
```bash
gunicorn --timeout 300 --workers 4 app:app
```

2. **Use Redis for caching** (optional):
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'redis'})
```

3. **Async processing** (for very large files):
```python
from celery import Celery

# Process evaluations in background
```

## 🐛 Troubleshooting

### Port already in use
```bash
# Find process
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 <PID>
```

### Permission denied
```bash
sudo chown -R $USER:$USER /path/to/AI_Coder_Metrics
```

### Large file upload fails
- Check `app.config['MAX_CONTENT_LENGTH']`
- Check nginx `client_max_body_size`
- Check system ulimits

## 📈 Monitoring

### Health Check Endpoint

Add to `app.py`:
```python
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': '1.0.0'}), 200
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## 🔄 Updates

### Update deployment:
```bash
git pull origin main
pip install -r requirements.txt
sudo systemctl restart ai-coder  # If using systemd
```

## 📞 Support

For issues, check:
1. Application logs
2. Web server logs (nginx/apache)
3. System logs (`journalctl -u ai-coder`)

---

Happy deploying! 🚀

