# 🚀 Deployment Guide

## Quick Deploy to Render

### 1. Prerequisites
- GitHub account with this repository
- [Render](https://render.com) account (free tier available)

### 2. Render Configuration

1. **Create New Web Service** on Render
   - Connect your GitHub repository
   - Select this repository

2. **Configure Settings**
   - **Name**: `ai-coder-evaluation` (or your choice)
   - **Environment**: `Python 3`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your branch name)
   - **Build Command**: 
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```bash
     python -m gunicorn app:app --bind 0.0.0.0:$PORT
     ```

3. **Environment Variables** (IMPORTANT)
   - Go to **Environment** tab
   - Add variable:
     - **Key**: `SECRET_KEY`
     - **Value**: Generate with: `python -c "import secrets; print(secrets.token_hex(32))"`
   
4. **Advanced Settings** (Optional)
   - **Health Check Path**: `/`
   - **Auto-Deploy**: Enable for `main` branch

5. **Deploy**
   - Click **"Create Web Service"**
   - Wait 3-5 minutes for build
   - Access at: `https://your-app-name.onrender.com`

### 3. Post-Deployment

✅ **Test Your App**:
1. Upload `ModelOutput/Kids_Meal_example.xml`
2. Select "Kids Meal" benchmark
3. Evaluate and verify metrics display
4. Test CSV download

⚠️ **Free Tier Note**: 
- App sleeps after 15 minutes of inactivity
- First request after sleep takes ~30 seconds

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Access at http://localhost:5000
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SECRET_KEY` | Flask session encryption key | Yes |
| `PORT` | Port to run on (auto-set by Render) | No |

---

## Troubleshooting

**"gunicorn: command not found" error?**
- ✅ **Solution 1** (Recommended): Update Start Command to:
  ```bash
  python -m gunicorn app:app --bind 0.0.0.0:$PORT
  ```
- ✅ **Solution 2** (Alternative): Use Flask's built-in server:
  ```bash
  python app.py
  ```
  ⚠️ Note: Flask server works but gunicorn is better for production
- Check that Build Command completed successfully in logs
- Verify `gunicorn>=21.2.0` is in `requirements.txt`

**Build fails?**
- Check Python version (3.7+ required, 3.11 recommended)
- Verify `requirements.txt` exists
- Check build logs for specific errors
- Try: `pip install --upgrade pip && pip install -r requirements.txt`

**App crashes on start?**
- Check logs: `Render Dashboard → Logs`
- Verify `SECRET_KEY` is set in Environment Variables
- Ensure port binding is correct: `--bind 0.0.0.0:$PORT`

**CSV download doesn't work?**
- Must run evaluation first
- Check browser console for errors
- Verify session is working (SECRET_KEY set)

---

## Support

- [Render Documentation](https://render.com/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- Check repository README.md for usage guide
