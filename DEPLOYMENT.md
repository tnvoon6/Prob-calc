# Streamlit Cloud Deployment Guide

Follow these steps to deploy your Game Probability Calculator to Streamlit Cloud for tablet access.

## Step 1: Prepare Your GitHub Repository

1. **Create a GitHub account** if you don't have one at github.com
2. **Create a new repository** named `game-probability-calculator`
3. **Upload these files** to your repository:
   - `app.py`
   - `probability_calculator.py`
   - `database.py`
   - `utils.py`
   - `streamlit_requirements.txt`
   - `.streamlit/config.toml`
   - `README.md`

## Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
2. **Sign in with GitHub**: Use your GitHub account to authenticate
3. **Create new app**:
   - Click "New app"
   - Select your repository: `your-username/game-probability-calculator`
   - Main file path: `app.py`
   - App URL: Choose a custom name like `your-name-probability-calc`

## Step 3: Configure Dependencies

Streamlit Cloud will automatically:
- Read `streamlit_requirements.txt` for Python packages
- Use `.streamlit/config.toml` for app configuration
- Install all required dependencies

## Step 4: Database Setup (Optional)

The app works without a database, but for session persistence:

### Option A: Free PostgreSQL (Neon.tech)
1. Sign up at [neon.tech](https://neon.tech) (free tier available)
2. Create a new database
3. Copy the connection string

### Option B: Free PostgreSQL (Supabase)
1. Sign up at [supabase.com](https://supabase.com) (free tier available)
2. Create a new project
3. Go to Settings > Database
4. Copy the connection string

### Add Database to Streamlit Cloud:
1. In your Streamlit Cloud app settings
2. Go to "Secrets"
3. Add this configuration:
```toml
[database]
url = "your-postgresql-connection-string-here"
```

## Step 5: Access Your App

After deployment (2-3 minutes):
- Your app will be available at: `https://your-app-name.streamlit.app`
- Access from any tablet browser
- Share the URL with others

## Features Available on Tablet

✅ **Full functionality**:
- W/L/D button inputs optimized for touch
- Interactive dual-track chart with zoom/pan
- Real-time probability calculations with confidence intervals
- Session management (if database configured)
- Accuracy tracking and analytics

✅ **Responsive design**:
- Works on iOS Safari, Android Chrome
- Touch-friendly interface
- Optimized for tablet screens

## Troubleshooting

**App won't start?**
- Check that `streamlit_requirements.txt` is in the root directory
- Verify all Python files are uploaded correctly

**Database features not working?**
- App works fine without database (no session persistence)
- Check database URL in Streamlit Cloud secrets
- Ensure database service is running

**Performance issues?**
- Streamlit Cloud free tier has resource limits
- Consider upgrading for heavy usage

## Cost Summary

- **Streamlit Cloud**: Free tier available
- **Database (Optional)**: Neon.tech and Supabase offer free tiers
- **Total**: $0 for basic usage

Your probability calculator will be accessible from any tablet with internet access!