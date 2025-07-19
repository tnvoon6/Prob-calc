# Game Probability Calculator

A sophisticated Streamlit application that analyzes game history to predict the probability of winning, losing, or drawing the next hand. Features advanced statistical analysis including confidence intervals, session management, and interactive visualizations.

## Features

- **Three-Outcome Prediction**: Win/Loss/Draw probability forecasting
- **Confidence Intervals**: Statistical uncertainty bounds using Wilson score intervals
- **Interactive Visualization**: Dual-track chart with north/south stacking design
- **Session Management**: Save and load game sessions with PostgreSQL database
- **Accuracy Tracking**: Monitor prediction performance over time
- **Adaptive Learning System**: Automatically adjusts prediction weights based on historical accuracy
- **Real-time Analysis**: Multiple statistical methods including Markov chains
- **Performance Optimization**: Methods with higher accuracy get increased influence on predictions

## Live Demo

[Add your Streamlit Cloud URL here after deployment]

## Local Installation

```bash
pip install -r streamlit_requirements.txt
streamlit run app.py
```

## Database Setup

The app uses PostgreSQL for session storage. For local development, set the `DATABASE_URL` environment variable:

```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/gamedb"
```

For Streamlit Cloud deployment, add database credentials in the app settings.

## Usage

1. Click W/L/D buttons to add game results
2. View real-time probability predictions with confidence intervals
3. Save sessions for long-term analysis
4. Track prediction accuracy over time
5. Analyze patterns in the interactive chart

## Technology Stack

- **Frontend**: Streamlit with custom HTML/CSS styling
- **Backend**: Python with NumPy, Pandas, SciPy
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Visualization**: Plotly for interactive charts
- **Statistics**: Wilson score intervals, Markov chain analysis