import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from probability_calculator import ProbabilityCalculator
from utils import validate_history, format_probability
from database import (
    init_database, save_game_session, load_game_session, 
    get_all_sessions, delete_game_session, get_session_statistics,
    save_prediction_record, get_prediction_accuracy, get_all_prediction_accuracy
)

# Page configuration
st.set_page_config(
    page_title="Game Probability Calculator",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for better styling and reduced font sizes
st.markdown("""
<style>
/* Reduce font sizes globally */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

/* Header styling */
h1 {
    font-size: 1.8rem !important;
    margin-bottom: 0.5rem !important;
    color: #1f77b4;
}

h2 {
    font-size: 1.4rem !important;
    margin-bottom: 0.3rem !important;
    margin-top: 0.5rem !important;
}

h3 {
    font-size: 1.2rem !important;
    margin-bottom: 0.2rem !important;
    margin-top: 0.3rem !important;
}

/* Button styling */
.stButton > button {
    font-size: 0.9rem !important;
    padding: 0.3rem 0.6rem !important;
    font-weight: bold !important;
    border-radius: 8px !important;
}

/* Metric styling */
[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    padding: 0.5rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

[data-testid="metric-container"] label {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1.1rem !important;
    font-weight: bold !important;
}

/* Text area styling */
.stTextArea textarea {
    font-size: 0.8rem !important;
    font-family: 'Courier New', monospace !important;
}

/* Input styling */
.stTextInput input {
    font-size: 0.9rem !important;
}

/* Info boxes styling */
.stAlert {
    font-size: 0.85rem !important;
    padding: 0.5rem !important;
    margin: 0.25rem 0 !important;
}

/* Sidebar styling */
.css-1d391kg {
    padding-top: 1rem !important;
}

.sidebar .sidebar-content {
    font-size: 0.85rem !important;
}

/* DataFrame styling */
.stDataFrame {
    font-size: 0.8rem !important;
}

/* Expandable section styling */
.streamlit-expanderHeader {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

/* Success/Error/Warning message styling */
.stSuccess, .stError, .stWarning, .stInfo {
    font-size: 0.8rem !important;
    padding: 0.4rem !important;
}

/* Custom instruction text styling */
.instruction-text {
    font-size: 0.8rem !important;
    color: #666 !important;
    margin: 0.3rem 0 !important;
}

/* Placeholder text styling */
.stTextInput input::placeholder {
    font-size: 0.8rem !important;
    color: #999 !important;
}

/* Chart container styling */
.js-plotly-plot {
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

/* Custom compact spacing */
.compact-metric {
    margin: 0.1rem 0 !important;
}

/* Table styling */
.stDataFrame table {
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def initialize_database():
    return init_database()

# Initialize session state
if 'game_history' not in st.session_state:
    st.session_state.game_history = []
if 'calculator' not in st.session_state:
    st.session_state.calculator = ProbabilityCalculator()
if 'current_session_name' not in st.session_state:
    st.session_state.current_session_name = ""
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'prediction_pending' not in st.session_state:
    st.session_state.prediction_pending = False
if 'show_save_dialog' not in st.session_state:
    st.session_state.show_save_dialog = False
if 'prediction_correct' not in st.session_state:
    st.session_state.prediction_correct = 0
if 'prediction_wrong' not in st.session_state:
    st.session_state.prediction_wrong = 0

# Initialize database
db_initialized = initialize_database()

# Title and description
st.markdown("<h1 style='font-size: 1.8rem; margin-bottom: 0.5rem;'>🎯 Game Probability Calculator</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 0.9rem; color: #666; margin-bottom: 1rem;'><strong>Analyze game history to predict next hand probabilities</strong> • Win/Loss/Draw forecasting with accuracy tracking</p>", unsafe_allow_html=True)

# Sidebar for input methods and session management
st.sidebar.header("Session Management")

if db_initialized:
    # Session name input
    session_name = st.sidebar.text_input(
        "Session Name:",
        value=st.session_state.current_session_name,
        placeholder="Enter session name to save/load"
    )
    
    # Save/Load buttons
    col_save, col_load = st.sidebar.columns([1, 1])
    
    with col_save:
        if st.button("💾 Save Session", disabled=not session_name or not st.session_state.game_history):
            if save_game_session(session_name, st.session_state.game_history):
                st.session_state.current_session_name = session_name
                st.success(f"Session '{session_name}' saved!")
                st.rerun()
    
    with col_load:
        if st.button("📂 Load Session", disabled=not session_name):
            loaded_data = load_game_session(session_name)
            if loaded_data:
                st.session_state.game_history = loaded_data['game_history']
                st.session_state.current_session_name = session_name
                st.success(f"Session '{session_name}' loaded!")
                st.rerun()
            else:
                st.error(f"Session '{session_name}' not found!")
    
    # Saved sessions list
    st.sidebar.subheader("Saved Sessions")
    saved_sessions = get_all_sessions()
    
    if saved_sessions:
        for session in saved_sessions[:5]:  # Show last 5 sessions
            session_info = f"{session['name']} ({session['total_games']} games)"
            if st.sidebar.button(session_info, key=f"load_{session['name']}"):
                loaded_data = load_game_session(session['name'])
                if loaded_data:
                    st.session_state.game_history = loaded_data['game_history']
                    st.session_state.current_session_name = session['name']
                    st.rerun()
        
        if len(saved_sessions) > 5:
            st.sidebar.text(f"...and {len(saved_sessions) - 5} more sessions")
    else:
        st.sidebar.text("No saved sessions")
    
    st.sidebar.markdown("---")

st.sidebar.header("Quick Actions")
if st.sidebar.button("🔄 Generate Random Pattern"):
    pattern = np.random.choice([1, 0, 2], size=20, p=[1/3, 1/3, 1/3])
    st.session_state.game_history = pattern.tolist()
    st.rerun()

# Auto-save dialog when 50 entries reached
if st.session_state.show_save_dialog and len(st.session_state.game_history) == 50:
    st.error("🔒 **Session Complete!** You've reached the 50-entry limit.")
    st.info("Please save this session to continue with a new one:")
    
    # Initialize session name in session state if not exists
    if 'auto_save_session_name' not in st.session_state:
        st.session_state.auto_save_session_name = ""
    
    # Session name input with form
    with st.form("save_session_form"):
        st.session_state.auto_save_session_name = st.text_input(
            "Enter Session Name:",
            value=st.session_state.auto_save_session_name,
            placeholder="e.g., 'Evening Games', 'Tournament Round 1'",
            help="Give this 50-game session a memorable name"
        )
        
        save_col1, save_col2 = st.columns([1, 1])
        
        with save_col1:
            save_submitted = st.form_submit_button("💾 Save Session", type="primary", use_container_width=True)
        
        with save_col2:
            discard_submitted = st.form_submit_button("🗑️ Discard & Start New", type="secondary", use_container_width=True)
    
    # Handle form submissions
    if save_submitted:
        if st.session_state.auto_save_session_name.strip():
            if db_initialized:
                try:
                    save_game_session(st.session_state.auto_save_session_name.strip(), st.session_state.game_history)
                    st.success(f"✅ Session '{st.session_state.auto_save_session_name.strip()}' saved successfully!")
                    
                    # Reset for new session
                    st.session_state.game_history = []
                    st.session_state.show_save_dialog = False
                    st.session_state.prediction_pending = False
                    st.session_state.current_session_name = ""
                    st.session_state.auto_save_session_name = ""
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving session: {str(e)}")
            else:
                st.error("❌ Database not available")
        else:
            st.error("⚠️ Please enter a session name")
    
    if discard_submitted:
        st.session_state.game_history = []
        st.session_state.show_save_dialog = False
        st.session_state.prediction_pending = False
        st.session_state.current_session_name = ""
        st.session_state.auto_save_session_name = ""
        st.warning("🗑️ Session discarded. Starting fresh.")
        st.rerun()

# Main content area - Input and Forecast on same page
st.header("Game Input & Forecast")

# Add Game Results Section

input_col1, input_col2, input_col3, input_col4 = st.columns([1, 1, 1, 1])
    
with input_col1:
    if st.button("**W**", help="Add Win", use_container_width=True):
        if len(st.session_state.game_history) < 50:
            # Check if we had a pending prediction and update counters
            if st.session_state.prediction_pending and st.session_state.last_prediction:
                # Determine what was predicted as most likely
                win_prob = st.session_state.last_prediction['win']['weighted_average']
                loss_prob = st.session_state.last_prediction['loss']['weighted_average']
                draw_prob = st.session_state.last_prediction['draw']['weighted_average']
                
                max_prob = max(win_prob, loss_prob, draw_prob)
                predicted_outcome = 'win' if max_prob == win_prob else ('draw' if max_prob == draw_prob else 'loss')
                
                # Update prediction counters
                if predicted_outcome == 'win':  # Predicted win, actual win
                    st.session_state.prediction_correct += 1
                else:  # Predicted loss/draw, actual win
                    st.session_state.prediction_wrong += 1

                
                # Save prediction record if we have a session name
                if st.session_state.current_session_name and db_initialized:
                    try:
                        save_prediction_record(
                            st.session_state.current_session_name,
                            len(st.session_state.game_history),
                            st.session_state.last_prediction,
                            1  # Win
                        )
                    except Exception:
                        pass  # Database error - continue without saving
                st.session_state.prediction_pending = False
            
            st.session_state.game_history.append(1)
            
            # Auto-save when reaching 50 entries
            if len(st.session_state.game_history) == 50:
                st.session_state.show_save_dialog = True
            
            st.rerun()
        else:
            st.warning("Session complete (50 hands). Save current session to start a new one.")

with input_col2:
    if st.button("**L**", help="Add Loss", use_container_width=True):
        if len(st.session_state.game_history) < 50:
            # Check if we had a pending prediction and update counters
            if st.session_state.prediction_pending and st.session_state.last_prediction:
                # Determine what was predicted as most likely
                win_prob = st.session_state.last_prediction['win']['weighted_average']
                loss_prob = st.session_state.last_prediction['loss']['weighted_average']
                draw_prob = st.session_state.last_prediction['draw']['weighted_average']
                
                max_prob = max(win_prob, loss_prob, draw_prob)
                predicted_outcome = 'win' if max_prob == win_prob else ('draw' if max_prob == draw_prob else 'loss')
                
                # Update prediction counters
                if predicted_outcome == 'loss':  # Predicted loss, actual loss
                    st.session_state.prediction_correct += 1
                else:  # Predicted win/draw, actual loss
                    st.session_state.prediction_wrong += 1
                
                # Save prediction record if we have a session name
                if st.session_state.current_session_name and db_initialized:
                    try:
                        save_prediction_record(
                            st.session_state.current_session_name,
                            len(st.session_state.game_history),
                            st.session_state.last_prediction,
                            0  # Loss
                        )
                    except Exception:
                        pass  # Database error - continue without saving
                st.session_state.prediction_pending = False
            
            st.session_state.game_history.append(0)
            
            # Auto-save when reaching 50 entries
            if len(st.session_state.game_history) == 50:
                st.session_state.show_save_dialog = True
            
            st.rerun()
        else:
            st.warning("Session complete (50 hands). Save current session to start a new one.")

with input_col3:
    if st.button("**D**", help="Add Draw", use_container_width=True):
        if len(st.session_state.game_history) < 50:
            # Check if we had a pending prediction and update counters
            if st.session_state.prediction_pending and st.session_state.last_prediction:
                # Determine what was predicted as most likely
                win_prob = st.session_state.last_prediction['win']['weighted_average']
                loss_prob = st.session_state.last_prediction['loss']['weighted_average']
                draw_prob = st.session_state.last_prediction['draw']['weighted_average']
                
                max_prob = max(win_prob, loss_prob, draw_prob)
                predicted_outcome = 'win' if max_prob == win_prob else ('draw' if max_prob == draw_prob else 'loss')
                
                # Update prediction counters
                if predicted_outcome == 'draw':  # Predicted draw, actual draw
                    st.session_state.prediction_correct += 1
                else:  # Predicted win/loss, actual draw
                    st.session_state.prediction_wrong += 1
                
                # Save prediction record if we have a session name
                if st.session_state.current_session_name and db_initialized:
                    try:
                        save_prediction_record(
                            st.session_state.current_session_name,
                            len(st.session_state.game_history),
                            st.session_state.last_prediction,
                            2  # Draw
                        )
                    except Exception:
                        pass  # Database error - continue without saving
                st.session_state.prediction_pending = False
            
            st.session_state.game_history.append(2)
            
            # Auto-save when reaching 50 entries
            if len(st.session_state.game_history) == 50:
                st.session_state.show_save_dialog = True
            
            st.rerun()
        else:
            st.warning("Session complete (50 hands). Save current session to start a new one.")

with input_col4:
    if st.button("↩️", help="Undo Last", use_container_width=True) and st.session_state.game_history:
        st.session_state.game_history.pop()
        st.session_state.prediction_pending = False  # Reset prediction tracking
        st.rerun()

# Create main layout with game history and forecast side by side
main_col1, main_col2 = st.columns([3, 2])

@st.cache_data
def create_dual_track_plotly_chart(history):
    """Create a dual-track chart showing all 50 columns, populated as games are played"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create single plot with center line
    fig = go.Figure()
    
    # Colors
    win_color = '#28a745'
    loss_color = '#dc3545'
    draw_color = '#fd7e14'
    
    # Plot symbols for each game position
    win_x = []
    win_y = []
    
    loss_x = []
    loss_y = []
    
    # Separate lists for X symbols (draws)
    draw_x = []
    draw_y = []
    draw_track = []  # 1 for win track, 2 for loss track
    
    # Process the actual game history - track when result type changes
    if history:
        current_position = 0
        last_result_type = None  # Track the last win/loss (not draw)
        
        for i, result in enumerate(history):
            if result == 2:  # Draw - show if following a win/loss or another draw
                previous_result = history[i-1] if i > 0 else None
                
                # Skip draws only if they are at the very beginning
                if previous_result is None:
                    continue
                
                # For consecutive draws, find the original win/loss that started the sequence
                original_result = None
                j = i - 1
                while j >= 0 and history[j] == 2:
                    j -= 1
                if j >= 0:
                    original_result = history[j]
                
                # Draw goes in the same column, stacking based on the original win/loss
                if original_result == 1 or (previous_result == 1):  # Stack on north side (win side)
                    # Find all positions in current column
                    all_positions = [(x, y) for x, y in zip(win_x + loss_x + draw_x, win_y + loss_y + draw_y) if x == current_position]
                    if all_positions:
                        max_height = max([y for x, y in all_positions if y > 0], default=0)
                        draw_x.append(current_position)
                        draw_y.append(max_height + 4)
                    else:
                        draw_x.append(current_position) 
                        draw_y.append(8)  # On top of win at 4
                    draw_track.append(1)
                    
                elif original_result == 0 or (previous_result == 0):  # Stack on south side (loss side)
                    # Find all positions in current column
                    all_positions = [(x, y) for x, y in zip(win_x + loss_x + draw_x, win_y + loss_y + draw_y) if x == current_position]
                    if all_positions:
                        min_depth = min([y for x, y in all_positions if y < 0], default=0)
                        draw_x.append(current_position)
                        draw_y.append(min_depth - 4)
                    else:
                        draw_x.append(current_position)
                        draw_y.append(-8)  # Below loss at -4
                    draw_track.append(2)
                    
            elif result == 1:  # Win
                # Check if we need to advance column (when switching from loss to win)
                if last_result_type == 0:  # Previous was loss - start new column
                    current_position += 1
                
                # Find all positions in current column
                all_positions = [(x, y) for x, y in zip(win_x + loss_x + draw_x, win_y + loss_y + draw_y) if x == current_position]
                if all_positions:
                    max_height = max([y for x, y in all_positions if y > 0], default=0)
                    win_x.append(current_position)
                    win_y.append(max_height + 4)
                else:
                    win_x.append(current_position)
                    win_y.append(4)
                
                last_result_type = 1  # Update last result type
                    
            elif result == 0:  # Loss
                # Check if we need to advance column (when switching from win to loss)
                if last_result_type == 1:  # Previous was win - start new column
                    current_position += 1
                
                # Find all positions in current column  
                all_positions = [(x, y) for x, y in zip(win_x + loss_x + draw_x, win_y + loss_y + draw_y) if x == current_position]
                if all_positions:
                    min_depth = min([y for x, y in all_positions if y < 0], default=0)
                    loss_x.append(current_position)
                    loss_y.append(min_depth - 4)
                else:
                    loss_x.append(current_position)
                    loss_y.append(-4)
                
                last_result_type = 0  # Update last result type
    
    # Add center line
    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=2)
    
    # Add winning symbols (north of center line)
    if win_x:
        fig.add_trace(
            go.Scatter(
                x=win_x,
                y=win_y,
                mode='markers',
                marker=dict(
                    size=8,
                    color='blue',
                    symbol='circle-open',
                    line=dict(width=2, color='blue')
                ),
                name='Wins',
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Add losing symbols (south of center line)
    if loss_x:
        fig.add_trace(
            go.Scatter(
                x=loss_x,
                y=loss_y,
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='circle-open',
                    line=dict(width=2, color='red')
                ),
                name='Losses',
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Add draw symbols
    if draw_x:
        fig.add_trace(
            go.Scatter(
                x=draw_x,
                y=draw_y,
                mode='markers',
                marker=dict(
                    size=8,
                    color='black',
                    symbol='x'
                ),
                name='Draws',
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Update layout to use full container width with optimizations
    fig.update_layout(
        height=350,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=20, t=50, b=40),
        dragmode=False,  # Disable interactions for better performance
        hovermode=False
    )
    
    # Set x-axis to show all 50 positions
    fig.update_xaxes(
        range=[-1, 50],  # Show columns 0-49 with padding
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1,
        zeroline=False,
        showticklabels=True,
        tick0=0,
        dtick=5,  # Show every 5th column number
        title_text="Game Position"
    )
    
    # Set y-axis with center line at 0 and bigger row spacing
    fig.update_yaxes(
        range=[-40, 40],  # Expanded range for more spacing
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='gray',
        zerolinewidth=2,
        showticklabels=True,
        dtick=8,  # Show labels every 8 units for clarity
        tickfont=dict(size=10),
        title_text="Wins ↑ Center Line ↓ Losses",
        title_font=dict(size=11)
    )
    
    return fig

with main_col1:
    # Game History Visualization
    st.markdown(f"**Game History ({len(st.session_state.game_history)}/50)**")
    
    # Display the chart
    # Convert to tuple for caching (lists aren't hashable)
    history_tuple = tuple(st.session_state.game_history) if st.session_state.game_history else ()
    fig = create_dual_track_plotly_chart(history_tuple)
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
    
    # Display session info and quick stats
    info_col1, info_col2 = st.columns([1, 1])
    
    with info_col1:
        if st.session_state.current_session_name:
            st.info(f"**Session:** {st.session_state.current_session_name}")
    
    with info_col2:
        pass

with main_col2:
    # Next Game Forecast Section
    st.markdown("<h3 style='font-size: 1.1rem; margin-bottom: 0.5rem;'>Next Game Forecast</h3>", unsafe_allow_html=True)
    
    if st.session_state.game_history and len(st.session_state.game_history) >= 3:
        # Update calculator weights based on prediction accuracy
        calculator = st.session_state.calculator
        
        # Get overall accuracy data to improve predictions
        if db_initialized:
            try:
                overall_accuracy = get_all_prediction_accuracy()
                if overall_accuracy:
                    calculator.update_method_weights(overall_accuracy)
            except Exception as e:
                # Database connection error - continue without weight updates
                pass
        
        # Calculate probabilities with updated weights
        results = calculator.calculate_probabilities(st.session_state.game_history)
        
        # Store prediction for accuracy tracking
        st.session_state.last_prediction = results
        st.session_state.prediction_pending = True  # Always set pending when we have a prediction
        
        # Display main probabilities
        prob_col1, prob_col2, prob_col3 = st.columns([1, 1, 1])
        
        with prob_col1:
            win_prob = results['win']['weighted_average']
            win_ci = results['win']['confidence_interval']
            st.markdown(f"<div style='text-align: center; padding: 0.3rem; border: 1px solid #ddd; border-radius: 6px; background: white;'><div style='font-size: 0.7rem; color: #666; margin-bottom: 0.1rem;'>🏆 Win</div><div style='font-size: 0.85rem; font-weight: bold;'>{win_prob:.1%}</div><div style='font-size: 0.6rem; color: #888; margin-top: 0.1rem;'>CI: {win_ci['lower']:.1%}-{win_ci['upper']:.1%}</div></div>", unsafe_allow_html=True)
        
        with prob_col2:
            loss_prob = results['loss']['weighted_average']
            loss_ci = results['loss']['confidence_interval']
            st.markdown(f"<div style='text-align: center; padding: 0.3rem; border: 1px solid #ddd; border-radius: 6px; background: white;'><div style='font-size: 0.7rem; color: #666; margin-bottom: 0.1rem;'>❌ Loss</div><div style='font-size: 0.85rem; font-weight: bold;'>{loss_prob:.1%}</div><div style='font-size: 0.6rem; color: #888; margin-top: 0.1rem;'>CI: {loss_ci['lower']:.1%}-{loss_ci['upper']:.1%}</div></div>", unsafe_allow_html=True)
        
        with prob_col3:
            draw_prob = results['draw']['weighted_average']
            draw_ci = results['draw']['confidence_interval']
            st.markdown(f"<div style='text-align: center; padding: 0.3rem; border: 1px solid #ddd; border-radius: 6px; background: white;'><div style='font-size: 0.7rem; color: #666; margin-bottom: 0.1rem;'>🤝 Draw</div><div style='font-size: 0.85rem; font-weight: bold;'>{draw_prob:.1%}</div><div style='font-size: 0.6rem; color: #888; margin-top: 0.1rem;'>CI: {draw_ci['lower']:.1%}-{draw_ci['upper']:.1%}</div></div>", unsafe_allow_html=True)
        
        # Most likely outcome
        max_prob = max(win_prob, loss_prob, draw_prob)
        if max_prob == win_prob:
            st.markdown(f"<div style='background: #d4edda; color: #155724; padding: 0.4rem; border-radius: 6px; font-size: 0.8rem; text-align: center; margin: 0.3rem 0;'><strong>Most Likely: WIN</strong> ({win_prob:.1%})</div>", unsafe_allow_html=True)
        elif max_prob == draw_prob:
            st.markdown(f"<div style='background: #fff3cd; color: #856404; padding: 0.4rem; border-radius: 6px; font-size: 0.8rem; text-align: center; margin: 0.3rem 0;'><strong>Most Likely: DRAW</strong> ({draw_prob:.1%})</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background: #f8d7da; color: #721c24; padding: 0.4rem; border-radius: 6px; font-size: 0.8rem; text-align: center; margin: 0.3rem 0;'><strong>Most Likely: LOSS</strong> ({loss_prob:.1%})</div>", unsafe_allow_html=True)
        
        # Confidence interval explanation
        st.markdown("<div style='font-size: 0.65rem; color: #666; text-align: center; margin: 0.3rem 0; font-style: italic;'>95% Confidence Intervals show the likely range for each probability</div>", unsafe_allow_html=True)
        
        # Learning system indicator
        if db_initialized:
            overall_accuracy = get_all_prediction_accuracy()
            if overall_accuracy and overall_accuracy['total_predictions'] > 10:
                st.markdown(f"<div style='background: #e7f3ff; color: #0066cc; padding: 0.3rem; border-radius: 6px; font-size: 0.7rem; text-align: center; margin: 0.2rem 0;'><strong>🧠 Learning Active:</strong> Weights optimized from {overall_accuracy['total_predictions']} predictions</div>", unsafe_allow_html=True)
                
                # Show current method weights (expandable)
                with st.expander("📊 Current Method Weights", expanded=False):
                    st.markdown("<div style='font-size: 0.8rem; color: #666; margin-bottom: 0.5rem;'>The system adapts these weights based on historical accuracy:</div>", unsafe_allow_html=True)
                    
                    weight_col1, weight_col2 = st.columns([1, 1])
                    with weight_col1:
                        st.metric("Simple Frequency", f"{calculator.method_weights['simple_frequency']:.1%}", help="Basic win/loss/draw ratios")
                        st.metric("Streak Adjusted", f"{calculator.method_weights['streak_adjusted']:.1%}", help="Pattern-based adjustments")
                    with weight_col2:
                        st.metric("Recent Form", f"{calculator.method_weights['recent_form']:.1%}", help="Last 10 games emphasis")
                        st.metric("Markov Chain", f"{calculator.method_weights['markov_chain']:.1%}", help="Transition analysis")
        
        # Current session accuracy if available
        if st.session_state.current_session_name:
            session_accuracy = get_prediction_accuracy(st.session_state.current_session_name)
            if session_accuracy:
                st.markdown(f"<div style='background: #d1ecf1; color: #0c5460; padding: 0.3rem; border-radius: 6px; font-size: 0.75rem; text-align: center; margin: 0.2rem 0;'><strong>Session Accuracy:</strong> {session_accuracy['overall_accuracy']:.1%} ({session_accuracy['correct_predictions']}/{session_accuracy['total_predictions']})</div>", unsafe_allow_html=True)
                
                # Quick accuracy chart for current session
                if session_accuracy['recent_records'] and len(session_accuracy['recent_records']) >= 3:
                    accuracy_series = [1 if record.was_correct else 0 for record in session_accuracy['recent_records'][-10:]]
                    colors = ['green' if x == 1 else 'red' for x in accuracy_series]
                    
                    fig_mini = go.Figure(data=[go.Bar(
                        x=list(range(1, len(accuracy_series) + 1)),
                        y=accuracy_series,
                        marker_color=colors,
                        name="Recent Accuracy"
                    )])
                    
                    fig_mini.update_layout(
                        title="Recent Accuracy (Last 10)",
                        height=200,
                        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Wrong', 'Correct']),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_mini, use_container_width=True)
    else:
        st.markdown("<p style='font-size: 0.8rem; color: #666; font-style: italic;'>Add at least 3 games to see forecast</p>", unsafe_allow_html=True)
    
    # Game statistics
    if st.session_state.game_history:
        # Quick stats
        wins = st.session_state.game_history.count(1)
        losses = st.session_state.game_history.count(0)
        draws = st.session_state.game_history.count(2)
        
        st.markdown("<div style='font-size: 0.7rem; color: #666; margin: 0.3rem 0; text-align: center;'>Game Totals</div>", unsafe_allow_html=True)
        stat_col1, stat_col2, stat_col3 = st.columns([1, 1, 1])
        with stat_col1:
            st.markdown(f"<div style='text-align: center; padding: 0.2rem; border: 1px solid #ddd; border-radius: 4px; background: white;'><div style='font-size: 0.6rem; color: #666; margin-bottom: 0.1rem;'>W</div><div style='font-size: 0.75rem; font-weight: bold;'>{wins}</div></div>", unsafe_allow_html=True)
        with stat_col2:
            st.markdown(f"<div style='text-align: center; padding: 0.2rem; border: 1px solid #ddd; border-radius: 4px; background: white;'><div style='font-size: 0.6rem; color: #666; margin-bottom: 0.1rem;'>L</div><div style='font-size: 0.75rem; font-weight: bold;'>{losses}</div></div>", unsafe_allow_html=True)
        with stat_col3:
            st.markdown(f"<div style='text-align: center; padding: 0.2rem; border: 1px solid #ddd; border-radius: 4px; background: white;'><div style='font-size: 0.6rem; color: #666; margin-bottom: 0.1rem;'>D</div><div style='font-size: 0.75rem; font-weight: bold;'>{draws}</div></div>", unsafe_allow_html=True)
        
        # Prediction Success Counter - Only show if we have predictions
        total_predictions = st.session_state.prediction_correct + st.session_state.prediction_wrong
        if total_predictions > 0:
            st.markdown("<div style='font-size: 0.7rem; color: #666; margin: 0.4rem 0 0.2rem 0; text-align: center;'>Prediction Success</div>", unsafe_allow_html=True)
            
            pred_col1, pred_col2 = st.columns([1, 1])
            with pred_col1:
                st.markdown(f"<div style='text-align: center; padding: 0.2rem; border: 1px solid #28a745; border-radius: 4px; background: #f8fff8;'><div style='font-size: 0.6rem; color: #28a745; margin-bottom: 0.1rem;'>✓ Correct</div><div style='font-size: 0.75rem; font-weight: bold; color: #28a745;'>{st.session_state.prediction_correct}</div></div>", unsafe_allow_html=True)
            with pred_col2:
                st.markdown(f"<div style='text-align: center; padding: 0.2rem; border: 1px solid #dc3545; border-radius: 4px; background: #fff8f8;'><div style='font-size: 0.6rem; color: #dc3545; margin-bottom: 0.1rem;'>✗ Wrong</div><div style='font-size: 0.75rem; font-weight: bold; color: #dc3545;'>{st.session_state.prediction_wrong}</div></div>", unsafe_allow_html=True)
            
            # Overall accuracy rate
            accuracy_rate = st.session_state.prediction_correct / total_predictions
            st.markdown(f"<div style='text-align: center; padding: 0.2rem; border: 1px solid #007bff; border-radius: 4px; background: #f0f8ff; margin-top: 0.2rem;'><div style='font-size: 0.6rem; color: #007bff; margin-bottom: 0.1rem;'>Accuracy Rate</div><div style='font-size: 0.75rem; font-weight: bold; color: #007bff;'>{accuracy_rate:.1%}</div></div>", unsafe_allow_html=True)
    
    # Clear history button
    if st.session_state.game_history:
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.game_history = []
            st.session_state.current_session_name = ""
            st.session_state.prediction_pending = False
            st.session_state.prediction_correct = 0
            st.session_state.prediction_wrong = 0
            st.rerun()

# Additional Analysis Section (below main input/forecast)
if st.session_state.game_history and len(st.session_state.game_history) >= 3:
    st.markdown("---")
    st.header("📊 Detailed Analysis")
    
    # Get the results again for detailed analysis
    calculator = st.session_state.calculator
    results = calculator.calculate_probabilities(st.session_state.game_history)
    
    # Detailed Analysis
    st.subheader("Detailed Statistical Analysis")
    
    analysis_col1, analysis_col2 = st.columns([1, 1])
    
    with analysis_col1:
        # Win/Loss/Draw Distribution
        wins = st.session_state.game_history.count(1)
        losses = st.session_state.game_history.count(0)
        draws = st.session_state.game_history.count(2)
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Wins', 'Losses', 'Draws'],
            values=[wins, losses, draws],
            marker=dict(colors=['#28a745', '#dc3545', '#ffc107']),
            hole=0.3
        )])
        fig_pie.update_layout(
            title="Win/Loss/Draw Distribution",
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with analysis_col2:
        # Recent Performance Trend
        if len(st.session_state.game_history) >= 10:
            window_size = min(10, len(st.session_state.game_history))
            rolling_avg = pd.Series(st.session_state.game_history).rolling(window=window_size).mean()
            
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=list(range(1, len(rolling_avg) + 1)),
                y=rolling_avg,
                mode='lines+markers',
                name=f'Rolling Average ({window_size} games)',
                line=dict(color='#007bff', width=2)
            ))
            fig_line.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50% Line")
            fig_line.update_layout(
                title="Performance Trend",
                xaxis_title="Game Number",
                yaxis_title="Win Rate",
                height=300,
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_line, use_container_width=True)
    
    # Pattern Analysis
    st.subheader("Pattern Recognition")
    st.markdown("<p style='font-size: 0.85rem; color: #666; margin-bottom: 1rem;'>Understanding your gaming patterns and streaks to improve predictions</p>", unsafe_allow_html=True)
    
    pattern_info = calculator.analyze_patterns(st.session_state.game_history)
    
    # Current Performance Overview
    current_col1, current_col2 = st.columns([1, 1])
    
    with current_col1:
        streak_type = pattern_info['current_streak_type']
        streak_length = pattern_info['current_streak_length']
        streak_emoji = "🏆" if streak_type == "win" else ("❌" if streak_type == "loss" else "🤝")
        
        # Add streak context
        if streak_length >= 3:
            if streak_type == "win":
                streak_status = "🔥 Hot streak! You're on fire!"
            elif streak_type == "loss":
                streak_status = "⚠️ Cold streak - time to bounce back!"
            else:
                streak_status = "📊 Draw pattern - games are tight!"
        else:
            streak_status = "🎯 Recent mixed results"
        
        st.success(f"**Current Streak:** {streak_length} consecutive {streak_type}s {streak_emoji}")
        st.caption(streak_status)
    
    with current_col2:
        # Overall distribution with percentages
        total_games = pattern_info['total_games']
        win_pct = (pattern_info['win_count'] / total_games) * 100 if total_games > 0 else 0
        loss_pct = (pattern_info['loss_count'] / total_games) * 100 if total_games > 0 else 0
        draw_pct = (pattern_info['draw_count'] / total_games) * 100 if total_games > 0 else 0
        
        st.info(f"**Session Overview:** {total_games} games played")
        st.caption(f"Wins: {pattern_info['win_count']} ({win_pct:.1f}%) • Losses: {pattern_info['loss_count']} ({loss_pct:.1f}%) • Draws: {pattern_info['draw_count']} ({draw_pct:.1f}%)")
    
    # Historical Streak Analysis
    st.markdown("**Historical Performance Peaks:**")
    streak_col1, streak_col2, streak_col3 = st.columns([1, 1, 1])
    
    with streak_col1:
        win_streak = pattern_info['longest_win_streak']
        if win_streak >= 3:
            st.metric("🏆 Best Win Streak", f"{win_streak} games", help="Your longest winning run")
        else:
            st.metric("🏆 Best Win Streak", f"{win_streak} games", help="Room for improvement!")
    
    with streak_col2:
        loss_streak = pattern_info['longest_loss_streak']
        if loss_streak >= 3:
            st.metric("❌ Worst Loss Streak", f"{loss_streak} games", help="Your toughest stretch", delta=f"-{loss_streak}", delta_color="inverse")
        else:
            st.metric("❌ Worst Loss Streak", f"{loss_streak} games", help="Good consistency!")
    
    with streak_col3:
        draw_streak = pattern_info.get('longest_draw_streak', 0)
        if draw_streak > 0:
            st.metric("🤝 Longest Draw Run", f"{draw_streak} games", help="Most consecutive draws")
        else:
            st.metric("🤝 Longest Draw Run", "0 games", help="No draw streaks yet")
    
    # Performance Insights
    st.markdown("**📊 Pattern Insights:**")
    
    # Generate meaningful insights
    insights = []
    
    if win_pct > 40:
        insights.append("✅ Strong performer - winning more than expected!")
    elif win_pct < 25:
        insights.append("🎯 Focus area - wins below average")
    
    if streak_length >= 5:
        insights.append(f"🔥 Currently in a significant {streak_type} streak")
    
    if win_streak >= loss_streak + 2:
        insights.append("💪 Better at maintaining winning momentum")
    elif loss_streak >= win_streak + 2:
        insights.append("🎲 Experiences longer losing streaks - stay resilient!")
    
    if draw_pct > 35:
        insights.append("⚖️ High draw rate - games are very competitive")
    
    if abs(win_pct - loss_pct) < 10:
        insights.append("📊 Balanced performance - close win/loss ratio")
    
    if not insights:
        insights.append("📈 Still building your pattern - play more games for better insights!")
    
    for insight in insights:
        st.markdown(f"• {insight}")
    
    # Recent trend analysis
    if total_games >= 10:
        recent_games = st.session_state.game_history[-10:]
        recent_wins = recent_games.count(1)
        recent_performance = "improving" if recent_wins >= 4 else ("declining" if recent_wins <= 2 else "steady")
        trend_emoji = "📈" if recent_performance == "improving" else ("📉" if recent_performance == "declining" else "➡️")
        
        st.markdown(f"**Recent Trend (last 10 games):** {trend_emoji} {recent_performance.title()} - {recent_wins} wins, {10-recent_wins-recent_games.count(2)} losses, {recent_games.count(2)} draws")
    

    
    # Historical Performance Chart
    st.subheader("Game History Visualization")
    
    if len(st.session_state.game_history) > 1:
        fig_hist = go.Figure()
        
        colors = ['#28a745' if x == 1 else ('#dc3545' if x == 0 else '#ffc107') for x in st.session_state.game_history]
        
        fig_hist.add_trace(go.Bar(
            x=list(range(1, len(st.session_state.game_history) + 1)),
            y=st.session_state.game_history,
            marker_color=colors,
            name="Game Results"
        ))
        
        fig_hist.update_layout(
            title="Individual Game Results",
            xaxis_title="Game Number",
            yaxis_title="Result (0=Loss, 1=Win, 2=Draw)",
            height=400,
            yaxis=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=['Loss', 'Win', 'Draw'])
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.info("Enter your game history using the buttons above to see probability analysis.")

# Additional Data Sections (below main content)
st.markdown("---")

# Forecast Accuracy Tracking Section
if db_initialized:
    st.header("🎯 Accuracy Tracking")
    
    # Session List for Loading
    st.subheader("📂 Available Sessions")
    saved_sessions = get_all_sessions()
    
    if saved_sessions:
        st.markdown("<p style='font-size: 0.85rem; color: #666; margin-bottom: 0.5rem;'>Click on a session to load and view its accuracy data:</p>", unsafe_allow_html=True)
        
        # Create columns for session buttons
        session_cols = st.columns(min(4, len(saved_sessions)))
        
        for idx, session in enumerate(saved_sessions[:8]):  # Show up to 8 sessions
            with session_cols[idx % 4]:
                session_info = f"{session['name']}\n({session['total_games']} games)"
                if st.button(session_info, key=f"accuracy_load_{session['name']}", use_container_width=True):
                    loaded_data = load_game_session(session['name'])
                    if loaded_data:
                        st.session_state.game_history = loaded_data['game_history']
                        st.session_state.current_session_name = session['name']
                        st.success(f"Loaded session: {session['name']}")
                        st.rerun()
        
        if len(saved_sessions) > 8:
            st.markdown(f"<p style='font-size: 0.8rem; color: #999; font-style: italic;'>...and {len(saved_sessions) - 8} more sessions available in sidebar</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size: 0.85rem; color: #666; font-style: italic;'>No saved sessions available. Save a session to see accuracy tracking.</p>", unsafe_allow_html=True)
    
    st.markdown("---")

if db_initialized and st.session_state.current_session_name:
    st.subheader("📊 Current Session Analysis")
    
    # Get session-specific accuracy
    session_accuracy = get_prediction_accuracy(st.session_state.current_session_name)
    
    if session_accuracy:
        st.subheader(f"Current Session: {st.session_state.current_session_name}")
        
        acc_col1, acc_col2, acc_col3, acc_col4 = st.columns([1, 1, 1, 1])
        
        with acc_col1:
            st.metric(
                "Overall Accuracy", 
                f"{session_accuracy['overall_accuracy']:.1%}",
                help="Percentage of correct predictions"
            )
        
        with acc_col2:
            st.metric(
                "Total Predictions", 
                session_accuracy['total_predictions'],
                help="Number of predictions made"
            )
        
        with acc_col3:
            st.metric(
                "Correct Predictions", 
                session_accuracy['correct_predictions'],
                help="Number of accurate forecasts"
            )
        
        with acc_col4:
            wrong_predictions = session_accuracy['total_predictions'] - session_accuracy['correct_predictions']
            st.metric(
                "Wrong Predictions", 
                wrong_predictions,
                help="Number of incorrect forecasts"
            )
        
        # Detailed accuracy breakdown
        if session_accuracy['total_predictions'] > 0:
            st.subheader("Accuracy by Prediction Type")
            
            breakdown_col1, breakdown_col2, breakdown_col3 = st.columns([1, 1, 1])
            
            with breakdown_col1:
                if session_accuracy['win_predictions_count'] > 0:
                    st.metric(
                        "Win Predictions",
                        f"{session_accuracy['win_accuracy']:.1%}",
                        delta=f"{session_accuracy['win_predictions_count']} total"
                    )
                else:
                    st.metric("Win Predictions", "No data")
            
            with breakdown_col2:
                if session_accuracy['loss_predictions_count'] > 0:
                    st.metric(
                        "Loss Predictions",
                        f"{session_accuracy['loss_accuracy']:.1%}",
                        delta=f"{session_accuracy['loss_predictions_count']} total"
                    )
                else:
                    st.metric("Loss Predictions", "No data")
            
            with breakdown_col3:
                if session_accuracy['draw_predictions_count'] > 0:
                    st.metric(
                        "Draw Predictions",
                        f"{session_accuracy['draw_accuracy']:.1%}",
                        delta=f"{session_accuracy['draw_predictions_count']} total"
                    )
                else:
                    st.metric("Draw Predictions", "No data")
        
        # Recent predictions table
        if session_accuracy['recent_records']:
            st.subheader("Recent Predictions")
            
            records_data = []
            for record in session_accuracy['recent_records']:
                outcome_map = {0: "Loss", 1: "Win", 2: "Draw"}
                prediction_result = "✅ Correct" if record.was_correct else "❌ Wrong"
                
                records_data.append({
                    "Game #": record.game_number + 1,
                    "Predicted": record.most_likely_outcome.title(),
                    "Confidence": record.prediction_confidence,
                    "Actual": outcome_map[record.actual_outcome],
                    "Result": prediction_result
                })
            
            df_predictions = pd.DataFrame(records_data)
            st.dataframe(df_predictions, use_container_width=True, hide_index=True)
            
            # Interactive accuracy visualization
            st.subheader("📈 Prediction Accuracy Visualization")
            
            viz_col1, viz_col2 = st.columns([1, 1])
            
            with viz_col1:
                # Accuracy over time chart
                correct_series = [1 if record.was_correct else 0 for record in session_accuracy['recent_records']]
                game_numbers = [record.game_number + 1 for record in session_accuracy['recent_records']]
                
                # Calculate cumulative accuracy
                cumulative_correct = np.cumsum(correct_series)
                cumulative_total = np.arange(1, len(correct_series) + 1)
                cumulative_accuracy = cumulative_correct / cumulative_total
                
                fig_accuracy = go.Figure()
                fig_accuracy.add_trace(go.Scatter(
                    x=game_numbers,
                    y=cumulative_accuracy,
                    mode='lines+markers',
                    name='Cumulative Accuracy',
                    line=dict(color='#007bff', width=3),
                    marker=dict(size=8)
                ))
                
                # Add individual prediction markers
                colors = ['green' if correct else 'red' for correct in correct_series]
                fig_accuracy.add_trace(go.Scatter(
                    x=game_numbers,
                    y=[1 if correct else 0 for correct in correct_series],
                    mode='markers',
                    name='Individual Predictions',
                    marker=dict(color=colors, size=12, symbol='circle'),
                    yaxis='y2'
                ))
                
                fig_accuracy.add_hline(
                    y=0.33, line_dash="dash", line_color="gray", 
                    annotation_text="Random Chance (33%)"
                )
                
                fig_accuracy.update_layout(
                    title="Prediction Accuracy Over Time",
                    xaxis_title="Game Number",
                    yaxis_title="Cumulative Accuracy",
                    yaxis2=dict(
                        title="Prediction Result",
                        overlaying='y',
                        side='right',
                        tickmode='array',
                        tickvals=[0, 1],
                        ticktext=['Wrong', 'Correct']
                    ),
                    height=400
                )
                
                st.plotly_chart(fig_accuracy, use_container_width=True)
            
            with viz_col2:
                # Prediction confidence vs accuracy scatter plot
                confidence_values = []
                accuracy_values = []
                outcome_types = []
                
                for record in session_accuracy['recent_records']:
                    confidence_values.append(float(record.prediction_confidence.strip('%')) / 100)
                    accuracy_values.append(1 if record.was_correct else 0)
                    outcome_types.append(record.most_likely_outcome.title())
                
                fig_confidence = go.Figure()
                
                # Group by outcome type for different colors
                for outcome in ['Win', 'Loss', 'Draw']:
                    outcome_indices = [i for i, x in enumerate(outcome_types) if x == outcome]
                    if outcome_indices:
                        fig_confidence.add_trace(go.Scatter(
                            x=[confidence_values[i] for i in outcome_indices],
                            y=[accuracy_values[i] for i in outcome_indices],
                            mode='markers',
                            name=f'{outcome} Predictions',
                            marker=dict(
                                size=15,
                                color='#28a745' if outcome == 'Win' else ('#dc3545' if outcome == 'Loss' else '#ffc107'),
                                opacity=0.7
                            )
                        ))
                
                fig_confidence.update_layout(
                    title="Prediction Confidence vs Accuracy",
                    xaxis_title="Prediction Confidence",
                    yaxis_title="Accuracy (1=Correct, 0=Wrong)",
                    xaxis=dict(tickformat='.0%'),
                    yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Wrong', 'Correct']),
                    height=400
                )
                
                st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Prediction heatmap showing patterns
            if len(session_accuracy['recent_records']) >= 10:
                st.subheader("🔥 Prediction Pattern Analysis")
                
                # Create prediction matrix
                prediction_matrix = np.zeros((3, 3))  # [Predicted][Actual]
                outcome_map = {'win': 0, 'loss': 1, 'draw': 2}
                actual_map = {1: 0, 0: 1, 2: 2}  # Win=0, Loss=1, Draw=2
                
                for record in session_accuracy['recent_records']:
                    pred_idx = outcome_map[record.most_likely_outcome]
                    actual_idx = actual_map[record.actual_outcome]
                    prediction_matrix[pred_idx][actual_idx] += 1
                
                # Normalize to percentages
                row_sums = prediction_matrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                prediction_matrix_pct = prediction_matrix / row_sums * 100
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=prediction_matrix_pct,
                    x=['Actual Win', 'Actual Loss', 'Actual Draw'],
                    y=['Predicted Win', 'Predicted Loss', 'Predicted Draw'],
                    colorscale='RdYlGn',
                    text=[[f'{val:.1f}%' for val in row] for row in prediction_matrix_pct],
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    colorbar=dict(title="Accuracy %")
                ))
                
                fig_heatmap.update_layout(
                    title="Prediction Accuracy Matrix",
                    xaxis_title="Actual Outcome",
                    yaxis_title="Predicted Outcome",
                    height=350
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Analysis insights
                diagonal_sum = prediction_matrix_pct[0][0] + prediction_matrix_pct[1][1] + prediction_matrix_pct[2][2]
                if diagonal_sum > 150:  # Above 50% average on diagonal
                    st.success("🎯 **Strong Performance**: Your predictions show good accuracy across all outcome types!")
                elif diagonal_sum > 100:  # Above 33% average
                    st.warning("📊 **Moderate Performance**: Predictions are better than random chance but have room for improvement.")
                else:
                    st.error("📉 **Needs Improvement**: Consider reviewing your prediction strategy.")
    
    else:
        st.info("No prediction data available for current session. Make predictions by entering game history and then add actual results.")

# Overall Accuracy Across All Sessions
if db_initialized:
    overall_accuracy = get_all_prediction_accuracy()
    
    if overall_accuracy:
        st.subheader("Overall Forecast Performance (All Sessions)")
        
        overall_col1, overall_col2, overall_col3 = st.columns([1, 1, 1])
        
        with overall_col1:
            st.metric(
                "Global Accuracy", 
                f"{overall_accuracy['overall_accuracy']:.1%}",
                help="Accuracy across all sessions"
            )
        
        with overall_col2:
            st.metric(
                "Total Sessions Tracked", 
                overall_accuracy['session_count'],
                help="Number of sessions with predictions"
            )
        
        with overall_col3:
            st.metric(
                "Global Predictions", 
                overall_accuracy['total_predictions'],
                help="Total predictions made across all sessions"
            )
        
        # Best and worst performing sessions
        if overall_accuracy['best_session'] and overall_accuracy['worst_session']:
            performance_col1, performance_col2 = st.columns([1, 1])
            
            with performance_col1:
                best_name, best_acc = overall_accuracy['best_session']
                st.success(f"🏆 **Best Session:** {best_name} ({best_acc:.1%})")
            
            with performance_col2:
                worst_name, worst_acc = overall_accuracy['worst_session']
                st.error(f"📉 **Needs Improvement:** {worst_name} ({worst_acc:.1%})")
        
        # Interactive session comparison chart
        if len(overall_accuracy['session_accuracies']) > 1:
            st.subheader("📊 Session Performance Comparison")
            
            sessions = list(overall_accuracy['session_accuracies'].keys())
            accuracies = [overall_accuracy['session_accuracies'][session] for session in sessions]
            
            # Color sessions based on performance
            colors = ['green' if acc >= 0.5 else ('orange' if acc >= 0.33 else 'red') for acc in accuracies]
            
            fig_sessions = go.Figure(data=[
                go.Bar(
                    x=sessions,
                    y=accuracies,
                    marker_color=colors,
                    text=[f'{acc:.1%}' for acc in accuracies],
                    textposition='auto'
                )
            ])
            
            fig_sessions.add_hline(
                y=0.33, line_dash="dash", line_color="gray",
                annotation_text="Random Chance (33%)"
            )
            
            fig_sessions.update_layout(
                title="Accuracy by Session",
                xaxis_title="Session Name",
                yaxis_title="Accuracy",
                yaxis=dict(tickformat='.0%'),
                height=400
            )
            
            st.plotly_chart(fig_sessions, use_container_width=True)
            
            # Performance insights
            avg_accuracy = np.mean(accuracies)
            improving_sessions = sum(1 for acc in accuracies if acc > avg_accuracy)
            
            insight_col1, insight_col2, insight_col3 = st.columns([1, 1, 1])
            
            with insight_col1:
                st.metric("Average Accuracy", f"{avg_accuracy:.1%}")
            
            with insight_col2:
                st.metric("Above Average Sessions", f"{improving_sessions}/{len(sessions)}")
            
            with insight_col3:
                consistency = 1 - np.std(accuracies)
                st.metric("Consistency Score", f"{max(0, consistency):.1%}")
        
        # Real-time prediction confidence tracker
        st.subheader("🎯 Live Prediction Tracker")
        
        if st.session_state.last_prediction and st.session_state.prediction_pending:
            st.info("🔮 **Active Prediction:** Waiting for next game result to track accuracy")
            
            live_col1, live_col2, live_col3 = st.columns([1, 1, 1])
            
            with live_col1:
                win_prob = st.session_state.last_prediction['win']['weighted_average']
                st.metric("Win Forecast", f"{win_prob:.1%}")
            
            with live_col2:
                loss_prob = st.session_state.last_prediction['loss']['weighted_average']
                st.metric("Loss Forecast", f"{loss_prob:.1%}")
            
            with live_col3:
                draw_prob = st.session_state.last_prediction['draw']['weighted_average']
                st.metric("Draw Forecast", f"{draw_prob:.1%}")
            
            # Most confident prediction
            max_prob = max(win_prob, loss_prob, draw_prob)
            if max_prob == win_prob:
                prediction_text = f"🏆 Most Likely: WIN ({win_prob:.1%})"
                st.success(prediction_text)
            elif max_prob == draw_prob:
                prediction_text = f"🤝 Most Likely: DRAW ({draw_prob:.1%})"
                st.warning(prediction_text)
            else:
                prediction_text = f"❌ Most Likely: LOSS ({loss_prob:.1%})"
                st.error(prediction_text)
                
        else:
            st.info("💡 Add game history and create a session to start tracking prediction accuracy!")

# Enhanced Interactive Prediction Analysis
if db_initialized:
    st.markdown("---")
    st.header("🔍 Interactive Prediction Analysis")
    
    # Get all prediction records for interactive analysis
    all_predictions = get_all_prediction_accuracy()
    if all_predictions and all_predictions['total_predictions'] > 0:
        
        # Interactive filters
        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
        
        with filter_col1:
            # Session filter
            available_sessions = list(all_predictions['session_accuracies'].keys())
            selected_sessions = st.multiselect(
                "Filter by Sessions:",
                available_sessions,
                default=available_sessions,
                help="Select sessions to include in analysis"
            )
        
        with filter_col2:
            # Outcome type filter
            outcome_types = ['Win', 'Loss', 'Draw']
            selected_outcomes = st.multiselect(
                "Filter by Prediction Type:",
                outcome_types,
                default=outcome_types,
                help="Select prediction types to analyze"
            )
        
        with filter_col3:
            # Confidence range filter
            confidence_range = st.slider(
                "Confidence Range:",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05,
                format="%.0%",
                help="Filter predictions by confidence level"
            )
        
        # Interactive comparison chart
        if selected_sessions:
            st.subheader("📊 Interactive Session Performance Analysis")
            
            # Create detailed comparison data
            comparison_data = []
            for session in selected_sessions:
                if session in all_predictions['session_accuracies']:
                    # Get session-specific data
                    session_data = get_prediction_accuracy(session)
                    if session_data:
                        comparison_data.append({
                            'Session': session,
                            'Overall Accuracy': session_data['overall_accuracy'],
                            'Total Predictions': session_data['total_predictions'],
                            'Win Accuracy': session_data['win_accuracy'] if session_data['win_predictions_count'] > 0 else 0,
                            'Loss Accuracy': session_data['loss_accuracy'] if session_data['loss_predictions_count'] > 0 else 0,
                            'Draw Accuracy': session_data['draw_accuracy'] if session_data['draw_predictions_count'] > 0 else 0
                        })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                
                # Interactive bar chart with outcome breakdown
                chart_col1, chart_col2 = st.columns([2, 1])
                
                with chart_col1:
                    # Multi-trace bar chart
                    fig_comparison = go.Figure()
                    
                    if 'Win' in selected_outcomes:
                        fig_comparison.add_trace(go.Bar(
                            name='Win Predictions',
                            x=df_comparison['Session'],
                            y=df_comparison['Win Accuracy'],
                            marker_color='#28a745'
                        ))
                    
                    if 'Loss' in selected_outcomes:
                        fig_comparison.add_trace(go.Bar(
                            name='Loss Predictions',
                            x=df_comparison['Session'],
                            y=df_comparison['Loss Accuracy'],
                            marker_color='#dc3545'
                        ))
                    
                    if 'Draw' in selected_outcomes:
                        fig_comparison.add_trace(go.Bar(
                            name='Draw Predictions',
                            x=df_comparison['Session'],
                            y=df_comparison['Draw Accuracy'],
                            marker_color='#ffc107'
                        ))
                    
                    fig_comparison.add_hline(
                        y=0.33, line_dash="dash", line_color="gray",
                        annotation_text="Random Chance (33%)"
                    )
                    
                    fig_comparison.update_layout(
                        title="Prediction Accuracy by Session and Outcome Type",
                        xaxis_title="Session",
                        yaxis_title="Accuracy",
                        yaxis=dict(tickformat='.0%'),
                        barmode='group',
                        height=450
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                with chart_col2:
                    # Summary statistics for filtered data
                    st.subheader("Filtered Summary")
                    
                    avg_overall = df_comparison['Overall Accuracy'].mean()
                    best_session = df_comparison.loc[df_comparison['Overall Accuracy'].idxmax()]
                    total_filtered_predictions = df_comparison['Total Predictions'].sum()
                    
                    st.metric("Average Accuracy", f"{avg_overall:.1%}")
                    st.metric("Best Session", f"{best_session['Session'][:10]}...")
                    st.metric("Total Predictions", total_filtered_predictions)
                    
                    # Consistency analysis
                    accuracy_std = df_comparison['Overall Accuracy'].std()
                    consistency_score = max(0, 1 - accuracy_std)
                    st.metric("Consistency Score", f"{consistency_score:.1%}")
                
                # Detailed session breakdown table
                st.subheader("📋 Detailed Session Analysis")
                
                # Format the dataframe for better display
                df_display = df_comparison.copy()
                df_display['Overall Accuracy'] = df_display['Overall Accuracy'].apply(lambda x: f"{x:.1%}")
                df_display['Win Accuracy'] = df_display['Win Accuracy'].apply(lambda x: f"{x:.1%}" if x > 0 else "No data")
                df_display['Loss Accuracy'] = df_display['Loss Accuracy'].apply(lambda x: f"{x:.1%}" if x > 0 else "No data")
                df_display['Draw Accuracy'] = df_display['Draw Accuracy'].apply(lambda x: f"{x:.1%}" if x > 0 else "No data")
                
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                # Export functionality
                csv = df_comparison.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis as CSV",
                    data=csv,
                    file_name=f"prediction_analysis_{len(selected_sessions)}_sessions.csv",
                    mime="text/csv"
                )
        
        # Real-time accuracy trend
        st.subheader("📈 Real-Time Accuracy Trends")
        
        # Chart showing recent performance
        if st.session_state.current_session_name:
            current_session_data = get_prediction_accuracy(st.session_state.current_session_name)
            if current_session_data and current_session_data['recent_records']:
                
                # Moving average accuracy
                records = current_session_data['recent_records']
                window_size = min(5, len(records))
                
                accuracy_series = [1 if record.was_correct else 0 for record in records]
                if len(accuracy_series) >= window_size:
                    moving_avg = pd.Series(accuracy_series).rolling(window=window_size).mean()
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=list(range(1, len(moving_avg) + 1)),
                        y=moving_avg,
                        mode='lines+markers',
                        name=f'Moving Average (last {window_size})',
                        line=dict(color='#007bff', width=3)
                    ))
                    
                    # Add individual accuracy points
                    fig_trend.add_trace(go.Scatter(
                        x=list(range(1, len(accuracy_series) + 1)),
                        y=accuracy_series,
                        mode='markers',
                        name='Individual Results',
                        marker=dict(
                            color=['green' if x == 1 else 'red' for x in accuracy_series],
                            size=10
                        )
                    ))
                    
                    fig_trend.add_hline(
                        y=0.33, line_dash="dash", line_color="gray",
                        annotation_text="Random Chance"
                    )
                    
                    fig_trend.update_layout(
                        title=f"Accuracy Trend - {st.session_state.current_session_name}",
                        xaxis_title="Prediction Number",
                        yaxis_title="Accuracy",
                        yaxis=dict(tickformat='.0%'),
                        height=400
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Trend analysis
                    if len(accuracy_series) >= 3:
                        recent_trend = np.mean(accuracy_series[-3:]) - np.mean(accuracy_series[:-3]) if len(accuracy_series) > 3 else 0
                        if recent_trend > 0.1:
                            st.success(f"📈 **Improving Trend**: Recent accuracy is {recent_trend:.1%} better!")
                        elif recent_trend < -0.1:
                            st.warning(f"📉 **Declining Trend**: Recent accuracy has dropped by {abs(recent_trend):.1%}")
                        else:
                            st.info("📊 **Stable Performance**: Accuracy trend is consistent")
    
    else:
        st.info("📊 No prediction data available yet. Start making predictions to see interactive analysis!")

# Database Management Section
if db_initialized:
    st.markdown("---")
    st.header("📊 Database Management")
    
    # Overall statistics
    try:
        overall_stats = get_session_statistics()
    except Exception:
        overall_stats = None
    if overall_stats and overall_stats['total_games'] > 0:
        st.subheader("Overall Statistics Across All Sessions")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns([1, 1, 1, 1])
        
        with stat_col1:
            st.metric("Total Sessions", overall_stats['total_sessions'])
        with stat_col2:
            st.metric("Total Games", overall_stats['total_games'])
        with stat_col3:
            st.metric("Overall Win Rate", f"{overall_stats['overall_win_rate']:.1%}")
        with stat_col4:
            st.metric("Overall Draw Rate", f"{overall_stats['overall_draw_rate']:.1%}")
    
    # Session management
    st.subheader("Session Management")
    
    # Display all sessions in a table
    try:
        saved_sessions = get_all_sessions()
    except Exception:
        saved_sessions = []
    if saved_sessions:
        # Convert to DataFrame for better display
        df_sessions = pd.DataFrame(saved_sessions)
        df_sessions['Win Rate'] = (df_sessions['wins'] / df_sessions['total_games']).round(3)
        df_sessions['Draw Rate'] = (df_sessions['draws'] / df_sessions['total_games']).round(3)
        df_sessions['Created'] = pd.to_datetime(df_sessions['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        df_sessions['Updated'] = pd.to_datetime(df_sessions['updated_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Select columns to display
        display_df = df_sessions[['name', 'total_games', 'wins', 'losses', 'draws', 'Win Rate', 'Draw Rate', 'Updated']]
        display_df.columns = ['Session Name', 'Games', 'Wins', 'Losses', 'Draws', 'Win Rate', 'Draw Rate', 'Last Updated']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Session deletion
        st.subheader("Delete Sessions")
        session_to_delete = st.selectbox(
            "Select session to delete:",
            options=[''] + [session['name'] for session in saved_sessions],
            key="delete_session_selector"
        )
        
        if session_to_delete and st.button(f"🗑️ Delete '{session_to_delete}'", type="secondary"):
            if delete_game_session(session_to_delete):
                st.success(f"Session '{session_to_delete}' deleted!")
                st.rerun()
            else:
                st.error(f"Failed to delete session '{session_to_delete}'")
    
    else:
        st.info("No saved sessions found. Create and save some game sessions to see statistics here.")

# Footer
st.markdown("---")
st.markdown("""
**Note:** This probability calculator uses statistical analysis based on your historical data. 
Past performance does not guarantee future results. Use these predictions as guidance only.
""")

if db_initialized:
    st.markdown("💾 **Database:** Your game sessions are automatically saved to a PostgreSQL database for persistence across browser sessions.")
