import streamlit as st
import re

def validate_history(history):
    """
    Validate game history input.
    
    Args:
        history (list): List of game results
        
    Returns:
        dict: Validation result with success flag and message
    """
    if not isinstance(history, list):
        return {'success': False, 'message': 'History must be a list'}
    
    if len(history) == 0:
        return {'success': False, 'message': 'History cannot be empty'}
    
    if len(history) > 50:
        return {'success': False, 'message': 'Maximum 50 games allowed'}
    
    # Check if all values are 0, 1, or 2
    for i, value in enumerate(history):
        if value not in [0, 1, 2]:
            return {'success': False, 'message': f'Invalid value at position {i+1}: {value}. Use only 0 (loss), 1 (win), or 2 (draw)'}
    
    return {'success': True, 'message': 'History is valid'}

def format_probability(probability):
    """
    Format probability for display.
    
    Args:
        probability (float): Probability value between 0 and 1
        
    Returns:
        str: Formatted probability string
    """
    if probability is None:
        return "N/A"
    
    percentage = probability * 100
    return f"{percentage:.1f}%"

def parse_bulk_input(input_string):
    """
    Parse bulk input string into game history.
    
    Args:
        input_string (str): Input string with W/L/D characters
        
    Returns:
        dict: Parsing result with success flag, history, and message
    """
    if not input_string:
        return {'success': False, 'history': [], 'message': 'Input string is empty'}
    
    # Clean the input string
    cleaned_input = re.sub(r'[^WLDwld]', '', input_string.upper())
    
    if not cleaned_input:
        return {'success': False, 'history': [], 'message': 'No valid characters found. Use W for wins, L for losses, and D for draws.'}
    
    if len(cleaned_input) > 50:
        return {'success': False, 'history': [], 'message': 'Maximum 50 games allowed'}
    
    # Convert to history
    history = []
    for char in cleaned_input:
        if char == 'W':
            history.append(1)
        elif char == 'L':
            history.append(0)
        elif char == 'D':
            history.append(2)
    
    return {'success': True, 'history': history, 'message': f'Successfully parsed {len(history)} games'}

def get_streak_emoji(streak_type, streak_length):
    """
    Get emoji representation for streaks.
    
    Args:
        streak_type (str): 'win' or 'loss'
        streak_length (int): Length of the streak
        
    Returns:
        str: Emoji representation
    """
    if streak_type == 'win':
        if streak_length >= 5:
            return "🔥"  # Hot streak
        elif streak_length >= 3:
            return "📈"  # Upward trend
        else:
            return "✅"  # Win
    else:
        if streak_length >= 5:
            return "❄️"  # Cold streak
        elif streak_length >= 3:
            return "📉"  # Downward trend
        else:
            return "❌"  # Loss

def get_probability_color(probability):
    """
    Get color coding for probability values.
    
    Args:
        probability (float): Probability value between 0 and 1
        
    Returns:
        str: Color name or hex code
    """
    if probability >= 0.7:
        return "#28a745"  # Green
    elif probability >= 0.5:
        return "#ffc107"  # Yellow
    elif probability >= 0.3:
        return "#fd7e14"  # Orange
    else:
        return "#dc3545"  # Red

def get_recommendation_style(probability):
    """
    Get styling information for recommendations.
    
    Args:
        probability (float): Probability value between 0 and 1
        
    Returns:
        dict: Style information with color and icon
    """
    if probability >= 0.6:
        return {
            'color': '#28a745',
            'icon': '🟢',
            'level': 'success'
        }
    elif probability >= 0.4:
        return {
            'color': '#ffc107',
            'icon': '🟡',
            'level': 'warning'
        }
    else:
        return {
            'color': '#dc3545',
            'icon': '🔴',
            'level': 'error'
        }

def calculate_statistics(history):
    """
    Calculate basic statistics for the game history.
    
    Args:
        history (list): List of game results
        
    Returns:
        dict: Dictionary with various statistics
    """
    if not history:
        return {
            'total_games': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate': 0,
            'loss_rate': 0,
            'draw_rate': 0
        }
    
    total_games = len(history)
    wins = history.count(1)
    losses = history.count(0)
    draws = history.count(2)
    
    win_rate = wins / total_games if total_games > 0 else 0
    loss_rate = losses / total_games if total_games > 0 else 0
    draw_rate = draws / total_games if total_games > 0 else 0
    
    return {
        'total_games': total_games,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate
    }

def generate_summary_text(history, probabilities):
    """
    Generate a summary text based on history and probabilities.
    
    Args:
        history (list): Game history
        probabilities (dict): Probability calculations
        
    Returns:
        str: Summary text
    """
    if not history:
        return "No game history available."
    
    stats = calculate_statistics(history)
    main_prob = probabilities.get('weighted_average', 0.5)
    
    summary = f"""
    Based on {stats['total_games']} games with {stats['wins']} wins and {stats['losses']} losses:
    
    • Overall win rate: {stats['win_rate']:.1%}
    • Predicted probability for next hand: {main_prob:.1%}
    • Recommendation: {get_recommendation_text(main_prob)}
    """
    
    return summary.strip()

def get_recommendation_text(probability):
    """
    Get recommendation text based on probability.
    
    Args:
        probability (float): Probability value
        
    Returns:
        str: Recommendation text
    """
    if probability >= 0.6:
        return "Strong probability of winning next hand"
    elif probability >= 0.4:
        return "Moderate probability, proceed with caution"
    else:
        return "Low probability of winning next hand"

def format_history_display(history, max_per_line=10):
    """
    Format history for display with line breaks.
    
    Args:
        history (list): Game history
        max_per_line (int): Maximum characters per line
        
    Returns:
        str: Formatted history string
    """
    if not history:
        return "No history"
    
    display_chars = ["W" if x == 1 else ("L" if x == 0 else "D") for x in history]
    
    # Add line breaks every max_per_line characters
    lines = []
    for i in range(0, len(display_chars), max_per_line):
        line = "".join(display_chars[i:i + max_per_line])
        lines.append(line)
    
    return "\n".join(lines)
