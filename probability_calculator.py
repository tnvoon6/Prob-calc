import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import math

class ProbabilityCalculator:
    """
    A comprehensive probability calculator that analyzes game history
    to predict winning chances for the next hand.
    Supports three outcomes: Win (1), Loss (0), Draw (2)
    """
    
    def __init__(self):
        self.min_data_points = 3
        self.method_weights = {
            'simple_frequency': 0.15,
            'recent_form': 0.30,
            'streak_adjusted': 0.20,
            'markov_chain': 0.15,
            'momentum_analysis': 0.10,
            'pattern_recognition': 0.10
        }
    
    def calculate_probabilities(self, history):
        """
        Calculate various probability metrics based on game history.
        
        Args:
            history (list): List of 1s (wins), 0s (losses), and 2s (draws)
            
        Returns:
            dict: Dictionary containing different probability calculations for each outcome
        """
        if not history or len(history) < self.min_data_points:
            return self._default_probabilities()
        
        results = {}
        
        # Calculate probabilities for each outcome
        for outcome in ['win', 'loss', 'draw']:
            results[outcome] = {}
            
            # Simple frequency analysis
            results[outcome]['simple_frequency'] = self._simple_frequency(history, outcome)
            
            # Recent form analysis (last 10 games or available)
            results[outcome]['recent_form'] = self._recent_form_analysis(history, outcome)
            
            # Streak-adjusted probability
            results[outcome]['streak_adjusted'] = self._streak_adjusted_probability(history, outcome)
            
            # Markov chain analysis
            results[outcome]['markov_chain'] = self._markov_chain_analysis(history, outcome)
            
            # Momentum analysis
            results[outcome]['momentum_analysis'] = self._momentum_analysis(history, outcome)
            
            # Pattern recognition
            results[outcome]['pattern_recognition'] = self._pattern_recognition(history, outcome)
            
            # Weighted average of all methods (using dynamic weights)
            results[outcome]['weighted_average'] = self._calculate_weighted_average(results[outcome])
            
            # Calculate confidence intervals
            results[outcome]['confidence_interval'] = self._calculate_confidence_interval(history, outcome)
        
        return results
    
    def _default_probabilities(self):
        """Return default probabilities when insufficient data."""
        return {
            'win': {
                'simple_frequency': 1/3,
                'recent_form': 1/3,
                'streak_adjusted': 1/3,
                'markov_chain': 1/3,
                'momentum_analysis': 1/3,
                'pattern_recognition': 1/3,
                'weighted_average': 1/3,
                'confidence_interval': {'lower': 0.1, 'upper': 0.6, 'confidence_level': 0.95}
            },
            'loss': {
                'simple_frequency': 1/3,
                'recent_form': 1/3,
                'streak_adjusted': 1/3,
                'markov_chain': 1/3,
                'momentum_analysis': 1/3,
                'pattern_recognition': 1/3,
                'weighted_average': 1/3,
                'confidence_interval': {'lower': 0.1, 'upper': 0.6, 'confidence_level': 0.95}
            },
            'draw': {
                'simple_frequency': 1/3,
                'recent_form': 1/3,
                'streak_adjusted': 1/3,
                'markov_chain': 1/3,
                'momentum_analysis': 1/3,
                'pattern_recognition': 1/3,
                'weighted_average': 1/3,
                'confidence_interval': {'lower': 0.1, 'upper': 0.6, 'confidence_level': 0.95}
            }
        }
    
    def _simple_frequency(self, history, outcome='win'):
        """Calculate simple rate for given outcome."""
        if not history:
            return 1/3
        
        target_value = self._get_outcome_value(outcome)
        return history.count(target_value) / len(history)
    
    def _recent_form_analysis(self, history, outcome='win', window=15):
        """Enhanced recent form analysis with adaptive weighting and hot/cold streak detection."""
        if len(history) < 5:
            return self._simple_frequency(history, outcome)
        
        target_value = self._get_outcome_value(outcome)
        recent_games = history[-window:]
        
        # Apply exponential decay weights with adaptive scaling
        base_weights = [0.6 ** (len(recent_games) - i - 1) for i in range(len(recent_games))]
        
        # Detect hot/cold streaks and adjust weights
        adjusted_weights = []
        for i, game in enumerate(recent_games):
            weight = base_weights[i]
            
            # Check for streak context around this game
            streak_bonus = self._calculate_streak_bonus(recent_games, i, target_value)
            adjusted_weights.append(weight * (1 + streak_bonus))
        
        total_weight = sum(adjusted_weights)
        
        # Calculate weighted probability
        weighted_matches = sum(w for i, w in enumerate(adjusted_weights) if recent_games[i] == target_value)
        
        # Add variance adjustment based on consistency
        consistency_factor = self._calculate_consistency_factor(recent_games, target_value)
        result = weighted_matches / total_weight if total_weight > 0 else 1/3
        
        # Apply consistency adjustment
        result = result * consistency_factor + (1/3) * (1 - consistency_factor)
        
        return max(0.05, min(0.95, result))
    
    def _streak_adjusted_probability(self, history, outcome='win'):
        """
        Adjust probability based on current streak.
        Streaks of the same type slightly increase probability for that outcome.
        """
        if len(history) < 3:
            return self._simple_frequency(history, outcome)
        
        base_prob = self._simple_frequency(history, outcome)
        current_streak = self._get_current_streak(history)
        
        # Streak adjustment factor (diminishing returns)
        if current_streak['type'] == outcome:
            # Current streak matches outcome: slight increase, but capped
            adjustment = min(0.1, current_streak['length'] * 0.02)
        else:
            # Current streak doesn't match: slight decrease, but capped
            adjustment = -min(0.05, current_streak['length'] * 0.01)
        
        # Apply adjustment with bounds
        adjusted_prob = base_prob + adjustment
        return max(0.05, min(0.95, adjusted_prob))
    
    def _markov_chain_analysis(self, history, outcome='win'):
        """
        Use Markov chain analysis to predict next outcome
        based on transition probabilities.
        """
        if len(history) < 2:
            return self._simple_frequency(history, outcome)
        
        transitions = self.calculate_transition_matrix(history)
        
        # Use the last result to predict next outcome
        last_result = history[-1]
        last_outcome = self._get_outcome_name(last_result)
        
        transition_key = f"{last_outcome}_to_{outcome}"
        return transitions.get(transition_key, 1/3)
    
    def calculate_transition_matrix(self, history):
        """Calculate transition probabilities between states."""
        if len(history) < 2:
            default_prob = 1/3
            return {
                'win_to_win': default_prob, 'win_to_loss': default_prob, 'win_to_draw': default_prob,
                'loss_to_win': default_prob, 'loss_to_loss': default_prob, 'loss_to_draw': default_prob,
                'draw_to_win': default_prob, 'draw_to_loss': default_prob, 'draw_to_draw': default_prob
            }
        
        transitions = defaultdict(int)
        
        for i in range(len(history) - 1):
            current_state = self._get_outcome_name(history[i])
            next_state = self._get_outcome_name(history[i + 1])
            transitions[f"{current_state}_to_{next_state}"] += 1
        
        # Calculate probabilities for each starting state
        result = {}
        for start_state in ['win', 'loss', 'draw']:
            total = sum(transitions[f"{start_state}_to_{end_state}"] for end_state in ['win', 'loss', 'draw'])
            
            if total > 0:
                for end_state in ['win', 'loss', 'draw']:
                    result[f"{start_state}_to_{end_state}"] = transitions[f"{start_state}_to_{end_state}"] / total
            else:
                # Default to equal probabilities if no data
                for end_state in ['win', 'loss', 'draw']:
                    result[f"{start_state}_to_{end_state}"] = 1/3
        
        return result
    
    def _momentum_analysis(self, history, outcome):
        """
        Advanced momentum analysis considering recent trend direction and acceleration.
        """
        if len(history) < 5:
            return self._simple_frequency(history, outcome)
        
        target_value = self._get_outcome_value(outcome)
        
        # Analyze last 5, 10, and 20 games with different weights
        windows = [5, 10, min(20, len(history))]
        weights = [0.5, 0.3, 0.2]
        
        momentum_score = 0
        
        for window, weight in zip(windows, weights):
            recent_history = history[-window:]
            recent_frequency = recent_history.count(target_value) / len(recent_history)
            
            # Calculate trend (comparing first half vs second half of window)
            if window >= 6:
                first_half = recent_history[:window//2]
                second_half = recent_history[window//2:]
                
                first_freq = first_half.count(target_value) / len(first_half)
                second_freq = second_half.count(target_value) / len(second_half)
                
                # Trend factor: positive if increasing, negative if decreasing
                trend_factor = (second_freq - first_freq) * 2
                adjusted_frequency = recent_frequency + trend_factor
                
                momentum_score += max(0, min(1, adjusted_frequency)) * weight
            else:
                momentum_score += recent_frequency * weight
        
        return max(0.05, min(0.95, momentum_score))
    
    def _pattern_recognition(self, history, outcome):
        """
        Advanced pattern recognition using sequence analysis and cyclical patterns.
        """
        if len(history) < 6:
            return self._simple_frequency(history, outcome)
        
        target_value = self._get_outcome_value(outcome)
        
        # Look for repeating patterns of length 2-5
        pattern_scores = []
        
        for pattern_length in range(2, min(6, len(history)//2 + 1)):
            pattern_prob = self._analyze_pattern_sequences(history, target_value, pattern_length)
            pattern_scores.append(pattern_prob)
        
        # Cyclical analysis - look for periods of 3, 5, 7, 10
        cycle_scores = []
        for cycle_length in [3, 5, 7, 10]:
            if len(history) >= cycle_length * 2:
                cycle_prob = self._analyze_cyclical_pattern(history, target_value, cycle_length)
                cycle_scores.append(cycle_prob)
        
        # Combine pattern and cycle scores
        pattern_avg = sum(pattern_scores) / len(pattern_scores) if pattern_scores else 0.33
        cycle_avg = sum(cycle_scores) / len(cycle_scores) if cycle_scores else 0.33
        
        # Weight pattern analysis more heavily
        final_score = pattern_avg * 0.7 + cycle_avg * 0.3
        
        return max(0.05, min(0.95, final_score))
    
    def _analyze_pattern_sequences(self, history, target_value, pattern_length):
        """Analyze repeating sequence patterns."""
        if len(history) < pattern_length * 2:
            return 0.33
        
        # Find the most recent pattern
        current_pattern = history[-pattern_length:]
        
        # Count how many times this pattern appears, and what follows
        pattern_matches = 0
        target_follows = 0
        
        for i in range(len(history) - pattern_length):
            if history[i:i+pattern_length] == current_pattern:
                pattern_matches += 1
                # Check what follows this pattern
                if i + pattern_length < len(history):
                    if history[i + pattern_length] == target_value:
                        target_follows += 1
        
        if pattern_matches == 0:
            return 0.33
        
        return target_follows / pattern_matches
    
    def _analyze_cyclical_pattern(self, history, target_value, cycle_length):
        """Analyze cyclical patterns."""
        if len(history) < cycle_length * 2:
            return 0.33
        
        # Look at positions in the cycle
        current_position = len(history) % cycle_length
        
        # Count occurrences of target_value at this position in the cycle
        position_count = 0
        target_count = 0
        
        for i in range(current_position, len(history), cycle_length):
            position_count += 1
            if history[i] == target_value:
                target_count += 1
        
        if position_count == 0:
            return 0.33
        
        return target_count / position_count
    
    def _calculate_streak_bonus(self, recent_games, position, target_value):
        """Calculate streak bonus for a specific position in recent games."""
        if position == 0:
            return 0
        
        # Look at surrounding games to detect streak context
        window_start = max(0, position - 2)
        window_end = min(len(recent_games), position + 3)
        window = recent_games[window_start:window_end]
        
        # If this game is part of a streak, give it a bonus
        matches_in_window = window.count(target_value)
        total_in_window = len(window)
        
        if matches_in_window >= total_in_window * 0.6:  # 60% or more matches
            return 0.3  # Hot streak bonus
        elif matches_in_window <= total_in_window * 0.2:  # 20% or fewer matches
            return -0.2  # Cold streak penalty
        else:
            return 0
    
    def _calculate_consistency_factor(self, recent_games, target_value):
        """Calculate consistency factor based on variance in recent performance."""
        if len(recent_games) < 5:
            return 0.5
        
        # Calculate rolling frequency over different windows
        frequencies = []
        for window_size in [3, 5, 7]:
            if len(recent_games) >= window_size:
                for i in range(len(recent_games) - window_size + 1):
                    window = recent_games[i:i + window_size]
                    freq = window.count(target_value) / len(window)
                    frequencies.append(freq)
        
        if not frequencies:
            return 0.5
        
        # Calculate variance - lower variance means more consistency
        variance = np.var(frequencies)
        
        # Convert variance to consistency factor (0.0 to 1.0)
        # Lower variance = higher consistency
        consistency = max(0.1, min(1.0, 1.0 - variance * 4))
        
        return consistency
    
    def _calculate_weighted_average(self, results):
        """Calculate weighted average of all probability methods using dynamic weights."""
        weighted_sum = 0
        for method, weight in self.method_weights.items():
            if method in results:
                weighted_sum += results[method] * weight
        
        return weighted_sum
    
    def update_method_weights(self, prediction_accuracy_data):
        """
        Update method weights based on prediction accuracy data.
        Methods with higher accuracy get higher weights.
        """
        if not prediction_accuracy_data:
            return
        
        # Calculate accuracy for each method by analyzing prediction records
        method_accuracies = self._calculate_method_accuracies(prediction_accuracy_data)
        
        # Update weights based on relative performance
        total_performance = sum(method_accuracies.values())
        if total_performance > 0:
            for method in self.method_weights:
                if method in method_accuracies:
                    # Base weight + performance bonus
                    base_weight = 0.15  # Minimum weight
                    performance_weight = 0.85 * (method_accuracies[method] / total_performance)
                    self.method_weights[method] = base_weight + performance_weight
        
        # Ensure weights sum to 1.0
        total_weight = sum(self.method_weights.values())
        if total_weight > 0:
            for method in self.method_weights:
                self.method_weights[method] /= total_weight
    
    def _calculate_method_accuracies(self, accuracy_data):
        """
        Calculate how well each method performed historically.
        This is a simplified version - in practice would need more detailed tracking.
        """
        # For now, return equal accuracies - would need enhanced tracking
        # to separate method performance
        base_accuracy = accuracy_data.get('overall_accuracy', 0.33)
        
        return {
            'simple_frequency': base_accuracy * 0.8,
            'recent_form': base_accuracy * 1.3,  # Strongly favor recent form
            'streak_adjusted': base_accuracy * 0.9,
            'markov_chain': base_accuracy * 1.0,
            'momentum_analysis': base_accuracy * 1.2,  # Favor momentum
            'pattern_recognition': base_accuracy * 1.1  # Favor pattern recognition
        }
    
    def _get_current_streak(self, history):
        """Get information about the current streak."""
        if not history:
            return {'type': 'none', 'length': 0}
        
        current_result = history[-1]
        streak_type = self._get_outcome_name(current_result)
        streak_length = 1
        
        # Count backwards to find streak length
        for i in range(len(history) - 2, -1, -1):
            if history[i] == current_result:
                streak_length += 1
            else:
                break
        
        return {'type': streak_type, 'length': streak_length}
    
    def analyze_patterns(self, history):
        """Analyze various patterns in the game history."""
        if not history:
            return self._default_pattern_analysis()
        
        patterns = {}
        
        # Current streak analysis
        current_streak = self._get_current_streak(history)
        patterns['current_streak_type'] = current_streak['type']
        patterns['current_streak_length'] = current_streak['length']
        
        # Find longest streaks for each outcome
        patterns['longest_win_streak'] = self._find_longest_streak(history, 1)
        patterns['longest_loss_streak'] = self._find_longest_streak(history, 0)
        patterns['longest_draw_streak'] = self._find_longest_streak(history, 2)
        
        # Calculate outcome distribution
        patterns['win_count'] = history.count(1)
        patterns['loss_count'] = history.count(0)
        patterns['draw_count'] = history.count(2)
        patterns['total_games'] = len(history)
        
        return patterns
    
    def _default_pattern_analysis(self):
        """Return default pattern analysis when no history."""
        return {
            'current_streak_type': 'none',
            'current_streak_length': 0,
            'longest_win_streak': 0,
            'longest_loss_streak': 0,
            'longest_draw_streak': 0,
            'win_count': 0,
            'loss_count': 0,
            'draw_count': 0,
            'total_games': 0
        }
    
    def _find_longest_streak(self, history, target_value):
        """Find the longest streak of a specific value."""
        if not history:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for value in history:
            if value == target_value:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _detect_alternating_pattern(self, history):
        """Detect how often the results alternate between wins and losses."""
        if len(history) < 2:
            return 0
        
        alternations = 0
        for i in range(len(history) - 1):
            if history[i] != history[i + 1]:
                alternations += 1
        
        return alternations / (len(history) - 1)
    
    def _detect_hot_periods(self, history, threshold=0.7, min_length=5):
        """Detect periods where win rate is above threshold."""
        if len(history) < min_length:
            return 0
        
        hot_periods = 0
        window_size = min_length
        
        for i in range(len(history) - window_size + 1):
            window = history[i:i + window_size]
            if sum(window) / len(window) >= threshold:
                hot_periods += 1
        
        return hot_periods
    
    def _detect_cold_periods(self, history, threshold=0.3, min_length=5):
        """Detect periods where win rate is below threshold."""
        if len(history) < min_length:
            return 0
        
        cold_periods = 0
        window_size = min_length
        
        for i in range(len(history) - window_size + 1):
            window = history[i:i + window_size]
            if sum(window) / len(window) <= threshold:
                cold_periods += 1
        
        return cold_periods
    
    def _calculate_confidence_interval(self, history, outcome, confidence_level=0.95):
        """
        Calculate confidence interval for the probability estimate using binomial distribution.
        
        Args:
            history: List of game results
            outcome: 'win', 'loss', or 'draw'
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            dict: Contains lower, upper bounds and confidence level
        """
        if not history or len(history) < 3:
            # For small samples, return wide confidence intervals
            return {
                'lower': 0.05,
                'upper': 0.95, 
                'confidence_level': confidence_level
            }
        
        n = len(history)
        target_value = self._get_outcome_value(outcome)
        successes = history.count(target_value)
        
        # Point estimate
        p_hat = successes / n
        
        # Use Wilson score interval for better performance with small samples
        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha/2)  # Critical value
        
        # Wilson score interval calculation
        denominator = 1 + (z**2 / n)
        center = (p_hat + (z**2 / (2*n))) / denominator
        margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / denominator
        
        lower = max(0.01, center - margin)  # Ensure minimum 1%
        upper = min(0.99, center + margin)  # Ensure maximum 99%
        
        return {
            'lower': lower,
            'upper': upper,
            'confidence_level': confidence_level
        }
    
    def calculate_confidence_interval(self, history, outcome='win', confidence=0.95):
        """Calculate confidence interval for the specified outcome probability."""
        if not history:
            return {'lower': 0.2, 'upper': 0.4}
        
        n = len(history)
        target_value = self._get_outcome_value(outcome)
        p = history.count(target_value) / n
        
        # Calculate standard error
        se = math.sqrt(p * (1 - p) / n)
        
        # Z-score for 95% confidence interval
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Calculate margin of error
        margin_of_error = z_score * se
        
        lower = max(0, p - margin_of_error)
        upper = min(1, p + margin_of_error)
        
        return {'lower': lower, 'upper': upper}
    
    def get_recommendation(self, history):
        """Get a recommendation based on the analysis."""
        if not history:
            return "Insufficient data for recommendation"
        
        probabilities = self.calculate_probabilities(history)
        win_prob = probabilities['win']['weighted_average']
        loss_prob = probabilities['loss']['weighted_average']
        draw_prob = probabilities['draw']['weighted_average']
        
        # Find the most likely outcome
        max_prob = max(win_prob, loss_prob, draw_prob)
        
        if max_prob == win_prob:
            if win_prob > 0.5:
                return f"Strong probability of winning next hand ({win_prob:.1%})"
            else:
                return f"Moderate probability of winning next hand ({win_prob:.1%})"
        elif max_prob == draw_prob:
            return f"Most likely outcome is a draw ({draw_prob:.1%})"
        else:
            return f"Higher probability of losing next hand ({loss_prob:.1%})"
    
    def _get_outcome_value(self, outcome):
        """Convert outcome name to numeric value."""
        mapping = {'win': 1, 'loss': 0, 'draw': 2}
        return mapping.get(outcome, 0)
    
    def _get_outcome_name(self, value):
        """Convert numeric value to outcome name."""
        mapping = {1: 'win', 0: 'loss', 2: 'draw'}
        return mapping.get(value, 'loss')
