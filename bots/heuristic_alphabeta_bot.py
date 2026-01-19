"""
Heuristic Alpha-Beta Chess Bot

A chess bot that uses alpha-beta pruning with heuristic evaluation functions
for positions that are not terminal states.
"""

from typing import Dict, Any, Callable

import chess

from adversarial_search.chess_game_state import ChessGameState
from adversarial_search.chess_problem import ChessProblem
from adversarial_search.game_algorithms import heuristic_alphabeta_search
from bots.chess_bot import ChessBot
from bots.features import get_position_features


def linear_weighted_heuristic(state: ChessGameState, player: chess.Color, weights: Dict[str, float]) -> float:
    """
    Generic linear weighted heuristic evaluation function.

    Collects position features, applies weighted linear combination, and clips the result.

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :param weights: Dictionary of feature weights (must sum to 1.0)
    :type weights: Dict[str, float]
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    board = state.get_board()
    features = get_position_features(board)
    score = weighted_linear_features(features, weights, player)
    return max(1e-2, min(1.0 - 1e-2, score))


def weighted_linear_features(features: Dict[str, float], weights: Dict[str, float], player: chess.Color) -> float:
    """
    Generic weighted linear combination of normalized features.

    :param features: Dictionary of features from get_position_features
    :type features: Dict[str, float]
    :param weights: Dictionary of feature weights (must sum to 1.0)
    :type weights: Dict[str, float]
    :param player: Player to evaluate for (chess.WHITE or chess.BLACK)
    :type player: chess.Color
    :return: Weighted evaluation score (0 to 1)
    :rtype: float
    :raises ValueError: If weights don't sum to 1.0 or contain invalid keys
    """
    # Validate weights
    normalized_features = {
        "material_diff_normalized",
        "positional_diff_normalized",
        "mobility_diff_normalized",
        "center_diff_normalized",
        "king_safety_diff_normalized",
        "development_diff_normalized"
    }

    # Check that all weight keys are valid normalized features
    invalid_keys = set(weights.keys()) - normalized_features
    if invalid_keys:
        raise ValueError(f"Invalid weight keys: {invalid_keys}. Valid keys: {normalized_features}")

    # Check that weights sum to approximately 1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    # Check that all weights are non-negative
    negative_weights = {k: v for k, v in weights.items() if v < 0}
    if negative_weights:
        raise ValueError(f"Weights must be non-negative, got negative weights: {negative_weights}")

    # Calculate weighted score
    if player == chess.WHITE:
        score = sum(weights.get(feature, 0.0) * features[feature] for feature in normalized_features)
    else:
        # For black, invert the features (1.0 - feature_value)
        score = sum(weights.get(feature, 0.0) * (1.0 - features[feature]) for feature in normalized_features)

    return score



def h1(state: ChessGameState, player: chess.Color) -> float:
    """
    Heuristic 1: Equal-weighted evaluation.

    Uses uniform weights (1/6 ≈ 16.67% each) for all six normalized features.

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    equal_weight = 1.0 / 6
    weights = {
        "material_diff_normalized": equal_weight,
        "positional_diff_normalized": equal_weight,
        "mobility_diff_normalized": equal_weight,
        "center_diff_normalized": equal_weight,
        "king_safety_diff_normalized": equal_weight,
        "development_diff_normalized": equal_weight
    }
    return linear_weighted_heuristic(state, player, weights)


def h2(state: ChessGameState, player: chess.Color) -> float:
    """
    Heuristic 2

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    weights = {
        "material_diff_normalized": 0.10,
        "positional_diff_normalized": 0.10,
        "mobility_diff_normalized": 0.0,
        "center_diff_normalized": 0.0,
        "king_safety_diff_normalized": 0.50,
        "development_diff_normalized": 0.30,
    }
    return linear_weighted_heuristic(state, player, weights)


def h3(state: ChessGameState, player: chess.Color) -> float:
    """
    Heuristic 3:

    :param state: Current chess game state
    :type state: ChessGameState
    :param player: Player to evaluate for
    :type player: chess.Color
    :return: Evaluation score (0 to 1)
    :rtype: float
    """
    weights = {
        "material_diff_normalized": 0.50,
        "positional_diff_normalized": 0.20,
        "mobility_diff_normalized": 0.20,
        "center_diff_normalized": 0.0,
        "king_safety_diff_normalized": 0.10,
        "development_diff_normalized": 0.0,
    }
    return linear_weighted_heuristic(state, player, weights)


class HeuristicAlphaBetaBot(ChessBot):
    """
    Chess bot using heuristic alpha-beta search with configurable parameters.
    """

    def __init__(self, eval_function: Callable[[ChessGameState, chess.Color], float]=h1,
                 max_depth: int = 3):
        """
        Initialize the heuristic alpha-beta bot.

        :param eval_function: Evaluation function (state, player) -> float
        :type eval_function: Callable[[ChessGameState, chess.Color], float]
        :param max_depth: Maximum search depth
        :type max_depth: int
        """
        super().__init__("HeuristicAlphaBetaBot", "david.perezperez@usp.ceu.es")
        self.eval_function = eval_function
        self.max_depth = max_depth

    def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Get the best move using heuristic alpha-beta search.

        :param board: Current chess position
        :type board: chess.Board
        :param time_limit: Maximum time to think in seconds
        :type time_limit: float
        :return: Best move found
        :rtype: chess.Move
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()

        game = ChessProblem(board)
        state = ChessGameState(board)
        player = board.turn

        max_depth = self.max_depth

        def cutoff_test(s: ChessGameState, depth: int, elapsed_time: float) -> bool:
            return depth >= max_depth or game.is_terminal(s)

        try:
            best_move, _stats = heuristic_alphabeta_search(
                game=game,
                state=state,
                eval_fn=self.eval_function,
                cutoff_test=cutoff_test,
            )
            if best_move is None:
                return legal_moves[0]
            return best_move
        except Exception:
            return legal_moves[0]
