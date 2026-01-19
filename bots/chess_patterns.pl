% chess_patterns.pl
% Prolog knowledge base for tactical pattern detection in chess

% Dynamic predicates that will be asserted from Python
:- dynamic piece/4.        % piece(Type, Color, Row, Col)
:- dynamic move/4.         % move(FromRow, FromCol, ToRow, ToCol)

% Define opposite colors
opposite_color(white, black).
opposite_color(black, white).

% Piece values (typical values, king has very high value)
piece_value(p, 1).      % Pawn
piece_value(n, 3).      % Knight
piece_value(b, 3).      % Bishop
piece_value(r, 5).      % Rook
piece_value(q, 9).      % Queen
piece_value(k, 100). % King

% MVV-LVA piece values (for capture scoring, king has practical value)
mvv_lva_piece_value(p, 1).  % Pawn
mvv_lva_piece_value(n, 3).  % Knight
mvv_lva_piece_value(b, 3).  % Bishop
mvv_lva_piece_value(r, 5).  % Rook
mvv_lva_piece_value(q, 9).  % Queen
mvv_lva_piece_value(k, 10). % King

% Square relationships
same_row(R, R).
same_col(C, C).

same_diagonal(R1, C1, R2, C2) :-
    Diff1 is R2 - R1,
    Diff2 is C2 - C1,
    abs(Diff1) =:= abs(Diff2).

% Line relationships (for pins and skewers)
on_same_line(R1, C1, R2, C2, R3, C3) :-
    same_row(R1, R2),
    same_row(R2, R3),
    ((C1 < C2, C2 < C3); (C3 < C2, C2 < C1)).

on_same_line(R1, C1, R2, C2, R3, C3) :-
    same_col(C1, C2),
    same_col(C2, C3),
    ((R1 < R2, R2 < R3); (R3 < R2, R2 < R1)).

on_same_line(R1, C1, R2, C2, R3, C3) :-
    same_diagonal(R1, C1, R2, C2),
    same_diagonal(R2, C2, R3, C3),
    ((R1 < R2, R2 < R3, C1 < C2, C2 < C3);
     (R3 < R2, R2 < R1, C3 < C2, C2 < C1);
     (R1 < R2, R2 < R3, C1 > C2, C2 > C3);
     (R3 < R2, R2 < R1, C3 > C2, C2 > C1)).

% Generate intermediate squares on a diagonal between two points
% diagonal_between(R1, C1, R2, C2, Ri, Ci)
% Works like between/3 but for diagonal coordinates
diagonal_between(R1, C1, R2, C2, Ri, Ci) :-
    same_diagonal(R1, C1, R2, C2),
    DiffR is R2 - R1,
    DiffC is C2 - C1,
    abs(DiffR) =:= abs(DiffC),
    Distance is abs(DiffR),
    StepR is sign(DiffR),
    StepC is sign(DiffC),
    % Generate step numbers from 1 to Distance-1 (exclude endpoints)
    between(1, Distance, Step),
    Step < Distance,
    % Calculate intermediate square
    Ri is R1 + Step * StepR,
    Ci is C1 + Step * StepC.

% Piece movement capabilities
can_attack_along_line(r, row).
can_attack_along_line(r, col).
can_attack_along_line(q, row).
can_attack_along_line(q, col).
can_attack_along_line(q, diagonal).
can_attack_along_line(b, diagonal).

% Check if path is clear between two squares
path_clear(R1, C1, R2, C2) :-
    R1 =:= R2, !,  % Same row
    C3 is min(C1, C2) + 1,
    C4 is max(C1, C2) - 1,
    \+ (between(C3, C4, C), piece(_, _, R1, C)).

path_clear(R1, C1, R2, C2) :-
    C1 =:= C2, !,  % Same column
    R3 is min(R1, R2) + 1,
    R4 is max(R1, R2) - 1,
    \+ (between(R3, R4, R), piece(_, _, R, C1)).

path_clear(R1, C1, R2, C2) :-
    same_diagonal(R1, C1, R2, C2),
    \+ (diagonal_between(R1, C1, R2, C2, Ri, Ci),
        piece(_, _, Ri, Ci)).


% Basic attack patterns for different pieces
can_attack(p, white, R, C, R2, C2) :-
    R2 is R + 1,
    (C2 is C + 1; C2 is C - 1),
    piece(_, black, R2, C2).

can_attack(p, black, R, C, R2, C2) :-
    R2 is R - 1,
    (C2 is C + 1; C2 is C - 1),
    piece(_, white, R2, C2).

can_attack(n, _, R, C, R2, C2) :-
    knight_move(R, C, R2, C2).

can_attack(b, _, R1, C1, R2, C2) :-
    same_diagonal(R1, C1, R2, C2),
    path_clear(R1, C1, R2, C2).

can_attack(r, _, R, C, R2, C2) :-
    (same_row(R, R2); same_col(C, C2)),
    path_clear(R, C, R2, C2).

can_attack(q, _, R1, C1, R2, C2) :-
    (same_row(R1, R2); same_col(C1, C2); same_diagonal(R1, C1, R2, C2)),
    path_clear(R1, C1, R2, C2).

can_attack(k, _, R, C, R2, C2) :-
    abs(R2 - R) =< 1,
    abs(C2 - C) =< 1,
    (R, C) \= (R2, C2).

% Knight movement pattern
knight_move(R1, C1, R2, C2) :-
    DR is abs(R2 - R1),
    DC is abs(C2 - C1),
    ((DR =:= 2, DC =:= 1); (DR =:= 1, DC =:= 2)).

% Helper: Check if path is clear between two squares, ignoring a specific square
path_clear_ignoring(R1, C1, R2, C2, IgnoreR, IgnoreC) :-
    R1 =:= R2, !,  % Same row
    C3 is min(C1, C2) + 1,
    C4 is max(C1, C2) - 1,
    \+ (between(C3, C4, C),
        piece(_, _, R1, C),
        (R1, C) \= (IgnoreR, IgnoreC)).

path_clear_ignoring(R1, C1, R2, C2, IgnoreR, IgnoreC) :-
    C1 =:= C2, !,  % Same column
    R3 is min(R1, R2) + 1,
    R4 is max(R1, R2) - 1,
    \+ (between(R3, R4, R),
        piece(_, _, R, C1),
        (R, C1) \= (IgnoreR, IgnoreC)).

path_clear_ignoring(R1, C1, R2, C2, IgnoreR, IgnoreC) :-
    same_diagonal(R1, C1, R2, C2),
    \+ (diagonal_between(R1, C1, R2, C2, Ri, Ci),
        piece(_, _, Ri, Ci),
        (Ri, Ci) \= (IgnoreR, IgnoreC)).

% Helper: Check if a piece at a given square can attack a target, ignoring a specific square
can_attack_ignoring(PieceType, Color, FromR, FromC, ToR, ToC, _IgnoreR, _IgnoreC) :-
    PieceType = p, Color = white, !,
    ToR is FromR + 1,
    (ToC is FromC + 1; ToC is FromC - 1),
    piece(_, black, ToR, ToC).

can_attack_ignoring(PieceType, Color, FromR, FromC, ToR, ToC, _IgnoreR, _IgnoreC) :-
    PieceType = p, Color = black, !,
    ToR is FromR - 1,
    (ToC is FromC + 1; ToC is FromC - 1),
    piece(_, white, ToR, ToC).

can_attack_ignoring(PieceType, _, FromR, FromC, ToR, ToC, _IgnoreR, _IgnoreC) :-
    PieceType = n, !,
    knight_move(FromR, FromC, ToR, ToC).

can_attack_ignoring(PieceType, _, FromR, FromC, ToR, ToC, IgnoreR, IgnoreC) :-
    PieceType = b, !,
    same_diagonal(FromR, FromC, ToR, ToC),
    path_clear_ignoring(FromR, FromC, ToR, ToC, IgnoreR, IgnoreC).

can_attack_ignoring(PieceType, _, FromR, FromC, ToR, ToC, IgnoreR, IgnoreC) :-
    PieceType = r, !,
    (same_row(FromR, ToR); same_col(FromC, ToC)),
    path_clear_ignoring(FromR, FromC, ToR, ToC, IgnoreR, IgnoreC).

can_attack_ignoring(PieceType, _, FromR, FromC, ToR, ToC, IgnoreR, IgnoreC) :-
    PieceType = q, !,
    (same_row(FromR, ToR); same_col(FromC, ToC); same_diagonal(FromR, FromC, ToR, ToC)),
    path_clear_ignoring(FromR, FromC, ToR, ToC, IgnoreR, IgnoreC).

can_attack_ignoring(PieceType, _, FromR, FromC, ToR, ToC, _IgnoreR, _IgnoreC) :-
    PieceType = k,
    abs(ToR - FromR) =< 1,
    abs(ToC - FromC) =< 1,
    (FromR, FromC) \= (ToR, ToC).

%==============================================================================
% MOVE-BASED PATTERN DETECTION
% These predicates check if a move creates a tactical pattern without
% actually updating the board state. They assume the piece exists at the
% destination square while ignoring interference from the source square.
%==============================================================================

% Check if a move creates an absolute pin and return its score
% An absolute pin occurs when a piece cannot move without exposing the king to check
% Tarea 6
move_creates_absolute_pin(FromR, FromC, ToR, ToC, PinScore) :-
    move(FromR, FromC, ToR, ToC),
    piece(PinnerType, Color, FromR, FromC),
    can_attack_along_line(PinnerType, _),
    opposite_color(Color, EnemyColor),
    piece(PinnedType, EnemyColor, PR, PC),
    piece(k, EnemyColor, KR, KC),
    (PR, PC) \= (FromR, FromC),
    (KR, KC) \= (FromR, FromC),
    on_same_line(ToR, ToC, PR, PC, KR, KC),
    path_clear_ignoring(ToR, ToC, PR, PC, FromR, FromC),
    path_clear_ignoring(PR, PC, KR, KC, FromR, FromC),
    piece_value(PinnedType, PinnedValue),
    piece_value(k, KingValue),
    piece_value(PinnerType, PinnerValue),
    PinScore is (PinnedValue + KingValue - PinnerValue) // 2.

% Check if a move creates a relative pin and return its score
% A relative pin occurs when a piece cannot move without exposing a more valuable piece
% Simplifying assumptions: TargetValue > PinnedValue AND TargetValue > PinnerValue
% Tarea 7
move_creates_relative_pin(FromR, FromC, ToR, ToC, PinScore) :-
    move(FromR, FromC, ToR, ToC),
    piece(PinnerType, Color, FromR, FromC),
    can_attack_along_line(PinnerType, _),
    opposite_color(Color, EnemyColor),
    piece(PinnedType, EnemyColor, PR, PC),
    piece(TargetType, EnemyColor, TR, TC),
    TargetType \= k,
    (PR, PC) \= (TR, TC),
    (PR, PC) \= (FromR, FromC),
    (TR, TC) \= (FromR, FromC),
    on_same_line(ToR, ToC, PR, PC, TR, TC),
    path_clear_ignoring(ToR, ToC, PR, PC, FromR, FromC),
    path_clear_ignoring(PR, PC, TR, TC, FromR, FromC),
    piece_value(PinnedType, PinnedValue),
    piece_value(TargetType, TargetValue),
    piece_value(PinnerType, PinnerValue),
    TargetValue > PinnedValue,
    TargetValue > PinnerValue,
    PinScore is (PinnedValue + TargetValue - PinnerValue) // 2.

% Check if a move creates a fork and return its score
% A fork occurs when the moved piece attacks two or more valuable opponent pieces
% Tarea 5
move_creates_fork(FromR, FromC, ToR, ToC, ForkScore) :-
    move(FromR, FromC, ToR, ToC),
    piece(AttackerType, Color, FromR, FromC),
    opposite_color(Color, TargetColor),
    piece(Target1Type, TargetColor, R1, C1),
    piece(Target2Type, TargetColor, R2, C2),
    (R1, C1) \= (R2, C2),
    (R1, C1) \= (FromR, FromC),
    (R2, C2) \= (FromR, FromC),
    (Target1Type = k; Target1Type = q; Target1Type = r),
    (Target2Type = k; Target2Type = q; Target2Type = r),
    can_attack_ignoring(AttackerType, Color, ToR, ToC, R1, C1, FromR, FromC),
    can_attack_ignoring(AttackerType, Color, ToR, ToC, R2, C2, FromR, FromC),
    piece_value(Target1Type, Target1Value),
    piece_value(Target2Type, Target2Value),
    piece_value(AttackerType, AttackerValue),
    ForkScore is Target1Value + Target2Value - AttackerValue.

% Check if a move creates a skewer and return its score
% A skewer forces a valuable piece to move, exposing a less valuable piece behind it
% Tarea 4
move_creates_skewer(FromR, FromC, ToR, ToC, SkewerScore) :-
    move(FromR, FromC, ToR, ToC),
    piece(AttackerType, Color, FromR, FromC),
    can_attack_along_line(AttackerType, _),
    opposite_color(Color, TargetColor),
    piece(FrontType, TargetColor, R2, C2),
    piece(BehindType, TargetColor, R3, C3),
    (R2, C2) \= (R3, C3),
    (FrontType = k; FrontType = q; FrontType = r),
    on_same_line(ToR, ToC, R2, C2, R3, C3),
    path_clear_ignoring(ToR, ToC, R2, C2, FromR, FromC),
    path_clear_ignoring(R2, C2, R3, C3, FromR, FromC),
    piece_value(FrontType, FrontValue),
    piece_value(BehindType, BehindValue),
    piece_value(AttackerType, AttackerValue),
    SkewerScore is FrontValue + BehindValue - AttackerValue.

% Check if a move is a capture and return MVV-LVA score
% MVV-LVA = Most Valuable Victim - Least Valuable Attacker
% Tarea 2
move_creates_capture(FromR, FromC, ToR, ToC, MvvLvaScore) :-
    move(FromR, FromC, ToR, ToC),
    piece(AttackerType, AttackerColor, FromR, FromC),
    piece(VictimType, VictimColor, ToR, ToC),
    opposite_color(AttackerColor, VictimColor),
    mvv_lva_piece_value(VictimType, VictimValue),
    mvv_lva_piece_value(AttackerType, AttackerValue),
    MvvLvaScore is (VictimValue * 10) - AttackerValue.

% Check if a move is a pawn promotion
% Tarea 3
move_creates_promotion(FromR, FromC, ToR, _ToC) :-
    piece(p, white, FromR, FromC),
    ToR =:= 8.
move_creates_promotion(FromR, FromC, ToR, _ToC) :-
    piece(p, black, FromR, FromC),
    ToR =:= 1.

