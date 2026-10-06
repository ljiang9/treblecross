"""treblecross —— 一维三连棋。

两人轮流在长度为 n 的一维棋盘上放 X（双方共用 X）。
谁先造出三个连续的 X，谁获胜。

Treblecross 的已知结论：
- 棋盘长度 n = 4k+3 时，先手必胜：占中后镜像对称。
- 小棋盘可用 minimax 精确求解。

本实现：n <= 13 用 alpha-beta minimax 精确求解；
更大的棋盘用启发式（先抢中心、避免送对手即时胜）。
纯标准库，无依赖。
"""

from __future__ import annotations

import argparse
import random
import sys
from functools import lru_cache


def new_board(n: int) -> tuple:
    return tuple(0 for _ in range(n))


def has_three(board: tuple) -> bool:
    """棋盘上是否存在三个连续的 X。"""
    return any(
        board[i] and board[i + 1] and board[i + 2]
        for i in range(len(board) - 2)
    )


def legal_moves(board: tuple) -> list:
    return [i for i, v in enumerate(board) if not v]


def apply_move(board: tuple, i: int) -> tuple:
    if not 0 <= i < len(board):
        raise ValueError(f"位置 {i} 超出棋盘范围")
    if board[i]:
        raise ValueError(f"位置 {i} 已被占用")
    b = list(board)
    b[i] = 1
    return tuple(b)


def game_result(board: tuple) -> int | None:
    """返回 1 表示刚走完的一方获胜，0 表示和棋（棋盘满），None 表示未结束。"""
    if has_three(board):
        return 1
    if not legal_moves(board):
        return 0
    return None


def render(board: tuple) -> str:
    cells = ["X" if v else "." for v in board]
    head = " ".join(f"{i:2d}" for i in range(len(board)))
    return head + "\n" + " ".join(f"{c:>2s}" for c in cells)


# ---------- 精确求解（小棋盘） ----------

@lru_cache(maxsize=None)
def _solve(board: tuple) -> int:
    """返回当前走棋方在最优对局下的结果：1 胜 / 0 和 / -1 负。"""
    # 对手上一步是否已胜出？检查在走这一步之前。
    # 递归中我们每走一步就检查胜负，所以这里只需看是否还有走法。
    best = -1  # 假设败
    for i in legal_moves(board):
        nb = apply_move(board, i)
        if has_three(nb):
            return 1
        v = _solve(nb)
        if v == -1:
            return 1  # 对手必败，我必胜
        if v == 0:
            best = 0
    return best


def solve(board: tuple) -> int:
    return _solve(board)


def best_move_exact(board: tuple) -> int:
    moves = legal_moves(board)
    random.shuffle(moves)
    # 1. 即时胜
    for i in moves:
        if has_three(apply_move(board, i)):
            return i
    # 2. minimax 精确值（自动包含攻防，因为 _solve 考虑了对手最优应对）
    scored = []
    for i in moves:
        nb = apply_move(board, i)
        if has_three(nb):
            return i
        scored.append((-_solve(nb), i))
    scored.sort()
    return scored[0][1]


def best_move_heuristic(board: tuple, rng: random.Random) -> int:
    """大棋盘启发式：中心优先 + 避免送即时胜。"""
    n = len(board)
    moves = legal_moves(board)
    # 即时胜
    for i in moves:
        if has_three(apply_move(board, i)):
            return i
    # 防守：对手下回合能胜的格子
    must_block = set()
    for i in moves:
        nb = apply_move(board, i)
        for j in legal_moves(nb):
            if has_three(apply_move(nb, j)):
                must_block.add(j)
                break
    safe = [i for i in moves if i in must_block] or [
        i for i in moves if i not in _gives_opp_win(board, moves)
    ] or moves
    center = (n - 1) / 2
    safe.sort(key=lambda i: (abs(i - center), rng.random()))
    return safe[0]


def _gives_opp_win(board: tuple, moves: list) -> set:
    bad = set()
    for i in moves:
        nb = apply_move(board, i)
        if has_three(nb):
            continue
        for j in legal_moves(nb):
            if has_three(apply_move(nb, j)):
                bad.add(i)
                break
    return bad


def ai_move(board: tuple, rng: random.Random) -> int:
    if len(board) <= 13:
        return best_move_exact(board)
    return best_move_heuristic(board, rng)


# ---------- 自动对局 ----------

def auto_game(n: int, seed: int, verbose: bool = False) -> int:
    """AI 对 AI 一局。返回 1/2 表示获胜方，0 和棋。"""
    rng = random.Random(seed)
    board = new_board(n)
    turn = 1
    while True:
        i = ai_move(board, rng)
        board = apply_move(board, i)
        if verbose:
            print(f"玩家 {turn} 走 {i}")
            print(render(board))
            print()
        r = game_result(board)
        if r == 1:
            return turn
        if r == 0:
            return 0
        turn = 3 - turn


def auto_games(n: int, games: int, seed: int) -> dict:
    stats = {1: 0, 2: 0, 0: 0}
    for g in range(games):
        w = auto_game(n, seed + g)
        stats[w] += 1
    return stats


# ---------- 交互 ----------

def parse_move(text: str, board: tuple) -> int:
    text = text.strip()
    if not text.lstrip("-").isdigit():
        raise ValueError("请输入数字坐标")
    i = int(text)
    if not 0 <= i < len(board):
        raise ValueError(f"坐标超出范围 0..{len(board) - 1}")
    if board[i]:
        raise ValueError(f"位置 {i} 已被占用")
    return i


def play_interactive(n: int, seed: int | None) -> None:
    rng = random.Random(seed)
    board = new_board(n)
    print(f"Treblecross：一维三连棋，棋盘长度 {n}")
    print("规则：两人轮流放 X，谁先造出三个连续的 X 谁胜。")
    print("你是玩家 1，AI 是玩家 2。输入坐标走棋，q 退出。")
    print(render(board))
    turn = 1
    while True:
        if turn == 1:
            try:
                text = input("你的走法> ").strip()
            except EOFError:
                print("\n再见。")
                return
            if text.lower() in ("q", "quit", "exit"):
                print("再见。")
                return
            try:
                i = parse_move(text, board)
            except ValueError as e:
                print(f"非法走法：{e}")
                continue
        else:
            i = ai_move(board, rng)
            print(f"AI 走 {i}")
        board = apply_move(board, i)
        print(render(board))
        r = game_result(board)
        if r == 1:
            print(f"玩家 {turn} 获胜！")
            return
        if r == 0:
            print("和棋，棋盘已满。")
            return
        turn = 3 - turn


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="treblecross 一维三连棋")
    ap.add_argument("-n", "--size", type=int, default=11, help="棋盘长度（默认 11）")
    ap.add_argument("--auto", action="store_true", help="AI 对 AI 自动演示")
    ap.add_argument("--games", type=int, default=10, help="自动对局数（默认 10）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--verbose", action="store_true", help="自动模式打印每一步")
    args = ap.parse_args(argv)

    if args.size < 3:
        print("棋盘长度至少为 3", file=sys.stderr)
        return 2

    if args.auto:
        stats = auto_games(args.size, args.games, args.seed)
        print(f"自动演示结束：共 {args.games} 局（棋盘长度 {args.size}），"
              f"先手胜 {stats[1]}，后手胜 {stats[2]}，和棋 {stats[0]}")
        return 0

    if not sys.stdin.isatty():
        print("交互模式需要终端；无头演示请用 --auto", file=sys.stderr)
        return 2
    play_interactive(args.size, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
