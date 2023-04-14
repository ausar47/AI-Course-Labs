import random
import math
import time
from copy import deepcopy
from board import Board

class Node:
    def __init__(self, board, color, parent=None, prev=None):
        self.board = board
        self.color = color
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.moves = list(board.get_legal_actions(self.color))
        self.prev = prev

    def ucb1(self, exploration_param=1):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration_param * math.sqrt(2 * math.log(self.parent.visits) / self.visits)

    def expand(self):
        n = random.randrange(len(self.moves))
        move = self.moves[n]
        self.moves.remove(move)
        child_board = deepcopy(self.board)
        child_board._move(move, self.color)
        child_color = "X" if self.color == "O" else "O"
        child = Node(child_board, child_color, self, move)
        self.children.append(child)

    def is_fully_expanded(self) -> bool:
        return len(self.moves) == 0

    def is_terminal(self) -> bool:
        return len(self.moves) == 0 and len(self.children) == 0

    def best_child(self, exploration_param=1):
        return max(self.children, key=lambda child: child.ucb1(exploration_param))


class MCTSearch:
    def __init__(self, board, color, exploration_param=1, max_iterations=100000):
        self.board = board
        self.color = color
        self.exploration_param = exploration_param
        self.max_iterations = max_iterations
        self.max_time = 59
        self.iter_times = 0
        self.root = Node(self.board, self.color)
        self.start_time = 0

    def backpropagate(self, node, result):
        cur = node
        while cur is not None:
            cur.visits += 1
            if cur.color == self.color:
                cur.value -= result
            else:
                cur.value += result
            cur = cur.parent

    def search(self):
        self.start_time = time.time()
        while time.time() - self.start_time < self.max_time and self.iter_times < self.max_iterations:
            node = self.select(self.root)
            result = self.simulate(node)
            self.backpropagate(node, result)
            self.iter_times += 1
        return self.root.best_child(0).prev

    def select(self, node):
        while not node.is_terminal():
            if not node.is_fully_expanded():
                node.expand()
                return node.children[-1]
            else:
                node = node.best_child(self.exploration_param)
        return node

    def is_game_over(self, board) -> bool:
        if board.count('.') == 0:
            return True
        b = list(board.get_legal_actions('X'))
        w = list(board.get_legal_actions('O'))
        if not b and not w:
            return True
        return False

    def simulate(self, node):
        cur_board = deepcopy(node.board)
        cur_color = node.color
        while not self.is_game_over(cur_board):
            moves = list(cur_board.get_legal_actions(cur_color))
            if len(moves) != 0:
                move = random.choice(moves)
                cur_board._move(move, cur_color)
            cur_color = "X" if cur_color == "O" else "O"
        tag = ['X', 'O', 'X' if self.color == 'O' else 'O']
        winner = tag[cur_board.get_winner()[0]]
        return winner == self.color

if __name__ == '__main__':
    board = Board()
    board.display()
    mcts = MCTSearch(board, "X")
    print(mcts.search())