import numpy as np
import random

class SudokuEnv:
    def __init__(self, size=3):

        # 初始化数独棋盘。难度等级 hidden_level 0.6，表示移除 60% 的数字
        self.size = size
        self.hidden_level = 0.5
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.solution = self.board.copy()
        self.origin_puzzle = self.board.copy()
        self.backtrack_actions = []
        self.correct_actions = []
        self.reset()

        # 词汇表/动作空间
        self.vocab = self._init_vocab()

    def _init_vocab(self):
        vocab = [
            "<PAD>",    # 填充标记。第0位约定是<PAD> 
            "<START>",  # 开始标记
            "<END>",    # 结束标记
        ]
        # 棋盘上每个位置都有可能填入0~9的数字,如第1行第2列填入9，则动作标签为 "019"
        for i in range(self.size):
            for j in range(self.size):
                for num in range(0, self.size + 1):
                    vocab.append(f"{i}{j}{num}")

        return vocab
    
    def reset(self):
        # 生成一个有效的数独解,保存完整解
        self._generate_solution()
        self.solution = self.board.copy()

        # 随机移除一些数字来创建谜题，保存谜题
        self._create_puzzle()
        self.origin_puzzle = self.board.copy()

    def _backtrack_solve(self, board=None, randomize=False):
        """通用的回溯法求解函数，可用于生成解和求解谜题
        
        Args:
            board: 可选，提供要解决的棋盘，默认使用当前棋盘
            randomize: 是否随机化数字顺序，用于生成解时使用
            
        Returns:
            success: 布尔值，指示是否成功解决
            board: 解决后的棋盘，如果无解则为None
            backtrack_actions: 所有操作步骤，包括回溯尝试，格式为[(行,列,数字),...]
            correct_actions: 最终解决方案的正确步骤，格式为[(行,列,数字),...]
        """
        # 如果提供了棋盘参数，使用它，否则使用当前棋盘的副本
        if board is None:
            working_board = self.board.copy()
        else:
            working_board = board.copy()
            
        # 初始化动作记录列表
        backtrack_actions = []  # 记录所有尝试的动作，包括回溯
        correct_actions = []    # 只记录最终解决方案的正确步骤
        
        # 使用回溯法求解
        def is_valid(board, row, col, num):
            # 检查行
            if num in board[row]:
                return False
            # 检查列
            if num in board[:, col]:
                return False
            # 检查块
            block_size = int(self.size**0.5)
            block_row = (row // block_size) * block_size
            block_col = (col // block_size) * block_size
            for i in range(block_row, block_row + block_size):
                for j in range(block_col, block_col + block_size):
                    if board[i][j] == num:
                        return False
            return True
            
        def backtrack(board):
            # 找到一个空格子
            for i in range(self.size):
                for j in range(self.size):
                    if board[i][j] == 0:  # 空格子
                        # 准备要尝试的数字
                        numbers = list(range(1, self.size + 1))
                        if randomize:
                            random.shuffle(numbers)  # 随机打乱数字顺序用于生成解
                            
                        # 尝试每个数字
                        for num in numbers:
                            if is_valid(board, i, j, num):
                                # 放置数字
                                board[i][j] = num
                                # 记录这个放置动作到回溯列表
                                backtrack_actions.append((i, j, num))
                                # 添加到正确动作列表
                                correct_actions.append((i, j, num))
                                # 递归尝试解决剩余部分
                                if backtrack(board):
                                    return True
                                # 回溯，记录移除动作
                                backtrack_actions.append((i, j, 0))
                                # 从正确动作列表中移除
                                correct_actions.pop()
                                # 重置该单元格
                                board[i][j] = 0
                        # 尝试了所有数字都不行，返回失败
                        return False
            # 所有格子都已填满，解决成功
            return True
            
        # 尝试解决数独
        success = backtrack(working_board)
        
        if success:
            return True, working_board, backtrack_actions, correct_actions
        else:
            return False, None, backtrack_actions, []  # 如果失败，正确动作列表为空

    def _generate_solution(self):
        """使用回溯法生成一个有效的数独解，加入随机性"""
        
        # 清空棋盘
        self.board = np.zeros((self.size, self.size), dtype=int)
        
        # 使用通用回溯函数生成解，设置randomize=True使用随机顺序
        success, solved_board, _, _ = self._backtrack_solve(self.board, randomize=True)
        
        if success:
            self.board = solved_board
            return True
        else:
            # 理论上生成解不应该失败
            print("警告: 生成数独解失败")
            return False

    def _create_puzzle(self):
        """根据难度等级移除一些数字来创建数独谜题"""

        # 计算需要移除的数字个数
        total_cells = self.size * self.size
        cells_to_remove = int(total_cells * self.hidden_level)

        # 随机选择位置移除数字
        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        positions_to_remove = random.sample(positions, cells_to_remove)

        # 记录移除动作
        remove_actions = []
        for pos in positions_to_remove:
            number = self.board[pos[0]][pos[1]]
            self.board[pos[0]][pos[1]] = 0
            action = [pos[0],pos[1],number]
            remove_actions.append(action)
        
        correct_actions = list(reversed(remove_actions))
        return correct_actions

    def _check_solution(self):
        """检测当前的棋盘是否是正确的解，记录，每行、列、块是否正确，计算reward"""

        def _check_row(row):
            """检查某一行是否有效"""
            numbers = set(range(1, self.size + 1))
            return set(self.board[row]) == numbers
        
        def _check_column(col):
            """检查某一列是否有效"""
            numbers = set(range(1, self.size + 1))
            return set(self.board[:, col]) == numbers
        
        def _check_box(start_i, start_j):
            """检查3x3方格是否有效,棋盘小就不检查"""
            if self.size < 9:
                return True
            
            numbers = set(range(1, self.size + 1))
            box_numbers = set()
            for i in range(start_i, start_i + 3):
                for j in range(start_j, start_j + 3):
                    box_numbers.add(self.board[i][j])
            return box_numbers == numbers
 
        # 检查每一行、列、块
        _count = 0
        _reward = 0
        for i in range(self.size):
            if _check_row(i): _reward += 1
            _count += 1
        # 检查每一列
        for j in range(self.size):
            if _check_column(j): _reward += 1
            _count += 1
        
        # 检查每一块
        for block_row in range(0, self.size, int(self.size**0.5)):
            for block_col in range(0, self.size, int(self.size**0.5)):
                if _check_box(block_row, block_col): _reward += 1
                _count += 1

        return _reward/_count
    
    def _check_number(self, row, col, number):
        # 不能修改原始谜题 origin_puzzle 的数字(非0的)
        can_fill = False
        if self.origin_puzzle[row][col] == 0:
            can_fill = True
        
        if self.origin_puzzle[row][col] == number:
            can_fill = True

        return can_fill
        
    def play(self, state_tokens=None, action_tokens=None, is_print = False):

        if state_tokens is not None:
            self.set_state_by_tokens(state_tokens)
            self.origin_puzzle = self.board.copy()

        #######################
        ### Reward Function ###
        #######################

        reward = -1

        actions = [self._token_to_action(token) for token in action_tokens]

        # 打印预测的解法
        if is_print:
            print("谜题: ")
            self.render()
            board = self.board.copy()
            # 把数字按序列填入棋盘
            for action in actions:
                i, j, number = action
                if i is not None and j is not None and number is not None:
                    board[i][j] = number
            print("预测的解法: ")
            self.render(board)

        # 做题
        for action in actions:
            i, j, number = action
            # 避免报错
            if i is not None and j is not None and number is not None:
                # 检查动作合法性
                if self._check_number(i, j, number):
                    self.board[i][j] = number
                else:
                    if is_print: print(f"非法动作: 在原始谜题位置({i}, {j})填入数字{number}")
                    reward = -1
                    return reward
                
        # 检查
        reward = self._check_solution()
        if reward == 1:
            if is_print: print("成功！！！！！！！！")
            reward = 1
            return reward
        else:
            if is_print: print("失败...")
            reward = -1
            #reward = round(reward * 0.5 - 0.5, 3)
            return reward

    def render(self, board=None):
        """打印当前数独棋盘"""
        if board is None: board = self.board

        print("----------↓")
        for i in range(self.size):
            for j in range(self.size):
                if j == self.size - 1:
                    print(board[i][j])
                else:
                    print(str(board[i][j]) + " ", end="")
        print("----------↑")

    def get_state_tokens(self, board=None):
        if board is None: board = self.board

        # 遍历棋盘上的所有数字
        state_tokens = []
        for i in range(self.size):
            for j in range(self.size):
                number = board[i][j]
                token = self._action_to_token(i,j,number)
                state_tokens.append(token)
        return state_tokens
    
    def set_state_by_tokens(self, tokens):
        # 把数字按序列填入棋盘
        for token in tokens:
            i, j, number = self._token_to_action(token)
            if i is not None and j is not None and number is not None:
                self.board[i][j] = number
    
    def _action_to_token(self, i, j, number):
        t = f"{i}{j}{number}"
        token = self.vocab.index(t)
        return token
    
    def _token_to_action(self, token):
        # 避免报错
        if token < len(self.vocab) and token > 2: # 排除特殊标签
            str_token = self.vocab[token]
            # 字符串转数字
            i = int(str_token[0]) 
            j = int(str_token[1])
            number = int(str_token[-1])
            return i, j, number
        else:
            return None, None, None

    def roll_cold_start_data(self, num_roll = 10):
        """生成指定数量的冷启动数据(正确解)"""

        cold_start_data = []
        for i in range(num_roll):

            self.hidden_level = 0.5 - 0.3 * (i / num_roll) #0.5~0.2之间

            self.reset()
            state_tokens = self.get_state_tokens()
            success, solved_board, backtrack_actions, correct_actions = self._backtrack_solve()

            action_tokens = []
            for action in backtrack_actions:
                action_tokens.append(self._action_to_token(action[0], action[1], action[2]))

            cold_start_data.append({"question": state_tokens, "answer": action_tokens, "reward": 1})

        return cold_start_data
