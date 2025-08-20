#include <vector>
#include <random>
#include <algorithm>

extern "C" {
    // 游戏状态结构
    struct GameState {
        int board[4][4];
        int score;
    };

    // 初始化游戏
    GameState* game_init() {
        GameState* state = new GameState();
        std::memset(state->board, 0, sizeof(state->board));
        state->score = 0;
        // 添加初始块
        add_random_tile(state);
        add_random_tile(state);
        return state;
    }

    // 添加随机方块
    void add_random_tile(GameState* state) {
        std::vector<std::pair<int, int>> empty;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (state->board[i][j] == 0) {
                    empty.push_back({i, j});
                }
            }
        }

        if (!empty.empty()) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, empty.size()-1);
            auto pos = empty[dist(gen)];
            state->board[pos.first][pos.second] = (dist(gen) % 10 == 0) ? 4 : 2;
        }
    }

    // 移动操作核心
    int move(GameState* state, int direction) {
        int score_add = 0;
        int (*board)[4] = state->board;
        int temp[4];

        // 0:上, 1:右, 2:下, 3:左
        auto process_line = [&](int* line) {
            int write_pos = 0;
            int last = -1;
            for (int read_pos = 0; read_pos < 4; read_pos++) {
                if (line[read_pos] != 0) {
                    if (last == line[read_pos]) {
                        temp[write_pos-1] *= 2;
                        score_add += temp[write_pos-1];
                        last = -1;
                    } else {
                        temp[write_pos] = line[read_pos];
                        last = temp[write_pos];
                        write_pos++;
                    }
                }
            }
            for (int i = write_pos; i < 4; i++) temp[i] = 0;
            std::copy(temp, temp+4, line);
        };

        // 根据方向处理
        if (direction == 0) { // 上
            for (int col = 0; col < 4; col++) {
                int line[4] = {board[0][col], board[1][col], board[2][col], board[3][col]};
                process_line(line);
                for (int i = 0; i < 4; i++) board[i][col] = line[i];
            }
        }
        // 其他方向类似处理...

        state->score += score_add;
        return score_add;
    }

    // 检查游戏状态
    int check_state(GameState* state) {
        // 0: 进行中, 1: 胜利, 2: 失败
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (state->board[i][j] == 2048) return 1; // 胜利
            }
        }

        // 检查空格
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (state->board[i][j] == 0) return 0;
            }
        }

        // 检查可合并
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                if (state->board[i][j] == state->board[i][j+1]) return 0;
            }
        }
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 3; i++) {
                if (state->board[i][j] == state->board[i+1][j]) return 0;
            }
        }

        return 2; // 失败
    }

    // 释放资源
    void free_game(GameState* state) {
        delete state;
    }
}