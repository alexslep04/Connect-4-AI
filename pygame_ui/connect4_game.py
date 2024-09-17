import pygame
import numpy as np

# Constants for the game
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARE_SIZE = 100  # Size of each square on the grid
RADIUS = int(SQUARE_SIZE / 2 - 5)  # Radius of the tokens
BLUE = (0, 0, 255)  # Color for the board
BLACK = (0, 0, 0)  # Background color for empty spaces
RED = (255, 0, 0)  # Color for Player 1
YELLOW = (255, 255, 0)  # Color for Player 2

class Connect4Game:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
        self.current_player = 1
        self.game_over = False

        if self.render_mode:
            pygame.init()
            # Initialize rendering components
            self.width = COLUMN_COUNT * SQUARE_SIZE
            self.height = (ROW_COUNT + 1) * SQUARE_SIZE
            self.size = (self.width, self.height)
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("Connect 4")
            self.draw_board()

    # Update other methods accordingly
    def render(self):
        if self.render_mode:
            # Rendering code goes here
            pass  # Replace with your rendering implementation

    def close(self):
        if self.render_mode:
            pygame.quit()

    def drop_token(self, col):
        """
        Drop a token into the specified column and switch player.
        """
        try:
            # Drop the token in the next open row
            row = self._get_next_open_row(col)
            self.board[row, col] = self.current_player

            # Check for a win after placing the token
            if self.check_win(self.current_player):
                print(f"Player {self.current_player} wins!")
                self.game_over = True

            # Switch players
            self.current_player = 3 - self.current_player

        except ValueError as e:
            print(e)

    def check_win(self, player):
        """
        Check whether the current player has won by forming four consecutive tokens vertically,
        horizontally, or diagonally.
        """
        board = self.board

        # Check horizontal locations for a win
        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT - 3):
                if (board[r][c] == player and board[r][c+1] == player and
                    board[r][c+2] == player and board[r][c+3] == player):
                    return True

        # Check vertical locations for a win
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == player and board[r+1][c] == player and
                    board[r+2][c] == player and board[r+3][c] == player):
                    return True

        # Check positively sloped diagonals
        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                if (board[r][c] == player and board[r+1][c+1] == player and
                    board[r+2][c+2] == player and board[r+3][c+3] == player):
                    return True

        # Check negatively sloped diagonals
        for r in range(3, ROW_COUNT):
            for c in range(COLUMN_COUNT - 3):
                if (board[r][c] == player and board[r-1][c+1] == player and
                    board[r-2][c+2] == player and board[r-3][c+3] == player):
                    return True

        return False  # No winning condition found

    def _get_next_open_row(self, col):
        """
        Find the next open row in the specified column.
        """
        for r in range(ROW_COUNT - 1, -1, -1):  # Start from the bottom row and go upwards
            if self.board[r][col] == 0:
                return r
        raise ValueError("Column is full!")  # This should not be reached if properly checked

    def is_board_full(self):
        """Check if the board is full."""
        return np.all(self.board != 0)

    def get_valid_actions(self):
        """Return a list of columns that are not full."""
        return [c for c in range(COLUMN_COUNT) if self.board[0][c] == 0]

    def render(self):
        if self.render_mode:
            for c in range(COLUMN_COUNT):
                for r in range(ROW_COUNT):
                    # Draw the blue board
                    pygame.draw.rect(self.screen, BLUE, (c * SQUARE_SIZE, r * SQUARE_SIZE + SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                    # Draw the empty or filled circles for the tokens
                    if self.board[r][c] == 0:
                        pygame.draw.circle(self.screen, BLACK, (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE / 2)), RADIUS)
                    elif self.board[r][c] == 1:
                        pygame.draw.circle(self.screen, RED, (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE / 2)), RADIUS)
                    else:
                        pygame.draw.circle(self.screen, YELLOW, (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE / 2)), RADIUS)

            # Update the display
            pygame.display.update()

    def draw_board(self):
        # Draw the Connect 4 board in Pygame (this will be called during initialization)
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                pygame.draw.rect(self.screen, BLUE, (c * SQUARE_SIZE, r * SQUARE_SIZE + SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                pygame.draw.circle(self.screen, BLACK, (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE / 2)), RADIUS)
        
        # Refresh display to show the board
        pygame.display.update()

if __name__ == "__main__":
    game = Connect4Game()
    game.render()

    game_over = False  # Initialize game_over flag

    while not game_over:
        # Wait for player input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Get the column where the player clicked
                col = event.pos[0] // SQUARE_SIZE

                # Drop the token and render the updated board
                game.drop_token(col)
                game.render()

                # If the game is over after the move, exit loop
                if game.game_over:
                    game_over = True
