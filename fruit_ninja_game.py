import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hands (Allowing 2 Hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Game Variables
WIDTH, HEIGHT = 800, 600
fruits = []  # List to store fruits and bombs
score = 0
lives = 3
game_over = False

# Define Colors
WHITE = (255, 255, 255)
RED = (0, 0, 255)   # Apple
GREEN = (0, 255, 0) # Lime
BLUE = (255, 0, 0)  # Blueberry
BLACK = (0, 0, 0)   # Bomb

# OpenCV Window
cap = cv2.VideoCapture(0)

# Fruit & Bomb Class
class Fruit:
    def __init__(self, is_bomb=False):
        self.x = random.randint(100, WIDTH - 100)
        self.y = HEIGHT
        self.speed = random.uniform(7, 12)  # Faster speed
        self.radius = 40
        self.is_bomb = is_bomb  # True if this is a bomb
        self.color = BLACK if is_bomb else random.choice([RED, GREEN, BLUE])
        self.spawn_time = time.time()

    def move(self):
        self.y -= self.speed  # Move upward

    def draw(self, frame):
        cv2.circle(frame, (self.x, int(self.y)), self.radius, self.color, -1)

    def is_sliced(self, hands_positions):
        for hand_x, hand_y in hands_positions:
            if np.sqrt((self.x - hand_x) ** 2 + (self.y - hand_y) ** 2) < self.radius:
                return True
        return False

# Function to detect hands and return positions of index fingers
def detect_hand_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    hand_positions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH)
            index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * HEIGHT)
            hand_positions.append((index_x, index_y))

    return hand_positions

# Main Game Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    hands_positions = detect_hand_landmarks(frame)

    if not game_over:
        # Move and draw fruits
        for fruit in fruits[:]:
            fruit.move()
            fruit.draw(frame)

            # If fruit reaches the top without being sliced, lose a life
            if fruit.y < 0:
                if not fruit.is_bomb:  # Only lose life for missed fruits
                    lives -= 1
                fruits.remove(fruit)

        # Check for slicing using both hands
        if hands_positions:
            for fruit in fruits[:]:
                if fruit.is_sliced(hands_positions):
                    if fruit.is_bomb:
                        lives -= 1  # Lose 1 life when slicing a bomb
                    else:
                        score += 1
                    fruits.remove(fruit)

        # Game Over Condition (when lives reach 0)
        if lives <= 0:
            game_over = True

        # Spawn new fruits and bombs randomly
        if random.random() < 0.03:  # Higher probability of fruit spawn
            if random.random() < 0.2:  # 20% chance to spawn a bomb
                fruits.append(Fruit(is_bomb=True))
            else:
                fruits.append(Fruit())

        # Draw Score & Lives
        cv2.putText(frame, f"Score: {score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
        cv2.putText(frame, f"Lives: {lives}", (WIDTH - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

    else:
        # Display Game Over
        cv2.putText(frame, "GAME OVER!", (WIDTH // 3, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 3)
        cv2.putText(frame, "Press 'R' to Restart", (WIDTH // 3 - 50, HEIGHT // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

    # Show the Frame
    cv2.imshow("Fruit Ninja Clone", frame)

    # Key Events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('r') and game_over:  # Restart
        score = 0
        lives = 3
        game_over = False
        fruits.clear()

# Cleanup
cap.release()
cv2.destroyAllWindows()
