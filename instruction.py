import pygame
import time
import pickle

# Initialize Pygame
pygame.init()

# Screen dimensions and settings
width, height = 1000, 1000
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Text Display Timer")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)  # Red color

# Font settings
font_size = 100
font = pygame.font.Font(None, font_size)

# Function to display text and handle events
def display_text(text, color):
    screen.fill(white)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(width//2, height//2))
    screen.blit(text_surface, text_rect)
    pygame.display.update()

    # Event handling to keep the window responsive
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

# Function to display countdown text
def countdown_text(base_text, duration):
    start_time = time.time()
    for remaining in range(duration, 0, -1):
        if base_text == "Grasp":
            display_text(f"{base_text} ({remaining})", red)  # Display Release in red
        else:
            display_text(f"{base_text} ({remaining})", black)  # Other texts in black
        print(f"{base_text} ({remaining}):", time.time() - start_time)
        time.sleep(1)  # Delay for 1 second, updating every second
        
def save_to_pickle(data, base_filename):
    import datetime
    import os
    spath = os.getcwd() + '\\data_instruction\\'
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime('%Y%m%d%H%M%S')  # Format the current time as a string
    filename = f'{base_filename}_{timestamp}.pkl'
    with open(spath + filename, 'wb') as f:
        pickle.dump(data, f)

# Main function to run the display sequence
def main():
    start_time = time.time()
    display_text("Ready", black)
    print("Ready:", time.time() - start_time)
    pygame.time.delay(10000)  # Delay for 10 seconds
    
    mssave = []
    for i in range(100):  # Repeat the sequence 100 times
        mssave.append([i, "Release", time.time()])
        countdown_text("Release", 5)  # Display Release with countdown
        
        display_text("+", black)
        print("Plus:", time.time() - start_time)
        pygame.time.delay(1000)  # Delay for 1 second
        
        countdown_text("Grasp", 5)  # Display Grasp with countdown
        mssave.append([i, "Grasp", time.time()])
        
        display_text("+", black)
        print("Plus:", time.time() - start_time)
        pygame.time.delay(1000)  # Delay for 1 second
        
        save_to_pickle(mssave, 'grasp')
        mssave = []

# Run the main function
if __name__ == "__main__":
    try:
        main()
    finally:
        pygame.quit()
