import socket
import sock_config
import pygame
import xbox360_controller
import time

# sock = socket.create_connection((sock_config.SERVER_ADDRESS, sock_config.SERVER_PORT))
# sock.sendall("PING".encode('utf-8'))
# sock.close()

pygame.init()
# Initialize the joysticks

def right(value):
    center = 110
    return str(center + 30 * value)

def run(value):
    max = 60
    return str(max * (-value))


my_controller = xbox360_controller.Controller(0)

with socket.create_connection((sock_config.SERVER_ADDRESS, sock_config.SERVER_PORT)) as sock:
    done = False
    while not done:
        # Event processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        left_x, left_y = my_controller.get_left_stick()
        triggers = my_controller.get_triggers()

        wheel_value = "W " + right(left_x) + "\r\n"
        print("Руль: ", wheel_value)
        sock.sendall(wheel_value.encode('utf-8'))

        running = "E " + run(triggers) + "\r\n"
        print("Газ: ", running)
        sock.sendall(running.encode('utf-8'))
        time.sleep(0.1)