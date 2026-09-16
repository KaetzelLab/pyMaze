from devices import *

board = Breakout_1_2()  # Instantiate breakout board.
##LimitSwitch

home_lmt_bot  = Digital_input(pin=board.limit_1_bot, falling_event='homeDoor_up', rising_event='homeDoor_down')
left_lmt_bot  = Digital_input(pin=board.limit_2_bot, falling_event='leftDoor_up', rising_event='leftDoor_down')
Right_lmt_bot = Digital_input(pin=board.limit_3_bot, falling_event='rightDoor_up', rising_event='rightDoor_down')


#### Door stepper motor
home_door = Stepper_motor(port=board.port_1)
left_door = Stepper_motor(port=board.port_2)
right_door = Stepper_motor(port=board.port_3)


#### enabling pins Door stepper motor
HomeDoor_enable = Digital_output(pin=board.en1)
LeftDoor_enable = Digital_output(pin=board.en2)
RightDoor_enable = Digital_output(pin=board.en3)



