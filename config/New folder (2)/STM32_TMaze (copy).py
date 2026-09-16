from devices import *

board = Breakout_1_2()  # Instantiate breakout board.
##LimitSwitch
home_lmt_bot  = Digital_input(pin=board.limit_1_bot, falling_event='homeDoor_up', rising_event='homeDoor_down')




#### Door stepper motor
home_door = Stepper_motor(port=board.port_1)
left_door = Stepper_motor(port=board.port_2)
right_door = Stepper_motor(port=board.port_3)

###Reward Ports
home_reward = Stepper_motor(port=board.port_4)
left_reward = Stepper_motor(port=board.port_5)
right_reward = Stepper_motor(port=board.port_6)

#### enabling pins Door stepper motor
elevatedDoor_enable = Digital_output(pin=board.en1)
novelDoor_enable = Digital_output(pin=board.en2)
socialNorthDoor_enable = Digital_output(pin=board.en3)
objectDoor_enable = Digital_output(pin=board.en4)
familiarDoor_enable = Digital_output(pin=board.en5)
socialSouthDoor_enable = Digital_output(pin=board.en6)

#### TTL
TTL1 = Digital_output(pin=board.TTL1)
TTL2 = Digital_output(pin=board.TTL2)
TTL3 = Digital_output(pin=board.TTL3)
TTL4 = Digital_output(pin=board.TTL4)
TTL5 = Digital_output(pin=board.TTL5)
TTL6 = Digital_output(pin=board.TTL6)
TTL7 = Digital_output(pin=board.TTL7)
TTL8 = Digital_output(pin=board.TTL8)
