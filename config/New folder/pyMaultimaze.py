from devices import *

board = Breakout_1_2()  # Instantiate breakout board.
# Elevated arm
elevated_lmt_bot = Digital_input(pin=board.limit_1_bot, rising_event='elevatedBot_down', falling_event='elevatedBot_up')
# Novel arm A
novel_lmt_bot = Digital_input(pin=board.limit_2_bot, rising_event='novelBot_down', falling_event='novelBot_up')
# # Social arm A
socialA_lmt_bot = Digital_input(pin=board.limit_3_bot, rising_event='socialABot_down', falling_event='socialABot_up')
# Object arm
object_lmt_bot = Digital_input(pin=board.limit_4_bot, rising_event='objectBot_down', falling_event='objectBot_up')
# Familiar arm
familiar_lmt_bot = Digital_input(pin=board.limit_5_bot, rising_event='familiarBot_down', falling_event='familiarBot_up')
# Social arm B
socialB_lmt_bot = Digital_input(pin=board.limit_6_bot, rising_event='socialBBot_down', falling_event='socialBBot_up')

#### Door stepper motor
elevated_door = Stepper_motor(port=board.port_1)
novel_door = Stepper_motor(port=board.port_2)
socialA_door = Stepper_motor(port=board.port_3)
object_door = Stepper_motor(port=board.port_4)
familiar_door = Stepper_motor(port=board.port_5)
socialB_door = Stepper_motor(port=board.port_6)

TTL_1 = 'Y9'
TTL_2 = 'Y10'
TTL_3 = 'Y2'
TTL_4 = 'Y5'
TTL_5 = 'Y6'
TTL_6 = 'X3'
TTL_7 = 'X6'
TTL_8 = 'X12'