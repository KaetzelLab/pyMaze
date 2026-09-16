from devices import *
import pyControl.hardware as _h
import pyb

board = Breakout_1_2()  # Instantiate breakout board.
##LimitSwitch

home_lmt_bot  = _h.Digital_input(pin='PD6', falling_event='homeDoor_up', rising_event='homeDoor_down')
left_lmt_bot  = _h.Digital_input(pin='PG3', falling_event='leftDoor_up', rising_event='leftDoor_down')
Right_lmt_bot = _h.Digital_input(pin='PA7', falling_event='rightDoor_up', rising_event='rightDoor_down')


#### Door stepper motor
home_door = Stepper_motor(direction_pin= 'PG9', step_pin='PG13')
left_door = Stepper_motor(direction_pin= 'PE6', step_pin='PE1')
right_door = Stepper_motor(direction_pin= 'PG1', step_pin='PD0')


#### enabling pins Door stepper motor
HomeDoor_enable = _h.Digital_output(pin='PD9')
LeftDoor_enable = _h.Digital_output(pin='PF8')
RightDoor_enable = _h.Digital_output(pin='PG0')



