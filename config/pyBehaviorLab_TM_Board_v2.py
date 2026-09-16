from devices import *
import pyControl.hardware as _h

home_lmt_bot  = _h.Digital_input(pin='PD5', falling_event='homeDoor_up', rising_event='homeDoor_down')
left_lmt_bot  = _h.Digital_input(pin='PG2', falling_event='leftDoor_up', rising_event='leftDoor_down')
Right_lmt_bot = _h.Digital_input(pin='PA7', falling_event='rightDoor_up', rising_event='rightDoor_down')

home_lmt_top  = _h.Digital_input(pin='PD6', falling_event='homeDoor_top_up', rising_event='homeDoor_top_down')
left_lmt_top  = _h.Digital_input(pin='PG3', falling_event='leftDoor_top_up', rising_event='leftDoor_top_down')
Right_lmt_top = _h.Digital_input(pin='PC8', falling_event='rightDoor_top_up', rising_event='rightDoor_top_down')


#### Door stepper motor

left_door = Stepper_motor(direction_pin= 'PE6', step_pin='PE1')
right_door = Stepper_motor(direction_pin= 'PG1', step_pin='PD0')
home_door = Stepper_motor(direction_pin= 'PF2', step_pin='PF9')


#### enabling pins Door stepper motor
HomeDoor_enable = _h.Digital_output(pin='PD1', inverted = True)
LeftDoor_enable = _h.Digital_output(pin='PF8', inverted = True)
RightDoor_enable = _h.Digital_output(pin='PG0', inverted = True)



