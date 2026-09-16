import pyControl.hardware as _h


class Breakout_1_2(_h.Mainboard):
    def __init__(self):
        # pins for limit switch
        self.limit_1_bot = 'G6'
        self.limit_2_bot = 'G7'
        self.limit_3_bot = 'F15'

        self.limit_1_top = 'G4'
        self.limit_2_top = 'G8'
        self.limit_3_top = 'A3'
   

        # port pins for stepper motor
        self.port_1 = _h.Port(DIO_A='G11', DIO_B='G13', POW_A=None, POW_B=None)
        self.port_2 = _h.Port(DIO_A='G10', DIO_B='G9', POW_A=None, POW_B=None)
        self.port_3 = _h.Port(DIO_A='F0', DIO_B='F1', POW_A=None, POW_B=None)


        self.en1 = 'D9'
        self.en2 = 'G12'
        self.en3 = 'D1'



class Devboard_1_2(Breakout_1_2):
    def __init__(self):
        self.set_pull_updown({'down': ['G6', 'G7', 'F15', 'G4', 'G8', 'A3']})
        super().__init__()

