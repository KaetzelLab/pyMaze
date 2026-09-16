import pyControl.hardware as _h


class Breakout_1_2(_h.Mainboard):
    def __init__(self):
        # pins for limit switch
        #self.limit_1_top = 'B9' #A
        #self.limit_2_top = 'B12'
        #self.limit_3_top = 'C6'

        self.limit_1_bot = 'B8' #B
        self.limit_2_bot = 'B11'
        self.limit_3_bot = 'C7'
   

        # port pins for stepper motor
        #self.port_1 = _h.Port(DIO_A='A15', DIO_B='B7', POW_A=None, POW_B=None) #A = Dir
        #self.port_2 = _h.Port(DIO_A='F7', DIO_B='A13', POW_A=None, POW_B=None)
        #self.port_3 = _h.Port(DIO_A='C11', DIO_B='D2', POW_A=None, POW_B=None)


        self.port_1 = _h.Port(DIO_A='B7', DIO_B='A15', POW_A=None, POW_B=None) #A = Dir
        self.port_2 = _h.Port(DIO_A='A13', DIO_B='F7', POW_A=None, POW_B=None)
        self.port_3 = _h.Port(DIO_A='B0', DIO_B='C1', POW_A=None, POW_B=None)


        self.en1 = 'A14'
        self.en2 = 'F6'
        self.en3 = 'C0'

class Devboard_1_2(Breakout_1_2):
    def __init__(self):
        self.set_pull_updown({'down': ['B9', 'B12', 'C6', 'B8', 'B11', 'C7']})
        super().__init__()

