import pyControl.hardware as _h

class Breakout_1_2(_h.Mainboard):
    def __init__(self):
        # pins for limit switch
        self.limit_1_bot = 'X1'
        self.limit_2_bot = 'X2'
        self.limit_3_bot = 'Y3'
        self.limit_4_bot = 'Y4'
        self.limit_5_bot = 'X9'
        self.limit_6_bot = 'X10'

        # port pin for stepper motor
        self.port_1 = _h.Port(DIO_A='Y7', DIO_B='Y8', POW_A=None, POW_B=None)
        self.port_2 = _h.Port(DIO_A='Y11', DIO_B='Y12', POW_A=None, POW_B=None)
        self.port_3 = _h.Port(DIO_A='X4', DIO_B='X7', POW_A=None, POW_B=None)
        self.port_4 = _h.Port(DIO_A='X18', DIO_B='X19', POW_A=None, POW_B=None)
        self.port_5 = _h.Port(DIO_A='X21', DIO_B='X22', POW_A=None, POW_B=None)
        self.port_6 = _h.Port(DIO_A='X20', DIO_B='Y1', POW_A=None, POW_B=None)

        self.TTL_1 = 'Y9'
        self.TTL_2 = 'Y10'
        self.TTL_3 = 'Y2'
        self.TTL_4 = 'Y5'
        self.TTL_5 = 'Y6'
        self.TTL_6 = 'X3'
        self.TTL_7 = 'X6'
        self.TTL_8 = 'X12'
        self.button = 'X17'

class Devboard_1_2(Breakout_1_2):
    def __init__(self):
        self.set_pull_updown({'down': ['X1', 'X2', 'Y3', 'Y4', 'X9', 'X10',
                                       'Y9', 'Y10', 'Y5', 'Y6', 'Y1', 'X3', 'X6']})
        super().__init__()
