import pyControl.hardware as _h


class Breakout_1_2(_h.Mainboard):
    def __init__(self):
        # pins for limit switch
        self.limit_1_bot = 'PC9'
        self.limit_2_bot = 'PB9'
        self.limit_3_bot = 'PA5'
        self.limit_4_bot = 'PA7'
        self.limit_5_bot = 'PC7'
        self.limit_6_bot = 'PA8'

        # port pin for stepper motor
        self.port_1 = _h.Port(DIO_A='PC11', DIO_B='PC10', POW_A=None, POW_B=None)
        self.port_2 = _h.Port(DIO_A='PA1', DIO_B='PH1', POW_A=None, POW_B=None)
        self.port_3 = _h.Port(DIO_A='PC1', DIO_B='PC2', POW_A=None, POW_B=None)
        self.port_4 = _h.Port(DIO_A='PG2', DIO_B='PD5', POW_A=None, POW_B=None)
        self.port_5 = _h.Port(DIO_A='PE4', DIO_B='PE3', POW_A=None, POW_B=None)
        self.port_6 = _h.Port(DIO_A='PF8', DIO_B='PF0', POW_A=None, POW_B=None)

        self.en1 = 'PD2'
        self.en2 = 'PA4'
        self.en3 = 'PC0'
        self.en4 = 'PG3'
        self.en5 = 'PE5'
        self.en6 = 'PF9'

        self.TTL1 = 'PC8'
        self.TTL2 = 'PC5'
        self.TTL3 = 'PA12'
        self.TTL4 = 'PB12'
        self.TTL5 = 'PB2'
        self.TTL6 = 'PB15'
        self.TTL7 = 'PB13'
        self.TTL8 = 'PC4'


class Devboard_1_2(Breakout_1_2):
    def __init__(self):
        self.set_pull_updown({'down': ['PC9','PB9', 'PA5', 'PA7', 'PA8', 'PC8', 'PC5', 'PA12',
                                       'PB2', 'PB15', 'PB13', 'PC4']})
        super().__init__()
