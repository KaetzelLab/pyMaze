import pyControl.hardware as _h


class Breakout_1_2(_h.Mainboard):
    def __init__(self):
        # pins for limit switch
        self.limit_1_top = 'PA2'
        self.limit_1_bot = 'PA3'
        self.limit_2_top = 'PA0'
        self.limit_2_bot = 'PA1'
        self.limit_3_top = 'PC2'
        self.limit_3_bot = 'PC3'
        self.limit_4_top = 'PC0'
        self.limit_4_bot = 'PC1'
        self.limit_5_top = 'PE6'
        self.limit_5_bot = 'PC13'
        self.limit_6_top = 'PE4'
        self.limit_6_bot = 'PE5'
        self.limit_7_top = 'PE2'
        self.limit_7_bot = 'PE3'
        # port pin for stepper motor
        self.port_1 = _h.Port(DIO_A='PB15', DIO_B='PD8', POW_A=None, POW_B=None)
        self.port_2 = _h.Port(DIO_A='PD9', DIO_B='PD10', POW_A=None, POW_B=None)
        self.port_3 = _h.Port(DIO_A='PD11', DIO_B='PD12', POW_A=None, POW_B=None)
        self.port_4 = _h.Port(DIO_A='PD13', DIO_B='PD14', POW_A=None, POW_B=None)
        self.port_5 = _h.Port(DIO_A='PD15', DIO_B='PC6', POW_A=None, POW_B=None)
        self.port_6 = _h.Port(DIO_A='PC7', DIO_B='PC8', POW_A=None, POW_B=None)
        self.port_7 = _h.Port(DIO_A='PD4', DIO_B='PD5', POW_A=None, POW_B=None)
        self.port_8 = _h.Port(DIO_A='PD6', DIO_B='PD7', POW_A=None, POW_B=None)
        self.port_9 = _h.Port(DIO_A='PB3', DIO_B='PB5', POW_A=None, POW_B=None)

        # reward nose pokes
        self.port_10 = _h.Port(DIO_A='PE12', DIO_B=None, POW_A='PB14', POW_B='PB12')
        self.port_11 = _h.Port(DIO_A='PB13', DIO_B=None, POW_A='PE14', POW_B='PB10')

        # # pins for limit switch
        # self.limit_1_top = 'PA2'
        # self.limit_1_bot = 'PA3'
        # self.limit_2_top = 'PA0'
        # self.limit_2_bot = 'PA1'
        # self.limit_3_top = 'PC2'
        # self.limit_3_bot = 'PC3'
        # self.limit_4_top = 'PC0'
        # self.limit_4_bot = 'PC1'
        # self.limit_5_top = 'PE6'
        # self.limit_5_bot = 'PC13'
        # self.limit_6_top = 'PE4'
        # self.limit_6_bot = 'PE5'
        # self.limit_7_top = 'PE2'
        # self.limit_7_bot = 'PE3'
        # self.TTL_1 = 'PD12'
        # self.TTL_2 = 'X11'
        # self.TTL_3 = 'X5'
        # self.TTL_4 = 'X6'
        # self.TTL_5 = 'X6'
        # self.TTL_6 = 'X6'
        # self.TTL_7 = 'X6'


class Devboard_1_2(Breakout_1_2):
    def __init__(self):
        self.set_pull_updown({'down': ['PA2', 'PA3', 'PA0', 'PC2', 'PA1',
                                       'PC3', 'PC0', 'PC1', 'PE6', 'PC13', 'PE4',
                                       'PE5', 'PE2', 'PE3', 'PE12', 'PB13']})
        super().__init__()
