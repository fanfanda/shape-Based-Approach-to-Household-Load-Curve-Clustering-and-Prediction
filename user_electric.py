class user_electric:
    def __init__(self,profiles):
        self.ID=profiles[0]
        self.dataDate=profiles[1]
        self.dataType=profiles[2]
        self.orgNO=profiles[3]
        self.getDate=profiles[4]
        self.dataPointFlag=profiles[5]
        self.dataWholeFlag=profiles[6]
        self.electric_data=profiles[7:]
        self.normalized_electric_data=self.electric_data/sum(self.electric_data)