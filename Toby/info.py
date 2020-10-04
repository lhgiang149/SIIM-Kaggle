from __future__ import print_function
import numpy as np 

class AllLabel():
    def location(self):
        return ['empty',
                'head/neck',
                'lower extremity',
                'oral/genital',
                'palms/soles',
                'torso',
                'upper extremity']
    def sex(self):
        return ['female',
                'male',
                'Not sure']
    def age(self):
        return [-2147483648,   0,
                10,            15,
                20,            25,
                30,            35,
                40,            45,
                50,            55,
                60,            65,
                70,            75,
                80,            85,
                90]
