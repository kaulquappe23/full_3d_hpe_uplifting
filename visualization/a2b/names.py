# -*- coding: utf-8 -*-
"""
Created on 17.07.24

@author: Katja

"""
import numpy as np

anthro_names = ['height length', 'shoulder width length', 'torso height from back length', 'torso height from front length', 'head length', 'midline neck length', 'lateral neck length', 'left hand length', 'right hand length', 'left arm length', 'right arm length', 'left forearm length', 'right forearm length', 'left thigh length', 'right thigh length', 'left calf length', 'right calf length', 'left footwidth length', 'right footwidth length', 'left heel to ball length', 'right heel to ball length', 'left heel to toe length', 'right heel to toe length', 'waist circumference', 'chest circumference', 'hip circumference', 'head circumference', 'neck circumference', 'left arm circumference', 'right arm circumference', 'left forearm circumference', 'right forearm circumference', 'left thigh circumference', 'right thigh circumference', 'left calf circumference', 'right calf circumference']
anthro_names_no_leftright = ['height length', 'shoulder width length', 'torso height from back length', 'torso height from front length', 'head length', 'midline neck length', 'lateral neck length', 'hand length', 'arm length', 'forearm length', 'thigh length', 'calf length', 'footwidth length', 'heel to ball length', 'heel to toe length', 'waist circumference', 'chest circumference', 'hip circumference', 'head circumference', 'neck circumference', 'arm circumference', 'forearm circumference', 'thigh circumference', 'calf circumference']

# maps indices with directions to indices without any directions (left/right)
dirless_mapping = np.array([anthro_names_no_leftright.index(i) if i in anthro_names_no_leftright else
                   anthro_names_no_leftright.index(' '.join(i.split(' ')[1:])) for index, i in enumerate(anthro_names)])
# preimage of dirless_mapping
dirless_unmapping = [np.where(dirless_mapping == i)[0] for i in range(len(anthro_names_no_leftright))]