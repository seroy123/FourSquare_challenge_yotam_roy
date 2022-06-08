import numpy as np


class ExtractFeatures:

    def __init__(self, data):
        self.data = data

    def latitude(self):
        latitude = np.array(self["latitude"])
        return latitude

    def longitude(self):
        longitude = np.array(self["longitude"])
        return longitude

    def country(self):
        country = np.array(self["country"])
        return country

    def name(self):
        name = np.array(self['name'])
        return name
