import numpy as np


class ExtractFeatures:

    def __init__(self, data):
        self.data = data

    def latitude(self):
        latitude = np.array(self["latitude"].fillna(0, inplace=False))
        return latitude

    def longitude(self):
        longitude = np.array(self["longitude"].fillna(0, inplace=False))
        return longitude

    def country(self):
        country = np.array(self["country"].fillna('', inplace=False))
        return country

    def name(self):
        name = np.array(self['name'])
        return name

    def address(self):
        address = np.array(self['address'].fillna('', inplace=False))
        return address

    def state(self):
        state = np.array(self['state'].fillna('', inplace=False))
        return state

    def categories(self):
        categories = np.array(self['categories'].fillna('', inplace=False))
        return categories
