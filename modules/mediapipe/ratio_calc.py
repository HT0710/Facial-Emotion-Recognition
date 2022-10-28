""" Compute the Ratio of face points

Used to calculate:
    - d1: Face width
    - d2: Mouth width
    - d3: Mouth openness
    - d4: Eye width *
    - d5: Eye openness *
    - d6: Eyebrow length *
    - d7: Distance between two eyebrow
    - d8: Distance between eye and eyebrow *
    - d9: Nose length
    - d10: Distance between nose and upper lip
    - d11: Distance between eyebrow head and mouth *
    - d12: Distance between eye
* : Average of both sides

Ratio result:
    - a1: d1 / d2
    - a2: d2 / d3
    - a3: d4 / d5
    - a4: d6 / d7
    - a5: d6 / d8
    - a6: d9 / d10
    - a7: d11 / d9
    - a8: d12 / d2
"""

import math


class Distance:
    """Compute the distance between two point"""
    def __init__(self, obj):
        self.__obj = obj

    def result(self, a, b):
        return math.dist((self.__obj[a].x, self.__obj[a].y), (self.__obj[b].x, self.__obj[b].y))


def div(a: float, b: float):
    """Divide a by b - ZeroDivisionError Exception"""
    try:
        return a / b
    except:
        return 0.0


class RatioCalculator:
    def __init__(self, face: list) -> None:
        dist = Distance(face).result
        self.__d1 = dist(345, 116)
        self.__d2 = dist(57, 306)
        self.__d3 = dist(11, 17)
        self.__d4 = (dist(130, 243) + dist(463, 359)) / 2
        self.__d5 = (dist(145, 159) + dist(374, 386)) / 2
        self.__d6 = (dist(55, 70) + dist(285, 300)) / 2
        self.__d7 = dist(55, 285)
        self.__d8 = (dist(55, 243) + dist(285, 359)) / 2
        self.__d9 = dist(1, 8)
        self.__d10 = dist(1, 11)
        self.__d11 = (dist(55, 57) + dist(285, 306)) / 2
        self.__d12 = dist(130, 359)

    def result(self) -> list:
        a1 = div(self.__d1, self.__d2)
        a2 = div(self.__d2, self.__d3)
        a3 = div(self.__d4, self.__d5)
        a4 = div(self.__d6, self.__d7)
        a5 = div(self.__d6, self.__d8)
        a6 = div(self.__d9, self.__d10)
        a7 = div(self.__d11, self.__d9)
        a8 = div(self.__d12, self.__d2)
        result = [a1, a2, a3, a4, a5, a6, a7, a8]

        return [round(num, 5) for num in result]
