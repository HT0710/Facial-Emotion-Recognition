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
    - d11: Distance between two eyebrow middle point
    - d12: Distance between two irises
* : Average of both sides

Ratio result:
    - a1: Face width / Mouth width
    - a2: Mouth width / Mouth openness
    - a3: Eye width / Eye openness
    - a4: Eyebrow length / Distance between two eyebrow
    - a5: Eyebrow length / Distance between eye and eyebrow
    - a6: Nose length / Distance between nose and upper lip
    - a7: Distance between two eyebrow middle point / Nose length
    - a8: Distance between two irises / Mouth width
"""

import math


class RatioCalculator:
    def __init__(self, face: list) -> None:
        self.__d1 = self.__dis(face[0], face[16])
        self.__d2 = self.__dis(face[48], face[54])
        self.__d3 = self.__dis(face[51], face[57])
        self.__d4 = (self.__dis(face[36], face[39]) + self.__dis(face[42], face[45])) / 2
        self.__d5 = ((self.__dis(face[37], face[41]) + self.__dis(face[38], face[40])) / 2 +
                     (self.__dis(face[43], face[47]) + self.__dis(face[44], face[46])) / 2) / 2
        self.__d6 = (self.__dis(face[17], face[21]) + self.__dis(face[22], face[26])) / 2
        self.__d7 = self.__dis(face[21], face[22])
        self.__d8 = (self.__dis(face[19], face[37]) + self.__dis(face[24], face[44])) / 2
        self.__d9 = self.__dis(face[27], face[33])
        self.__d10 = self.__dis(face[33], face[51])
        self.__d11 = self.__dis(face[18], face[25])
        self.__d12 = self.__dis(face[37], face[44])

    @staticmethod
    def __dis(a: tuple, b: tuple) -> float:
        """Compute the distance between two point"""
        return math.dist(a, b)

    @staticmethod
    def __div(a: float, b: float) -> float:
        """Divide a by b"""
        try:
            return a / b
        except:
            return 0.0

    def result(self) -> list:
        a1 = self.__div(self.__d1, self.__d2)
        a2 = self.__div(self.__d2, self.__d3)
        a3 = self.__div(self.__d4, self.__d5)
        a4 = self.__div(self.__d6, self.__d7)
        a5 = self.__div(self.__d6, self.__d8)
        a6 = self.__div(self.__d9, self.__d10)
        a7 = self.__div(self.__d11, self.__d9)
        a8 = self.__div(self.__d12, self.__d2)
        result = [a1, a2, a3, a4, a5, a6, a7, a8]
        return [round(num, 10) for num in result]


def main():
    # Ratio of 2 image size
    _48x48 = [3.3607, 2.3074, 3.9005, 1.8216, 1.9501, 3.5, 2.2326, 1.6449]
    _1000x1000 = [2.9002, 2.9031, 3.2035, 2.2454, 2.6886, 4.3915, 1.7575, 1.4268]

    # Ratio is different but
    diff = [a / b for a, b in zip(_48x48, _1000x1000)]
    print(diff)  # % different of each Ratio
    print(sum(diff) / 8)  # but Average diff is 0.99

    # => Image size has no significant impact


if __name__ == '__main__':
    main()
