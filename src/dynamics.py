from collections import Counter

import numpy as np
from joblib import Parallel, delayed


class Struct:
    array = None

    def __init__(self, array_str: str):

        array = [s for s in array_str]
        self.check_consistency(array)
        self.internal = min(["".join(np.roll(array, i)) for i in range(len(array))])
        self.shape = len([x for x in array if x != "2"])

    def __eq__(self, other):
        return self.internal == other.internal

    def __gt__(self, other):
        return self.internal > other.internal

    def __lt__(self, other):
        return self.internal < other.internal

    def __repr__(self):
        return "{}".format(self.internal)

    def __hash__(self):
        return hash(self.internal)

    def remove_one(self):
        assert "2" not in self.internal, "already removed"

        index = np.random.choice(np.arange(6))
        return Struct(self.internal[:index] + "2" + self.internal[index + 1:]), Struct(self.internal[index])

    def check_consistency(self, array):

        if len(array) != 6 and len(array) != 1:
            raise Exception("Length should be 6 or 1")

        counter = Counter(array)
        for key, val in counter.items():
            if key not in ["0", "1", "2"]:
                raise Exception("Corrupted initialization")
            if key == "2" and val > 1:
                raise Exception("There can only be one whole")


class Sixer(Struct):

    def __init__(self, array):
        super().__init__(array)
        assert self.shape == 6, "String should contain 6 elements"

    @classmethod
    def hom(cls, color: str):
        return cls(color * 6)


class Oner(Struct):

    def __init__(self, array):
        super().__init__(array)
        assert self.shape == 1, "String should contain 1 elements"

    @classmethod
    def hom(cls, color: str):
        return cls(color)


class Fiver(Struct):

    def __init__(self, array):
        super().__init__(array)
        assert self.shape == 5, "String should contain 5 elements"

    def fill(self, oner: Oner) -> Sixer:
        return Sixer(self.internal.replace("2", oner.internal))

    @classmethod
    def hom(cls, color: str):
        return cls(color * 5 + "2")


class Population:
    oners: list[Oner]
    fivers: list[Fiver]
    sixers: list[Sixer]

    def __init__(self, oners: list[Oner], fivers: list[Fiver], sixers: list[Sixer]):
        self.oners = oners
        self.fivers = fivers
        self.sixers = sixers

    def __repr__(self):
        rep = str(Counter(self.oners)) + "\n"
        rep += str(Counter(self.fivers)) + "\n"
        rep += str(Counter(self.sixers))
        return rep

    def describe(self):
        return Counter(self.oners), Counter(self.fivers), Counter(self.sixers)

    @classmethod
    def hom(cls, n: int):
        sixers = [Sixer.hom("1") for i in range(n)] + [Sixer.hom("0") for i in range(n)]
        return cls([], [], sixers)

    @classmethod
    def initialize(cls, n: int, k1: float, k2: float):

        n_fivers = n_oners = int(np.floor(- k1 + np.sqrt(k1 ** 2 + 2 / 3 * k1 * k2 * n) / (2 * k2)))
        n_sixers = int(np.floor(n / 6 - n_fivers))

        sixers = [Sixer.hom("1") for _ in range(n_sixers)] + [Sixer.hom("0") for _ in range(n_sixers)]
        fivers = [Fiver.hom("1") for _ in range(n_fivers)] + [Fiver.hom("0") for _ in range(n_fivers)]
        oners = [Oner("1") for _ in range(n_oners)] + [Oner("0") for _ in range(n_oners)]

        return Population(oners, fivers, sixers)

    @classmethod
    def get_all_keys(cls):
        oners = set([Oner("0"), Oner("1")])
        sixers = set([Sixer(format(i, '#08b')[2:]) for i in range(64)])
        fivers = set([Fiver(format(i, '#07b')[2:] + "2") for i in range(32)])
        return oners, sixers, fivers

def sixers_next_step(sixers_lst: list[Sixer], k1: float) -> tuple[list[Oner], list[Fiver], list[Sixer]]:
    oners = []
    fivers = []
    sixers = []

    def proccess_sixer(sixer) -> tuple[list[Oner], list[Fiver], list[Sixer]]:
        o_ = []
        f_ = []
        s_ = []
        if np.random.random() < k1:
            fiver, oner = sixer.remove_one()
            f_.append(fiver)
            o_.append(oner)
        else:
            s_.append(sixer)
        return o_, f_, s_

    parallel = Parallel(n_jobs=-1)
    res = parallel(delayed(proccess_sixer)(sixer) for sixer in sixers_lst)
    for o_, f_, s_ in res:
        oners.extend(o_)
        fivers.extend(f_)
        sixers.extend(s_)

    return oners, fivers, sixers


def oner_fiver_next_step(oners_lst: list[Oner], fivers_lst: list[Fiver], k2: float):
    oners = []
    fivers = []
    sixers = []

    def proccess_oner_fiver(oner: Oner, fiver: Fiver) -> tuple[list[Oner], list[Fiver], list[Sixer]]:
        o_ = []
        f_ = []
        s_ = []
        if np.random.random() < k2:
            s_.append(fiver.fill(oner))
        else:
            o_.append(oner)
            f_.append(fiver)
        return o_, f_, s_

    parallel = Parallel(n_jobs=-1)
    res = parallel(delayed(proccess_oner_fiver)(oner, fiver) for oner, fiver in zip(oners_lst, fivers_lst))
    for o_, f_, s_ in res:
        oners.extend(o_)
        fivers.extend(f_)
        sixers.extend(s_)

    return oners, fivers, sixers


if __name__ == "__main__":
    print(Sixer.hom("1"))
    print(Fiver.hom("0"))
    print(Fiver.hom("0").fill(Oner.hom("1")))
    print(Struct("120111"))
    print(sixers_next_step([Sixer.hom("1") for _ in range(10)], 0.5))
    print(Population.hom(10))
    print(Population.get_all_keys())
