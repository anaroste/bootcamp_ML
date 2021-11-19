def is_shape(shape):
    if not isinstance(shape, tuple) and len(shape) != 2:
        return False
    if not isinstance(shape[0], int) and not isinstance(shape[1], int):
        return False
    return True


def check_liste(liste):
    for elt in liste:
        if not isinstance(elt, float):
            return False
    return True


def define_arg(arg):
    tab = []
    if isinstance(arg, list):
        if isinstance(arg[0], list):
            for elt in arg:
                if not check_liste(elt):
                    print("Liste incorrect")
                    exit()
            tab = arg
        elif isinstance(arg[0], float) and check_liste(arg):
            tab = arg
        else:
            print("Liste incorrect")
            exit()
    elif isinstance(arg, tuple):
        if len(arg) != 2 or not isinstance(arg[0], int) \
           or not isinstance(arg[1], int) or arg[0] >= arg[1]:
            print("Tuple incorrect")
            exit()
        for i in range(arg[0], arg[1]):
            tab.append([float(i)])
    elif isinstance(arg, int) and arg >= 0:
        for i in range(arg):
            tab.append([float(i)])
    else:
        print("Type de l'argument non gere ou negatif")
        exit()
    if isinstance(tab[0], list):
        dim = (len(tab), len(tab[0]))
    else:
        dim = (1, len(tab))
    return tab, dim


def init_tab(shape):
    r = []
    for i in range(shape[1]):
        r.append(0)
    tab = []
    for i in range(shape[0]):
        tab.append(r.copy())
    return tab


class Matrix:
    def __init__(self, arg):
        tab, dim = define_arg(arg)

        self.data = tab
        self.shape = dim

    def T(self):
        tab = init_tab((self.shape[1], self.shape[0]))
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                tab[j][i] = self.data[i][j]
        self.data = tab

    def __add__(self, other):
        if isinstance(other, Matrix) and other.shape == self.shape:
            tab = init_tab(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    tab[i][j] = self.data[i][j] + other.data[i][j]
            return Matrix(tab)
        else:
            print("Ce ne sont pas des matrices ou elles n'ont pas la meme dimension")
            return None

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Matrix) and other.shape == self.shape:
            tab = init_tab(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    tab[i][j] = self.data[i][j] - other.data[i][j]
            return Matrix(tab)
        else:
            print("Les matrices n'ont pas la meme dimension")
            return None

    __rsub__ = __sub__

    def __truediv__(self, oth):
        if ((isinstance(oth, int) or isinstance(oth, float)) and oth != 0):
            tab = init_tab(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    tab[i][j] = self.data[i][j] / oth
            return Matrix(tab)
        else:
            if oth == 0:
                raise ValueError("A scalar cannot be divided by a Vector")
            else:
                print("Erreur, division par autre chose qu'un int ou un float")
            return None

    def __rtruediv__(self, other):
        print("Erreur, impossible div scalaraire par vecteur")
        return None

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            tab = init_tab(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    tab[i][j] = self.data[i][j] * other
            return Matrix(tab)
        elif isinstance(other, Matrix) and other.shape == self.shape:
            ret = init_tab(other.shape)
            for i in range(other):
                for j in range(other[i]):
                    ret[i][j] = self.data[i][j] * other.data[i][j]
            return Matrix(ret)
        elif isinstance(other, Vector):
            ret = init_tab(other.shape)
            for i in range(other):
                for j in range(other[i]):
                    ret[i][j] = self.data[i][j] * other.data[i][j]
            return Vector(ret)
        else:
            print("Erreur, mul par autre chose qu'un int ou un float")
            return None

    __rmul__ = __mul__

    def __str__(self):
        return f"Dimension : {self.shape}\n{self.data}"

    def __repr__(self):
        return f"{self.shape}-{self.data}"


def check_vector(arg):
    if not isinstance(arg, list):
        return False
    if len(arg) == 1:
        return True
    for elt in arg:
        print(elt)
        if len(elt) != 1:
            return False
    return True


class Vector(Matrix):
    def __init__(self, arg):
        if not check_vector(arg):
            print('Wrong vector')
            exit()
        super().__init__(arg)

    def dot(self, other):
        if other.shape == self.shape:
            tab = init_tab(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    tab[i][j] = self.values[i][j] * other.values[i][j]
            return Vector(tab)
        else:
            print("Les matrices n'ont pas la meme dimension")
            return None
