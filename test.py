class P():

    def __init__(self):
        x, d = self.func()
        self.a = x
        self.b = self.a
        self.a[0] = 100
        print(self.a)
        print(self.b)

    def func(self):
        return [1,2,3,4], 5

obj = P()


