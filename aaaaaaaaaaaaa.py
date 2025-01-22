class Father():
    def __init__(self):
        self.a = 'aaa'
        self.b = 'bbb'

    def action(self):
        # print('调用父类的方法')
        print(self.a + self.b)


f = Father()
f.action()
