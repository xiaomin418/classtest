"""
类的属性和函数编写
"""
class Person:
    """
    Class to represent a person
    """
    def __init__(self,name='',age=0):
        #如果命名为__,连续两个下划线，则为私有属性；单个下划线表示公共属性
        self._name=name
        self._age=age

    def __str__(self):
        return "Person('%s',%d)"%(self._name,self._age)

    def __repr__(self):
        return str(self)

    @property
    def age(self):
        #获取函数返回变量的值，使用@property来指出
        return self._age

    @age.setter
    def age(self,age):
        #给age创建设置函数，用age.setter装饰set_age函数（重命名为age函数）
        if 0<age<=150:
            self._age=age

if __name__=="__main__":
    p =Person()
    p.name="xiaomin"
    s=str(p)
    print(p)