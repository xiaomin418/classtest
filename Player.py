"""
类的继承,玩家Player分为机器玩家和人类玩家
"""
class Player:
    """
    Class to represent a player
    """
    def __init__(self,name='',score=0):
        #如果命名为__,连续两个下划线，则为私有属性；单个下划线表示公共属性
        self._name=name
        self._score=score

    def __str__(self):
        return "name='%s',score=%d"%(self._name,self._score)

    def __repr__(self):
        return 'Player(%s)'%str(self)

    def incr_score(self):
        self._score=self._score+1

    def get_name(self):
        return self._name

class Human(Player):
    """
    Huamn从Player继承，重写__repr__()函数
    """
    #pass 表示什么都不做
    def __repr__(self):
        return 'Human(%s)'%str(self)

class Computer(Player):
    def __repr__(self):
        return 'Computer(%s)'%str(self)

if __name__=="__main__":
    p =Player()
    p.name="xiaomin"
    s=str(p)
    print(p)