import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
#np.seterr(divide='ignore',invalid='ignore')

img = cv2.imread("c:/opencv/aa.jpg") #“图片地址”
edge = cv2.Canny(img,100,125)
h,w = edge.shape
Y = edge.argmax(axis=0)
X = np.arange(edge.shape[1])
plt.imshow(edge,cmap='gray')
plt.plot(X,Y,'r')
plt.axis([0, w, h, 0])

# 模拟
def line_circle_intersect(b, cx, cy, r):
    dx = np.absolute(r**2 - (b - cy)**2)**0.5
    x1 = cx - dx
    x2 = cx + dx
    return x1, x2

def line_circle(x, b, cx, cy, r):
    x1, x2 = line_circle_intersect(b, cx, cy, r)
    return np.where(x < x1, b, 
                    np.where(x > x2, b, 
                             cy - np.absolute(r**2 - (cx - x)**2)**0.5))


def loss(p):
    return np.abs(line_circle(X, *p) - Y).sum()

rlist = np.linspace(10, w/2, 10) #半径的等差数列，10个

res = min([optimize.minimize(loss, (h/2, w/2, h/2, r)) for r in rlist], key=lambda res: res.fun)
# (h/2,w/2, h/2, r) 为初始值，for r in rlist，此处为不同半径下的优化结果
# key=lambda res: res.fun 表示这里min的比较对象为函数
# res.x 返回最佳拟合结果

Y2 = line_circle(X, *res.x)
plt.imshow(img)
plt.plot(X, Y2, "b", alpha=0.5)

b, cx, cy, r = res.x
angle = np.pi*0.5 - np.arcsin((cy - b) / r)

x1, x2 = line_circle_intersect(*res.x)

angle2 = np.pi - angle

length = 100
dy = -length*np.tan(angle2)
plt.plot([x2-length, x2+length], 
        [b-dy, b+dy], "g",alpha=0.8)

ax = plt.gca()
ax.axis((0, w, h, 0))
print('接触角：', np.rad2deg(angle))

plt.show()

