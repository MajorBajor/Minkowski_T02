import matplotlib.pyplot as pyplot
import numpy as np

class Gate:

    def __init__(self, post_one, post_two):
        self.dimension = 2
        self.post_one = post_one
        self.post_two = post_two
        self.center = (post_one + post_two) / 2
        self.radius = np.dot(post_one, post_two)
        # self.normal = none
        # self.circle = none

    """"
    def __init__(self, circle):
        self.dimension = 3
        self.center = center
        self.radius = radius
        self.normal = normal
        self.post_one = null
        self.post_two = null
        self.circle = circle

    def __init__(self, dimension, center, radius, normal):
        self.dimension = dimension
        self.center = center
        self.radius = radius
        self.normal = normal
        self.post_one = null
        self.post_two = null
        self.circle = null
    """

    def get_center(self):
        return self.center

    def get_radius(self):
        return self.radius

    def get_perimeter(self):
        perimeter = 2
        return perimeter

    def plot(self, plt):
        plt.scatter(self.post_one[0], self.post_one[1], label='Scatter Plot 1', color='b')
        plt.scatter(self.post_two[0], self.post_two[1], label='Scatter Plot 1', color='b')
        plt.scatter(self.center[0], self.center[1], label='Scatter Plot 1', color='r')

# end class Gate:


class Border:

    def __init__(self, start, end):
        self.dimension = 2
        self.start = start
        self.end = end
        self.resolution = (end[0] - start[0]) / 100
        self.amplitude = 0.01
        self.frequency = 5
        # self.center = (post_one + post_two) / 2
        # self.radius = np.dot(post_one, post_two)
        # self.normal = none
        # self.circle = none

    def get_perimeter(self):
        perimeter = 3
        return perimeter

    # def get_radius(self):
    #    return self.radius

    def plot(self, plt):
        # x = np.array([self.start[0], self.end[0]])
        # y = np.array([self.start[1], self.end[1]])
        x = np.arange(self.start[0], self.end[0], self.resolution)
        ysin = self.amplitude * np.sin( 2 *  self.frequency * np.pi * (x - self.start[0]) / (self.end[0] - self.start[0]))
        ylin = self.start[1] + (self.end[1] - self.start[1]) * (x - self.start[0]) / (self.end[0] - self.start[0])
        y = ysin + ylin
        plt.plot(x, y)

# end class Gate:

class Channel:

    def __init__(self, gate_one, gate_two):
        self.dimension = 2
        self.gate_one = gate_one
        self.gate_two = gate_two
        self.border_one = Border(gate_one.post_one, gate_two.post_one)
        self.border_two = Border(gate_one.post_two, gate_two.post_two)
        # self.normal = none
        # self.circle = none

    def get_area(self):
        area = 2*4
        return area

    def get_perimeter(self):
        perimeter = border_one.get_perimeter() + border_two.get_perimeter + gate_one.perimeter() + gate_two.perimeter()
        return perimeter

    def get_euler_char(self):
        euler_char = 2
        return euler_char

    def plot(self, plt):
        self.gate_one.plot(plt)
        self.gate_two.plot(plt)
        self.border_one.plot(plt)
        self.border_two.plot(plt)
        # plt.fill_between(self.border_one, self.border_two)

# end class Gate:


A0 = np.array([0.0, 0.5])
B0 = np.array([0.0, 1.5])
A1 = np.array([1.0, 0.5])
B1 = np.array([1.0, 1.5])
A2 = np.array([2.0, 0.1])
B2 = np.array([2.0, 1.9])
A3 = np.array([3.0, 0.0])
B3 = np.array([3.0, 2.0])
A4 = np.array([4.0, 0.1])
B4 = np.array([4.0, 1.9])
A5 = np.array([5.0, 0.5])
B5 = np.array([5.0, 1.0])
C5 = np.array([5.0, 1.0])
D5 = np.array([5.0, 1.5])
A6 = np.array([6.0, 0.2])
B6 = np.array([6.0, 0.5])
C6 = np.array([6.0, 1.5])
D6 = np.array([6.0, 1.8])

G0 = Gate(A0, B0)
G1 = Gate(A1, B1)
G2 = Gate(A2, B2)
G3 = Gate(A3, B3)
G4 = Gate(A4, B4)
GAD5 = Gate(A5, D5)
GAB5 = Gate(A5, B5)
GCD5 = Gate(C5, D5)
GAB6 = Gate(A6, B6)
GCD6 = Gate(C6, D6)

# gate_one.plot(pyplot)
# gate_two.plot(pyplot)

"""
border_one = Border(gate_one.post_one, gate_two.post_one)
border_two = Border(gate_one.post_two, gate_two.post_two)

border_one.plot(pyplot)
border_two.plot(pyplot)
"""

chanelG0G1 = Channel(G0, G1)
chanelG1G2 = Channel(G1, G2)
chanelG2G3 = Channel(G2, G3)
chanelG3G4 = Channel(G3, G4)
chanelGAD5 = Channel(G4, GAD5)
chanelGAB5GAB6 = Channel(GAB5, GAB6)
chanelGCD5GCD6 = Channel(GCD5, GCD6)

chanelG0G1.plot(pyplot)
chanelG1G2.plot(pyplot)
chanelG2G3.plot(pyplot)
chanelG3G4.plot(pyplot)
chanelGAD5.plot(pyplot)
chanelGAB5GAB6.plot(pyplot)
chanelGCD5GCD6.plot(pyplot)

area = 200
perimeter = 100

ax = pyplot.subplot()
offset = 72
bbox = dict(boxstyle="round", fc="0.8")
ax.annotate('area = %.1f, perimeter = %.1f)'%(area, perimeter),
            A0, xytext = (offset, offset), textcoords='offset points', bbox=bbox)

pyplot.show()

"""
#if __name__ == '__main__':
x = np.arange(0.0, 2, 0.01)
y1 = np.sin(2 * np.pi * x)
y2 = 0.8 * np.sin(4 * np.pi * x)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 6))

print("Example 1")

ax1.fill_between(x, y1)
ax1.set_title('fill between y1 and 0')

ax2.fill_between(x, y1, 1)
ax2.set_title('fill between y1 and 1')

ax3.fill_between(x, y1, y2)
ax3.set_title('fill between y1 and y2')
ax3.set_xlabel('x')
fig.tight_layout()

print("Example 2")

N = 21
x = np.linspace(0, 10, 11)
y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1,  9.9, 13.9, 15.1, 12.5]

# fit a linear curve an estimate its y-values and their error.
a, b = np.polyfit(x, y, deg=1)
y_est = a * x + b
y_err = x.std() * np.sqrt(1/len(x) +
                          (x - x.mean())**2 / np.sum((x - x.mean())**2))

fig, ax = plt.subplots()
ax.plot(x, y_est, '-')
ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
ax.plot(x, y, 'o', color='tab:brown')
plt.show()
"""
"""
my_car = Car()
print("I'm a car!")
while True:
    action = input("What should I do? [A]ccelerate, [B]rake, "
             "show [O]dometer, or show average [S]peed?").upper()
    if action not in "ABOS" or len(action) != 1:
        print("I don't know how to do that")
        continue
    if action == 'A':
        my_car.accelerate()
    elif action == 'B':
        my_car.brake()
    elif action == 'O':
        print("The car has driven {} kilometers".format(my_car.odometer))
    elif action == 'S':
        print("The car's average speed was {} kph".format(my_car.average_speed()))
    my_car.step()
    my_car.say_state()
"""