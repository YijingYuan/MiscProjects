#!/usr/bin/env python

# Class
# 1.1 Define class & bonus question

class Rectangular:
    length = 0
    width = 0
    def __init__(self, l, w):
        self.length = l
        self.width = w    
    def area(self):
        a = self.length * self.width
        return a
    def perimeter(self):
        p = 2 * (self.length + self.width)
        return p
        
class Square(Rectangular):
    def __init__(self, l):
        Rectangular.__init__(self, l, l)

# TEST        
myRec = Rectangular(10,20)
print(myRec.area())
print(myRec.perimeter())

mySqr = Square(10)
print(mySqr.area())
print(mySqr.perimeter())


# 1.2 Numpy applying on class
import numpy as np
length = np.arange(1, 11)
width = np.arange(1, 11)

arr1 = Rectangular(length, width)
# print(length.size, width.size)      #verify array size
print(arr1.area())
print(arr1.perimeter())

# Display Time

class Time:
    hours = 0
    minutes = 0
    seconds = 0
    def __init__(self, h = 0 , m = 0 , s = 0):
        self.hours = h
        self.minutes = m
        self.seconds = s

    def addTime(self, other):
        hrs = self.hours + other.hours
        mins = self.minutes + other.minutes
        secs = self.seconds + other.seconds
        
        while secs > 60:
            secs = secs - 60
            mins = mins + 1    
        while mins > 60:
            mins = mins - 60
            hrs = hrs + 1     
        
        print(hrs, 'hour', mins, 'minute', secs, 'seconds')    
        
    def displayTime(self):        
        while self.seconds > 60:
            self.seconds = self.seconds - 60
            self.minutes = self.minutes + 1    
        while self.minutes > 60:
            self.minutes = self.minutes - 60
            self.hours = self.hours + 1 
            
        print(self.hours, 'hour', self.minutes, 'minute', self.seconds, 'seconds')
        
    def displaySeconds(self):
        totalsecs = self.hours * 60 * 60 + self.minutes * 60 + self.seconds
        print (totalsecs, 'seconds')


# TEST
time1 = Time(2, 50, 10)
time2 = Time(1, 20 ,5)
print("Add time:")
time1.addTime(time2)
print("Display time:")
time1.displayTime()
time3 = Time(1, 2)
print("Dispaly seconds:")
time3.displaySeconds()       




