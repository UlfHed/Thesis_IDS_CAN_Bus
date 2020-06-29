# import scipy.stats as stats
import math
import numpy.random as r

class tester(object):
	"""docstring for tester"""
	def __init__(self):
		self.G_p = 0
		self.G_n = 0
		self.sum_p = 0
		self.sum_n = 0
		self.c = 0
		self.ammount = 1000000
		self.l = 0
		self.res = []

	def initiate(self):
		print("Give limit_p & limit_n")
		self.limit_p = input("limit_p: ")
		self.limit_n = input("limit_n: ")
		print("Give d_p & d_n")
		self.d_p = input("d_p: ")
		self.d_n = input("d_n: ")
		print("Give l in normal state")
		self.l = input("l: ")

	def test_normal(self,v):
		# Test funktion för skifte i mu 
		self.G_p = max(self.G_p + self.d_p *(v-(self.d_p/2)),0)
		self.G_n = max(self.G_n - abs(self.d_n)*(v+self.d_n/2),0)
	def test_normal_variance(self,v):
		# Test funktion för skifte i standardavvikelse?
		self.G_p = max(self.G_p - math.log(1+self.d_p)-((v**2)/2)*((1/(1+self.d_p)**2)-1),0)
		self.G_n = max(self.G_n - math.log(1-self.d_n)-((v**2)/2)*((1/(1-self.d_n)**2)-1),0)


	def run(self,mode='n'):
		if mode=='n':
			# Vi har valt att ge den värden manuelt här
			# Det finns en initiernings funktion som gör liknande för sig själv.
			print("RPM ARL 200 0.045")
			# limits är tröskelvärden som vi testar
			self.limit_n = self.limit_p = 2.414
			
			self.d_n = 0
			self.d_p = 3.3008
			for i in range(self.ammount):
				self.G_p = 0
				self.G_n = 0
				self.c = 0
				positive = False
				negative = False
				while True:
					v = r.normal(loc=0,scale=1,size=1)
					# den testar standardavvikelse!
					self.test_normal_variance(v)
					if self.G_p >= self.limit_p and not positive:
						self.sum_p += self.c
						positive = True
					# Under är ifall vi hade negativt skifte
					
					# if self.G_n >= self.limit_n and not negative:
					# 	self.sum_n += self.c
					# 	negative = True
					negative = True
						# print("Negative: ",self.c)

					if negative and positive:
						# print(self.sum_p,self.sum_n)
						break
					self.c += 1
				# Denna del är endast för att kunna se ungefär vart ARL befinner 
				# sig värdemässigt medans den fortfarande körs. Vilket kan vara hjälpsamt
				# om man som vi manuelt letar efter rätta tröskelvärde.
				# OBS! Den tar en massa processkraft så det tar längre tid om man
				# dynamiskt vill se resultaten.
				try:
					print(" i: ",round(i/self.ammount,4)," Normal_ARL_p: ",self.sum_p/i, " Normal_ARL_n: ",self.sum_n/i,end='\r')
				except ZeroDivisionError:
					pass
			print("Normal_ARL_p: ",self.sum_p/i, " Normal_ARL_n: ",self.sum_n/i," i: ",round(i/self.ammount,2))


t = tester()
t.run('n')