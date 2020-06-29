# Ja, CED blir ett v�rde d� theta=t=1, ett annat d� theta=t=2, osv. Sparar ni dessa v�rden i en vektor och plottar dem mot t.
# N�r ni skattar CED ska ni g�ra enligt f�ljande:



# 1. simulera ett v�rde p� change-pointen theta (eventuellt flera change-points om ni t�nker att processen ligger och skiftar fram och tillbaks)
# 2. simulera processen x_t: x_1, x_2, x_3, ... , x_{\theta-1} enligt N(0,1) och x_\theta, x_{\theta+1}, ... enligt N(1,1) eller likanande ur kontroll
# 3. ber�kna larmet T d� ni till�mpar er CUSUM p� dessa simulerade data och kontrollera om T<theta eller T>=theta.
# 4. Om T>=theta spara T-theta i en vektor
# 5. B�rja om fr�n 1. tills ni gjort 10^7 simuleringar typ.

import math
import numpy.random as r
import matplotlib.pyplot as plt
import random

def ced_normal():
	print("Normal")
	
	# Igen manuel initiering av trösklar osv
	# Ett exempel av vår Fuzzy för ARL 200 0.045 fönster syns
	
	print("Fuzzy ARL 200 0.045")
	# limits är trösklar
	limit_n = limit_p =  1.42
	G_p_l = []
	# d_p och d_n är differansen som vi räknar med.
	d_p = d_n = -4.53006
	# loc_change är mu ur kontroll
	loc_change = -4.53006
	# sigma_change är sigma ur kontroll 
	sigma_change = 3.60327
	
	# antal simuleringar för varje changepoint
	ammount = 10000000 #REAL
	# De changepoints som vi ska titta på just här från 1 till 6-1=5
	for cp in range(1,6):
		res = []
		sum = 0
		for i in range(1,ammount):
			c = 1
			G_p = 0 
			G_n = 0
			while True:
				if c >= cp:
					x = r.normal(loc=loc_change,scale=sigma_change,size=1)
				else:
					x = r.normal(loc=0,scale=1,size=1)
				# Beräkning för skifte i mu 
				G_p = max(G_p + abs(d_p) *(x-(abs(d_p)/2)),0)
				G_n = max(G_n - abs(d_n)*(x+abs(d_n)/2),0)
				
				# Beräkning för skifte i standardavvikelse
				
				# G_p = max(G_p - math.log(1+abs(d_p))-((x**2)/2)*((1/(1+abs(d_p))**2)-1),0)
				# G_n = max(G_n - math.log(1-d_n)-((x**2)/2)*((1/(1-d_n)**2)-1),0)
				
				
				
				
				# Kan vara användbart om det positiva skiftet är större än 1 då det skulle
				# vara matematiskt omöjligt att beräkna G_n i detta fall.
				# Det är bara en snabb och ful lösning då vi initierar d_p = d_n. 
				
				# G_n = 0
				
				# test för TRUE LARM
				if (G_p >= limit_p or G_n >= limit_n) and c >= cp:
					# print("FOUND ONE ",c-cp)
					res.append(c-cp)
					sum += c-cp
					break
				# Gå vidare för falskt utan att räknad det
				elif (G_p >= limit_p or G_n >= limit_n) and c < cp:
					# print("False larm")
					# print(c)
					break
				c += 1
			# Denna del är endast för att kunna se ungefär vart CED befinner 
			# sig värdemässigt medans den fortfarande körs.
			# OBS! Den tar en massa processkraft så det tar längre tid om man
			# dynamiskt vill se resultaten.
			
			# try:
			# 	print(i,res[-1],end='\r')
			# except IndexError:
			# 	pass
		
		plt.show()
		print("CP: ",cp," CED: ",sum/len(res))
		print(len(res))
		# print(res)


ced_normal()