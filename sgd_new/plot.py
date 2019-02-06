import csv
import matplotlib.pyplot as plt 
import numpy as np
import statistics


#1. 
def plotcp(logfile, k):
	x = []
	y = []
	with open(logfile) as cp:
		reader  = csv.reader(cp, delimiter=' ')
		i=0
		for row in reader:
			# print i
			# if (len(row)==2):
			if (row[0]=="k="+str(k)):
				x.append(float(row[1]))
				y.append(float(row[2]))
			i+=1
	# plt.xlim(-0.003, 0.006)
	# plt.ylim(-.02, 0.18)
	plt.scatter(x, y)
	plt.show()

# For debugging inner Product
def plotcpline(logfile, k):
	x = []
	y = []
	with open(logfile) as cp:
		reader  = csv.reader(cp, delimiter=' ')
		i=0
		for row in reader:
			# print i
			# if (len(row)==2):
			if (row[0]=="k="+str(k)):
				x.append(float(row[1]))
				y.append(float(row[2]))
			i+=1
	# plt.xlim(-0.003, 0.006)
	# plt.ylim(-.02, 0.18)
	# maxitem = max(x)
	# newx = maxitem*1.001
	# bins = numpy.arange(0, newx, maxitem/10)

	# for i in range(len(x)):

	bins = np.histogram(x, bins=19, range=None)[1]
	indices = np.digitize(x, bins)
	summ = [0]*20
	count = [0]*20

	for i in range(len(x)):
		summ[indices[i]-1]+= y[i]
		count[indices[i]-1] += 1.0


	print count

	plt.scatter(bins, list(np.array(summ)/np.array(count)))
	plt.show()


#2.
def plottokip(logfile):
	y1 = []
	y2 = []
	with open(logfile) as cp:
		reader  = csv.reader(cp, delimiter=' ')
		i=0
		for row in reader:
			# print i
			# if (len(row)==2):
			if (row[0]=="hashingtopkmean="):
				y1.append(float(row[1]))
			if (row[0]=="overallmean="):
				y2.append(float(row[1]))
			i+=1
	# plt.xlim(-0.003, 0.006)
	# plt.ylim(-.02, 0.18)
	plt.plot(range(len(y1)), y1, color='g')
	plt.plot(range(len(y2)), y2, color='r')
	plt.show()


#3. 
def plotnode(logfile):
	y1 = {}
	y2 = {}
	with open(logfile) as cp:
		reader  = csv.reader(cp, delimiter=' ')
		i=0
		for row in reader:
			# print i
			# if (len(row)==2):
			if (row[0]=="useNode"):
				if len(row)>1:
					for j in range(1, len(row)-1):
					
						node = int(row[j])
						if node in y1:
							y1[node] +=1
						else:
							y1[node] = 1
			if (row[0]=="useNodeat"):
				if len(row)>1:
					for j in range(1, len(row)-1):
					
						node = int(row[j])
						if node in y2:
							y2[node] +=1
						else:
							y2[node] = 1
			i+=1
			# if i>1000:
			# 	break

	# plt.xlim(-0.003, 0.006)
	# plt.ylim(-.02, 0.18)

	plt.plot(y1.keys(), y1.values())

	print(np.std(np.array(y1.values())))
	print(np.std(np.array(y2.values())))
	plt.plot(y2.keys(), y2.values(), color='r')
	plt.show()


#4
def plotmaxip(logfile):
	x = []
	y = []
	z = []
	with open(logfile) as cp:
		reader  = csv.reader(cp, delimiter=' ')
		i=0
		for row in reader:
			# print i
			# if (len(row)==2):
			if (row[0]=="minmax="):
				x.append(float(row[1]))
				y.append(float(row[2]))
				z.append(float(row[1])-float(row[2]))
			i+=1
	# plt.xlim(-0.003, 0.006)
	# plt.ylim(-2, 14)
	plt.plot(y, label="max inner")
	# plt.plot(range(len(y1)), y1, color='g', label="avg selected grad")
	# plt.plot(range(len(y2)), y2, color='r', label="avg grad")
	plt.legend(loc='upper left')
	print(sum(y))
	plt.show()


#5

def plotdelta(logfile):
	x = []
	y = []
	z = []
	with open(logfile) as cp:
		reader  = csv.reader(cp, delimiter=' ')
		for row in reader:
			# print i
			# if (len(row)==2):

			if (row[0]=="delta="):
				total = 0
				for j in range(1, len(row)-1):
					total+=abs(float(row[j]))
				y.append(total)
	# plt.ylim(0, 0.002)

	plt.plot(y, label="delta")
	print(sum(y))
	# plt.plot(range(len(y1)), y1, color='g', label="avg selected grad")
	# plt.plot(range(len(y2)), y2, color='r', label="avg grad")
	plt.legend(loc='upper left')
	plt.show()


logfile = "log_mnist_recall_adam"
logfile = "log_aloi"


# plotcp(logfile, 2)

plotcpline(logfile, 1)

# plottokip(logfile)

# plotnode(logfile)

# plotmaxip(logfile)

# plotdelta(logfile)







#log_oldwta_k=2_big_eval
#log_total_full
#log_oldwta_k=2_full
#log_origin_commit_no_normalization


# files = ["log_oldwta_k=2_big_eval", "log_total_full", "log_oldwta_k=2_full", "log_origin_commit_no_normalization"]
# legends = ["k=2", "full", "fulltest", "k=1, Bucketsize=1000"]
# i=0
# for file in files:
# 	with open("logs/"+file) as cp:
# 		reader = csv.reader(cp, delimiter=' ')
# 		y = []

# 		for row in reader:
# 			if (row[0]=="over"):
# 				y.append(float(row[2]))
# 		plt.plot(range(len(y)), y, label=legends[i])
# 		i+=1
# plt.legend()
# plt.show()















# with open("log_revert_commit_0.001") as cp:
# 	reader  = csv.reader(cp, delimiter=' ')
# 	maxacc = 0
# 	for row in reader:
# 		if row[0]=="over":
# 			if float(row[2])>maxacc:
# 				maxacc = float(row[2])

# 	print maxacc