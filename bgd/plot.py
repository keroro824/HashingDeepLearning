import csv
import matplotlib.pyplot as plt 
import numpy as np



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








# For debugging inner Product
# x = []
# y = []
# with open("logs/log_adam") as cp:
# 	reader  = csv.reader(cp, delimiter=' ')
# 	i=0
# 	for row in reader:
# 		# print i
# 		# if (len(row)==2):
# 		if (row[0]=="k=1"):
# 			x.append(float(row[1]))
# 			y.append(float(row[2]))
# 		i+=1
# # plt.xlim(-0.003, 0.006)
# # plt.ylim(-.02, 0.18)
# # maxitem = max(x)
# # newx = maxitem*1.001
# # bins = numpy.arange(0, newx, maxitem/10)

# # for i in range(len(x)):

# bins = np.histogram(x, bins=19, range=None)[1]
# indices = np.digitize(x, bins)
# summ = [0]*20
# count = [0]*20

# for i in range(len(x)):
# 	summ[indices[i]-1]+= y[i]
# 	count[indices[i]-1] += 1.0


# print count

# plt.scatter(bins, list(np.array(summ)/np.array(count)))
# plt.show()






x = []
y = []
with open("logs/cp_k=123") as cp:
	reader  = csv.reader(cp, delimiter=' ')
	i=0
	for row in reader:
		# print i
		# if (len(row)==2):
		if (row[0]=="k=1"):
			x.append(float(row[1]))
			y.append(float(row[2]))
		i+=1
# plt.xlim(-0.003, 0.006)
# plt.ylim(-.02, 0.18)
plt.scatter(x, y)
plt.show()


# with open("log_revert_commit_0.001") as cp:
# 	reader  = csv.reader(cp, delimiter=' ')
# 	maxacc = 0
# 	for row in reader:
# 		if row[0]=="over":
# 			if float(row[2])>maxacc:
# 				maxacc = float(row[2])

# 	print maxacc