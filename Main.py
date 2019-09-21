import monkdata as m
import dtree as d
import statistics as st
import matplotlib.pyplot as plt
import random
import drawtree_qt5 as draw


class SplitDataSet:
    def __init__(self):
        self.Train = list()
        self.Test = list()

#    def set_train(self, val):
#        self.Train = val
#
#    def set_test(self, val):
#        self.Test = val
#
#    def get_train(self):
#        return self.Train
#
#    def get_test(self):
#        return self.Test

# Class definition


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


#Dictionary: find the index of the maximum value
#This code is copied from: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
def key_with_maxval(di):
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""
    v = list(di.values())
    k = list(di.keys())
    return k[v.index(max(v))]


def seek_ratio(training_set):
    s_orig = list()
    s_dict = dict()
#    s_var  = list()
    k = SplitDataSet()
    for y in range(30, 81, 10):
        for z in range(0, 30, 1):
            k.Train, k.Test = partition(training_set, y/100.0)
            t_temp = d.buildTree(k.Train, m.attributes)
            s_orig.append(d.check(t_temp, k.Test))
        s_dict[y] = st.mean(s_orig)
#        s_var.append(st.variance(s_orig))
#        print("Ensemble properties:(ratio:", y/100.0, ")")
#        print("Max:", max(s_orig))
#        print("Min:", min(s_orig))
#        print("Average:", st.mean(s_orig))
#        print("Variance:", st.variance(s_orig))
        s_orig.clear()
    return key_with_maxval(s_dict)


def plot_graph(training_set,accuracy,xaxis):
    plt.plot(xaxis, accuracy)
    plt.plot(xaxis, accuracy, 'ro')
    plt.xlabel("Fraction Parameters: %")
    plt.ylabel("Accuracy: ")
    #plt.axis([25, 85, 0.5, 1])
    if training_set == m.monk1:
        plt.title("Data set: monk1.training")
    elif training_set == m.monk2:
        plt.title("Data set: monk2.training")
    elif training_set == m.monk3:
        plt.title("Data set: monk3.training")
    plt.show()
    #print(list(s_dict.keys()), "AND", list(s_dict.values()))


def check_pruning(data_set):
    s_dict = dict()
    t_temp = d.buildTree(data_set.Train, m.attributes)
    prun_set = d.allPruned(t_temp)
    for temp in prun_set:
        s_dict[temp] = (d.check(temp, data_set.Test))
    return key_with_maxval(s_dict)


def test_pruning_algo(train_data, test_data, ratio):
    monk_set = SplitDataSet()
    # Here some uncertainty occurs.
    monk_set.Train, monk_set.Test = partition(train_data, ratio)
    final_tree = check_pruning(monk_set)
    accuracy = d.check(final_tree, test_data)
    #print("Accuracy for Monk1.test", accuracy)
    return accuracy


print(d.entropy(m.monk1))
print(d.entropy(m.monk2))
print(d.entropy(m.monk3))
#Printout the entropy of all datasets.

a = list()
b = list()
c = list()

for i in range(0, 6, 1):
    a.append(d.averageGain(m.monk1, m.attributes[i]))
for i in range(0, 6, 1):
    b.append(d.averageGain(m.monk2, m.attributes[i]))
for i in range(0, 6, 1):
    c.append(d.averageGain(m.monk3, m.attributes[i]))

print(a)
print(b)
print(c)
#
#Calculate and printout the information get for all properties and datasets.
#

#r = d.select(m.monk1, m.attributes[1], 2)
#for x in r:
#    print(x.attribute, "Positive:", x.positive)
# next: calculate the info gain
#To get the majority of one dataset
#print(d.mostCommon(m.monk1test))
t = list()
t.append(d.buildTree(m.monk1, m.attributes))
print("Accuracy for Monk1.test", d.check(t[0], m.monk1test))
print("Accuracy for Monk1", d.check(t[0], m.monk1))
#draw.drawTree(t[0])

#print("Standard decision tree for monk1: ", t)

t.append(d.buildTree(m.monk2, m.attributes))
print("Accuracy for Monk2.test", d.check(t[1], m.monk2test))
print("Accuracy for Monk2", d.check(t[1], m.monk2))

t.append(d.buildTree(m.monk3, m.attributes))
print("Accuracy for Monk3.test", d.check(t[2], m.monk3test))
print("Accuracy for Monk3", d.check(t[2], m.monk3))
#
# Calculate the accuracy
#


#PrunSet = d.allPruned(t)
#draw.drawTree(PrunSet[13])
#Optimal_ratio = list()
#Optimal_ratio.append(seek_ratio(m.monk1)/100.0)
#Optimal_ratio.append(seek_ratio(m.monk2)/100.0)
#Optimal_ratio.append(seek_ratio(m.monk3)/100.0)
#print("Optimal ratio for monk1,2,3:", Optimal_ratio)
Accuracy_monk1 = list()
Accuracy_monk2 = list()
Accuracy_monk3 = list()
temp_1 = list()
temp_2 = list()
temp_3 = list()

for i in range(30, 81, 10):
    for j in range(0, 10, 1):
        temp_1.append(test_pruning_algo(m.monk1, m.monk1test, i/100.0))
        temp_2.append(test_pruning_algo(m.monk2, m.monk2test, i / 100.0))
        temp_3.append(test_pruning_algo(m.monk3, m.monk3test, i/100.0))

#    Accuracy_monk1.append(st.mean(temp_1))
#    Accuracy_monk2.append(st.mean(temp_2))
#    Accuracy_monk3.append(st.mean(temp_3))
    Accuracy_monk1.append(st.variance(temp_1))
    Accuracy_monk2.append(st.variance(temp_2))
    Accuracy_monk3.append(st.variance(temp_3))

    temp_1.clear()
    temp_2.clear()
    temp_3.clear()


plot_graph(m.monk1, Accuracy_monk1, range(30, 81, 10))
plot_graph(m.monk2, Accuracy_monk2, range(30, 81, 10))
plot_graph(m.monk3, Accuracy_monk3, range(30, 81, 10))

#print("Accuracy in average for Monk1:", st.mean(Accuracy_monk1))
#print("Accuracy in average for Monk1:", st.mean(Accuracy_monk2))
#print("Accuracy in average for Monk1:", st.mean(Accuracy_monk3))

#print("Pruning decision tree for monk1: ", final_tree)


#PrunSet = d.allPruned(t)
#draw.drawTree(PrunSet[1])
