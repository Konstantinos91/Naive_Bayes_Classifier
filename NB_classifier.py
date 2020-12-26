
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.classify.api import ClassifierI
import nltk
from nltk import re, word_tokenize, FreqDist

with open('sampleTrain.txt', 'r') as infile:
    data = infile.read() 
my_list = data.splitlines()

Counter0 = 0
Counter1 = 0

for i in range(0,5):
    x = my_list[i].split("	")    
    if x[1] == "0" :
        Counter0 = Counter0 + 1       
    if x[1] == "1" :
        Counter1 = Counter1 + 1
print("-------------------------")
print("First Part")
print("-------------------------")
print ("Training Prior probabilities")
print ("Class 0 =",Counter0/(i+1))
print ("Class 1 =",Counter1/(i+1))
print ("---------------------")

zero_list = []
one_list = []
prediction = []
for i in range(0,5):
    x = my_list[i].split("	")   
    x2 = x[2].split(" ")
    for p in range(0, len(x2)):
        if x[1] == "0" :
            zero_list.append(x2[p])      
        if x[1] == "1" :
            one_list.append(x2[p])

with open('sampleTrain.vocab.txt', 'r') as infile:
    data2 = infile.read() 
my_list2 = data2.splitlines()
zero_total = []
zero_total_list = []
zero_total_mylist = []
one_total = []
one_total_list = []
one_total_mylist = []
for line in my_list2:
    zero_count = 0
    one_count = 0    
    for p in range(0,len(zero_list)):
        if zero_list[p] == line :
            zero_count = zero_count + 1 
    for l in range(0,len(one_list)):
        if one_list[l] == line :
            one_count = one_count + 1
    zero_total.append(zero_count)
    zero_total_list.append(len(zero_list))
    zero_total_mylist.append(len(my_list))
    one_total.append(one_count)
    one_total_list.append(len(one_list))
    one_total_mylist.append(len(my_list))




print("Feature likelihoods")
print("          great    movie     just      and  enjoyable  sad      but     cheap    boring") 

print("class 0 ", ' {:01.5f}'.format((zero_total[0]+1)/(zero_total_list[0]+zero_total_mylist[0])), 
                  ' {:01.5f}'.format((zero_total[1]+1)/(zero_total_list[0]+zero_total_mylist[1])), 
                  ' {:01.5f}'.format((zero_total[2]+1)/(zero_total_list[0]+zero_total_mylist[2])), 
                  ' {:01.5f}'.format((zero_total[3]+1)/(zero_total_list[0]+zero_total_mylist[3])),
                  ' {:01.5f}'.format((zero_total[4]+1)/(zero_total_list[0]+zero_total_mylist[4])),
                  ' {:01.5f}'.format((zero_total[5]+1)/(zero_total_list[0]+zero_total_mylist[5])),
                  ' {:01.5f}'.format((zero_total[6]+1)/(zero_total_list[0]+zero_total_mylist[6])),
                  ' {:01.5f}'.format((zero_total[7]+1)/(zero_total_list[0]+zero_total_mylist[0])),
                  ' {:01.5f}'.format((zero_total[8]+1)/(zero_total_list[0]+zero_total_mylist[0]))
      ) 

print("class 1 ", ' {:01.5f}'.format((one_total[0]+1)/(one_total_list[0]+one_total_mylist[0])), 
                  ' {:01.5f}'.format((one_total[1]+1)/(one_total_list[0]+one_total_mylist[1])), 
                  ' {:01.5f}'.format((one_total[2]+1)/(one_total_list[0]+one_total_mylist[2])), 
                  ' {:01.5f}'.format((one_total[3]+1)/(one_total_list[0]+one_total_mylist[3])),
                  ' {:01.5f}'.format((one_total[4]+1)/(one_total_list[0]+one_total_mylist[4])),
                  ' {:01.5f}'.format((one_total[5]+1)/(one_total_list[0]+one_total_mylist[5])),
                  ' {:01.5f}'.format((one_total[6]+1)/(one_total_list[0]+one_total_mylist[6])),
                  ' {:01.5f}'.format((one_total[7]+1)/(one_total_list[0]+one_total_mylist[0])),
                  ' {:01.5f}'.format((one_total[8]+1)/(one_total_list[0]+one_total_mylist[0]))
      ) 
print ("---------------------")

train = [
    (dict(a="great",b="movie"), '0'),
    (dict(a="just",b="great",c="and" , d="enjoyable"), '0'),
    (dict(a="sad",b="movie",c="but" , d="enjoyable"), '0'),
    (dict(a="cheap",b="and",c="cheap"), '1'),
    (dict(a="great",b="sad",c="and" , d="boring"), '1'),
    ]
test = [
    (dict(a="just",b="great")),
    (dict(a="sad",b="and",c="boring")),
    (dict(a="great",b="great")),
    (dict(a="enjoyable")),
    (dict(a="sad",b="movie")),
    (dict(a="just",b="boring")),
    ]
classifier = nltk.classify.NaiveBayesClassifier.train(train)
prediction = classifier.classify_many(test)
print("Predictions on test data") 
print("d5 = ",prediction[0]) 
print("d6 = ",prediction[1]) 
print("d7 = ",prediction[2]) 
print("d8 = ",prediction[3]) 
print("d9 = ",prediction[4]) 
print("d10 = ",prediction[5])

true = 0
false = 0
with open('sampleTest.txt', 'r') as infile:
    data = infile.read() 
my_list = data.splitlines()



print ("---------------------")
for i in range(0,len(prediction)):
    x = my_list[i].split("	")    
    if x[1] == prediction[i]:
        true = true + 1    
    else :
        false = false + 1

print("Accuracy:",'%.2f'%((true / (false+true))*100),"%")

#Second Part

data = [] 
class_0 = [] 
class_1 = [] 
class_2 = [] 
class_3 = [] 
class_4 = []
print("\n\n-------------------------")
print("Second_Part")
print("-------------------------")
with open('train.txt', 'r') as rtext: 
    data = rtext.read() 
    for line in rtext: 
        data.append(line.strip().split()) 
 
all_tok = word_tokenize(data) 
def prior_prob(x): 
    n_class = all_tok.count(x) 
    total_data = all_tok.count("0") + all_tok.count("1") + all_tok.count("2")+ all_tok.count("3")+ all_tok.count("4") 
    return n_class / total_data 
 
#https://docs.python.org/2/tutorial/datastructures.html
reg = re.compile(r'[a-z0-9]+\s([01234])\s((.*))') 
for n in re.finditer(reg, data): 
    if n.group(1) == '0': 
        class_0.append(n.group(2)) 
    elif n.group(1) == '1': 
        class_1.append(n.group(2)) 
    elif n.group(1) == '2': 
        class_2.append(n.group(2)) 
    elif n.group(1) == '3': 
        class_3.append(n.group(2)) 
    elif n.group(1) == '4': 
        class_4.append(n.group(2)) 
  
#list to string: https://www.datacamp.com/community/tutorials/18-most-common-python-list-questions-learn-python#question1
# https://www.tutorialspoint.com/python/string_join.htm       
seqclass_0 = ''.join(class_0) 
seqclass_1 = ''.join(class_1) 
seqclass_2 = ''.join(class_2) 
seqclass_3 = ''.join(class_3) 
seqclass_4 = ''.join(class_4) 
 
tok_class0 = word_tokenize(seqclass_0) 
tok_class1 = word_tokenize(seqclass_1) 
tok_class2 = word_tokenize(seqclass_2) 
tok_class3 = word_tokenize(seqclass_3) 
tok_class4 = word_tokenize(seqclass_4) 
total_tok_vocab = tok_class0 + tok_class1 + tok_class2 + tok_class3 + tok_class4 
 
freq_class0 = FreqDist(tok_class0) 
freq_class1 = FreqDist(tok_class1) 
freq_class2 = FreqDist(tok_class2) 
freq_class3 = FreqDist(tok_class3) 
freq_class4 = FreqDist(tok_class4) 
 
voc = set(total_tok_vocab) 
len_voc = len(set(total_tok_vocab)) 
length0 = len(tok_class0) 
length1 = len(tok_class1) 
length2 = len(tok_class2) 
length3 = len(tok_class3) 
length4 = len(tok_class4) 

print("Prior probabilities") 
print("class 0 = ", '{:01.7f}'.format(prior_prob("0"))) 
print("class 1 = ", '{:01.7f}'.format(prior_prob("1"))) 
print("class 2 = ", '{:01.7f}'.format(prior_prob("2"))) 
print("class 3 = ", '{:01.7f}'.format(prior_prob("3"))) 
print("class 4 = ", '{:01.7f}'.format(prior_prob("4")))   
print("Feature likelihoods") 
print("           computer    baseball        god     doctor ") 
print("class 0 ", ' {:01.7f}'.format((freq_class0["computer"]+1)/(length0+len_voc)), 
                  '  {:01.7f}'.format((freq_class0["baseball"]+1)/(length0+len_voc)), 
                  ' {:01.7f}'.format((freq_class0["god"]+1)/(length0+len_voc)), 
                  ' {:01.7f}'.format((freq_class0["doctor"]+1)/(length0+len_voc)) 
      ) 
print("class 1 ", ' {:01.7f}'.format((freq_class1["computer"]+1)/(length1+len_voc)), 
                  '  {:01.7f}'.format((freq_class1["baseball"]+1)/(length1+len_voc)), 
                  ' {:01.7f}'.format((freq_class1["god"]+1)/(length1+len_voc)), 
                  ' {:01.7f}'.format((freq_class1["doctor"]+1)/(length1+len_voc)) 
      ) 
print("class 2 ", ' {:01.7f}'.format((freq_class2["computer"]+1)/(length2+len_voc)), 
                  '  {:01.7f}'.format((freq_class2["baseball"]+1)/(length2+len_voc)), 
                  ' {:01.7f}'.format((freq_class2["god"]+1)/(length2+len_voc)), 
                  ' {:01.7f}'.format((freq_class2["doctor"]+1)/(length2+len_voc)) 
      ) 
print("class 3 ", ' {:01.7f}'.format((freq_class3["computer"]+1)/(length3+len_voc)), 
                  '  {:01.7f}'.format((freq_class3["baseball"]+1)/(length3+len_voc)), 
                  ' {:01.7f}'.format((freq_class3["god"]+1)/(length3+len_voc)), 
                  ' {:01.7f}'.format((freq_class3["doctor"]+1)/(length3+len_voc)) 
      ) 
print("class 4 ", ' {:01.7f}'.format((freq_class4["computer"]+1)/(length4+len_voc)), 
                  '  {:01.7f}'.format((freq_class4["baseball"]+1)/(length4+len_voc)), 
                  ' {:01.7f}'.format((freq_class4["god"]+1)/(length4+len_voc)), 
                  ' {:01.7f}'.format((freq_class4["doctor"]+1)/(length4+len_voc)) 
      ) 
 
print("Accuracy:")








               
               
               
