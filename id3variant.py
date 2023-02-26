#SXB210184


import pandas as pd
import numpy as np
import math
from collections import defaultdict


#modify input
#dataset

def EntropyT(positive,negative):

   total = positive + negative
   #print(total) 
   p = positive/total
   #print(p)
   n = negative/total
   #print(n)

   if positive == negative:
       return 1
   elif positive == 0 or negative==0:
       return 0
   else:
        E = -1*(p * math.log(p,2)) + -1*(n * math.log(n,2))
        return(E)
#def calculateAttriEntropies():

dataset = np.loadtxt("dataset-2.txt", dtype=int)
cols = len(dataset[0])
headerCol = []
for i in range(cols):
    if i == cols-1:
        h = "T"
    else:
        h = "A" + str(i+1)
    headerCol.append(h)
trainData = pd.read_csv("dataset-2.txt",sep=" ", names=headerCol)
df = pd.DataFrame(trainData)
#print(df["T"][3])
#print(df.shape)
#print(list(df.columns))

#partition
#partitionSet = np.loadtxt(".txt", dtype=int)
partition = []
with open("d2partition-1.txt") as pFile:
    for line in pFile:
        partition.append(list(map(int, line.split())))
attributeCols = headerCol[:cols-1]
#print(attributeCols)
# masterIndex = 0
maxGainArrayPerPartition = []
partitionIDs = []
partitionDict = defaultdict(list)
for p in partition:
    pID = p[0]
    partitionIDs.append(pID)
    #print("Printing Partition")
    #print(p[1:])
    givenGainArray = []
    #totalPartitionEntropy
    #find subset of dataset
    subset = df.loc[p[1:]]
    partitionSubsetArray = subset["T"].to_numpy()
    pValues = np.count_nonzero(partitionSubsetArray)
    nValues = len(partitionSubsetArray) - pValues
    #print("Total Partition Stats -")
    #print("Positive Values {}".format(pValues))
    #print("Negative Values {}".format(nValues))
    totalPartitionEntropy = EntropyT(pValues, nValues)
    #print("Total Entropy {}".format(totalPartitionEntropy))

    # print(subset)
    startIndex = p[1]
    # print(startIndex)
    uniqueAttributeValues = []
    for h in attributeCols: #Iterate over attributes
        uniqueAttributeValues.append(set(subset[h]))
    #print(uniqueAttributeValues) #Unique values for A1,A2,A3
    for h in range(len(attributeCols)):
        #print("For attribute column - {}".format(attributeCols[h]))
        # print(h)
        setOfAttrValues = uniqueAttributeValues[h]
        #print(setOfAttrValues)
        #print("Master Index {}".format(masterIndex))
        if len(uniqueAttributeValues[h])==1:
            #print("Original Subset - ")
            #print(subset)
            subsetArray = subset["T"].to_numpy()
            pValues = np.count_nonzero(subsetArray)
            nValues = len(subsetArray) - pValues
            #print("Positive Values {}".format(pValues))
            #print("Negative Values {}".format(nValues))
            givenEntropy = EntropyT(pValues, nValues)
            givenGainArray.append(totalPartitionEntropy - givenEntropy)

            continue
        else:
            uniValSuperSet = []
            # print(setOfAttrValues)
            # print("Start Index {}".format(startIndex))
            for uniVal in setOfAttrValues:
                #print(uniVal)
                uniValSubsetIndex = []
                # print(p[1:])
                for row in p[1:]:
                    #print(row)
                    # print(row)
                    # print(attributeCols[h])
                    # print(subset[attributeCols[h]])
                    # print(uniVal)
                    if uniVal == subset[attributeCols[h]][row]:
                        uniValSubsetIndex.append(row)
                uniValSuperSet.append(uniValSubsetIndex)
            # print("UniVal Super Set {}".format(uniValSuperSet))
            subsetDivisionBasedOnAttribute = [] 
            for subsetOfSubset in uniValSuperSet:
                #print("UniVal Super Set {}".format(uniValSuperSet))
                # print(df.loc[subsetOfSubset[:]])
                subsetDivisionBasedOnAttribute.append(df.loc[subsetOfSubset[:]])
            givenEntropy = 0
            for sub in subsetDivisionBasedOnAttribute:
                subsetArray = sub["T"].to_numpy()
                pValues = np.count_nonzero(subsetArray)
                nValues = len(subsetArray) - pValues
                # print("Positive Values {}".format(pValues))
                # print("Negative Values {}".format(nValues))
                givenEntropy += (len(sub)/len(subset))*EntropyT(pValues, nValues)
            givenGainArray.append(totalPartitionEntropy - givenEntropy)
    # print("gain array -")
    # print(givenGainArray)
    for i,v in enumerate(givenGainArray):
        partitionSubList = []
        #partitionSubList.append(pID)
        partitionSubList.append(attributeCols[i])
        partitionSubList.append(v)
        partitionDict[pID].append(partitionSubList)

    #we have gainarray
    maxGainOfPartition = max(givenGainArray)
    maxGainArrayPerPartition.append(maxGainOfPartition)
            
            #print(subsetDivisionBasedOnAttribute)
    
    startIndex+=len(subset)
# print(maxGainArrayPerPartition)
probOfEachPartition = []
for p in partition:
    subset = df.loc[p[1:]]
    probOfPartition = len(subset)/len(dataset)
    probOfEachPartition.append(probOfPartition)

#print(probOfEachPartition)
FArray = []
for i in range(len(maxGainArrayPerPartition)):
    FArray.append(probOfEachPartition[i]*maxGainArrayPerPartition[i])

# print(FArray)
partitionIDWithMaxF = p[0]
MaxF = FArray[0]
for val in range(1,len(FArray)):
    if FArray[val] > MaxF:
        MaxF = FArray[val]
        partitionIDWithMaxF = partitionIDs[val]
#print(partitionDict)
#print(partitionIDs)
#print(partitionIDWithMaxF) #result
#print(MaxF)
# print(partitionDict)
# print(partitionIDWithMaxF)
splitAttribute = headerCol[0]
splitGain = partitionDict[partitionIDWithMaxF][0][1]
for attributeSpecificGain in range(1,len(partitionDict[partitionIDWithMaxF])):
    if partitionDict[partitionIDWithMaxF][attributeSpecificGain][1] > splitGain:
        splitAttribute = partitionDict[partitionIDWithMaxF][attributeSpecificGain][0]

# print(splitAttribute) #result

#partitionsubset
# print(partition)
for p in partition:
    if p[0] == partitionIDWithMaxF:
        splitSubset = df.loc[p[1:]]
        break
# print(splitSubset)
splitSubsets = []
for uniVal in set(splitSubset[splitAttribute]):
    uniValSets = []
    #print(splitSubset[splitAttribute]])
    s = splitSubset[splitSubset[splitAttribute] == uniVal].index.tolist()
    splitSubsets.append(s)
# print(splitSubsets)

#final partition ID
finalID = partition[len(partition)-1][0]
#print(finalID)

#remove old partition
for p in partition:
    if p[0] == partitionIDWithMaxF:
        partition.remove(p)
#print(partition)

substitutePIDS = []
for s in range(len(splitSubsets)):
    substitutePIDS.append(finalID+s+1)
    splitSubsets[s].insert(0,finalID+s+1)
    partition.append(splitSubsets[s])

#print(substitutePIDS)
substitutePIDS = list(map(str,substitutePIDS))
newIDsString = ",".join(substitutePIDS)
#print(newIDsString)
splitAttribute = splitAttribute.replace("A", "Attribute ")
#print(splitAttribute)
#Final print statement
print("Partition {} was replaced with partitions {} using {}".format(partitionIDWithMaxF, newIDsString, splitAttribute))
with open("d2partition-2.txt", "w") as f:
    for p in partition:
        i = map(str,p)
        i = list(i)
        j = " ".join(i)
        f.write(f"{j}\n")


    



    

        

                    
                    

        #find zero and onevalues for attribute-specific value
        
    #HvaluesForEach = []
    #For F-value calculation
    #probOfSet = len(p)/df.shape[0]
    
    #zeroValues,oneValues = calculateValues(p, df)
    # print(zeroValues)
    # print(oneValues)
    # print(EntropyT(oneValues,zeroValues))
        

#[[2,0,9], [5,1,2,3,4]]


