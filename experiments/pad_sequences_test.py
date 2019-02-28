from keras.preprocessing.sequence import pad_sequences


input1 = [[2,1], [3,4,5]]
output1 = pad_sequences(input1)
print(output1)

input2 = [[3, 4, 5]]
output2 = pad_sequences(input2)
print(output2)
