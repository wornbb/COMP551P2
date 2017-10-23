import pickle
buffer = pickle.load(open("buffer","rb"))
a = [sum(buffer[index][1]) for index in range(0, 26)]
print(a)