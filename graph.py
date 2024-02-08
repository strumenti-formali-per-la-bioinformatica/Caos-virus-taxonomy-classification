from cProfile import label
from os import name
from turtle import title
import matplotlib.pyplot as plt
import numpy as np

def metrics_plot():
    kmers_length = [4,6,8,10,12,14,16]
    precision = [66,77,78,79,80,79,80]
    recall = [65,75,77,78,80,78,80]
    f1_Score = [65,76,77,78,80,77,80]

    fig, ax = plt.subplots()
    ax.set(xlabel='Kmer length', ylabel='Value (%)', title="Precision")
    ax.grid()
    ax.plot(kmers_length, precision)
    plt.show()
    fig, ax = plt.subplots()
    ax.set(xlabel='Kmer length', ylabel='Value (%)', title="Recall")
    ax.grid()
    ax.plot(kmers_length, recall)
    plt.show()
    fig, ax = plt.subplots()
    ax.set(xlabel='Kmer length', ylabel='Value (%)', title="F1 Score")
    ax.grid()
    ax.plot(kmers_length, f1_Score)
    plt.show()
    
    
    
def timetakenplot():
    kmers_length = [4,6,8,10,12,14,16]
    time = [301,606,538,709,762,859,989]
    fig = plt.figure(figsize = (8, 5))
 
    # creating the bar plot
    plt.bar(kmers_length, time)
    
    plt.xlabel("Kmer length")
    plt.ylabel("Time taken (seconds)")
    plt.title("Time needed to preprocess the dataset given a specific kmer length")
    plt.show()
    
def metrics_plot2():
    kmers_length = [4,5,6,7,8,9]
    precision = [51,54,60,67,69,73]
    recall = [52,54,60,66,68,73]
    f1_Score = [51,54,60,66,68,73]

    fig, ax = plt.subplots()
    ax.set(xlabel='Kmer length', ylabel='Value (%)', title="Precision")
    ax.grid()
    ax.plot(kmers_length, precision)
    plt.show()
    fig, ax = plt.subplots()
    ax.set(xlabel='Kmer length', ylabel='Value (%)', title="Recall")
    ax.grid()
    ax.plot(kmers_length, recall)
    plt.show()
    fig, ax = plt.subplots()
    ax.set(xlabel='Kmer length', ylabel='Value (%)', title="F1 Score")
    ax.grid()
    ax.plot(kmers_length, f1_Score)
    plt.show()
    
def timetakenplot2():
    kmers_length = [5,6,7,8,9]
    time = [56,169,619,2508,9629]
    fig = plt.figure(figsize = (8, 5))
 
    # creating the bar plot
    plt.bar(kmers_length, time)
    
    plt.xlabel("Kmer length")
    plt.ylabel("Time needed for preprocessing (seconds)")
    plt.title("Time needed to preprocess the dataset on a Ryzen 9 7950X")
    plt.show()
    
if __name__ == '__main__':
    metrics_plot2()
    timetakenplot2()
    
    


