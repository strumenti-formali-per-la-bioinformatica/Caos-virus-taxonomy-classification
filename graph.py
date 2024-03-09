from cProfile import label
from os import name
from turtle import title
import matplotlib.pyplot as plt
import numpy as np

def metrics_plot():
    kmers_length = [4,6,8,10,12,14,16]
    accuracy = [0.649, 0.757,0.768,0.782,0.799,0.776,0.796]
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
    fig, ax = plt.subplots()
    ax.set(xlabel='Kmer length', ylabel='Value (%)', title="Accuracy")
    ax.grid()
    ax.plot(kmers_length, accuracy)
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
    accuracy = [51.9 ,53.8,59.8,65.8,68.3, 73.0]
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
    fig, ax = plt.subplots()
    ax.set(xlabel='Kmer length', ylabel='Value (%)', title="Accuracy")
    ax.grid()
    ax.plot(kmers_length, accuracy)
    plt.show()


def metrics_plot3():
    kmers_length = [7,9,12,14,16]
    accuracy = [66.8,72.4, 71.7, 70.2,  68.0]
    precision = [70.1,74.4,73.1,73.8 ,72.2]
    recall = [66.8,72.4,71.5 ,70.2,68.0]
    f1_Score = [66.6 ,72.5, 71.2,69.4,67.1]

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
    fig, ax = plt.subplots()
    ax.set(xlabel='Kmer length', ylabel='Value (%)', title="Accuracy")
    ax.grid()
    ax.plot(kmers_length, accuracy)
    plt.show()


def timetakenplot3():
    kmers_length = [7, 9, 12, 14, 16]
    time = [1, 3, 5, 59, 171]
    fig = plt.figure(figsize=(5, 5))
 
    # creating the bar plot with a specified bar width
    plt.bar(kmers_length, time, width=0.8)  # Adjust the width as needed
    
    plt.xlabel("Kmer length")
    plt.ylabel("Time taken (seconds)")
    plt.title("Time needed to preprocess the dataset given a specific kmer length")
    plt.xticks(kmers_length)  # Ensure x-ticks are at the correct k-mer lengths
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
    timetakenplot3()