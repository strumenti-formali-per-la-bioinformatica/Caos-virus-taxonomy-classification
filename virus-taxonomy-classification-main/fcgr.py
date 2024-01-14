
import os
from Bio import SeqIO
from complexcgr import FCGR

fasta_directory = './data/raw'
output_directory = './img'

if(not os.path.exists(output_directory)):
    os.makedirs(output_directory)
    
for file_name in os.listdir(fasta_directory):
    if file_name.endswith('fasta'):
        file_path = os.path.join(fasta_directory, file_name)
        for record in SeqIO.parse(file_path, 'fasta'):
            sequence = str(record.seq)
            for letter in "BDEFHIJKLMOPQRSUVWXYZ":
                sequence = sequence.replace(letter,"N")
            fcgr = FCGR(k=8, bits=16)
            chaos = fcgr(sequence)
            image_name=f"{os.path.splitext(file_name)[0]}_{record.id}.jpg"
            image_path=os.path.join(output_directory, image_name)
            fcgr.save_img(chaos, path=image_path)
        