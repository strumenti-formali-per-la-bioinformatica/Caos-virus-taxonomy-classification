from typing import Final

import subprocess
import time
import sys
import os

DATA_DIR: Final = os.path.join(os.getcwd(), 'data')
RAW_DIR: Final = os.path.join(DATA_DIR, 'raw')
FILE_NAME: Final = 'sequence_identifiers.txt'

if __name__ == "__main__":
    # check if data/raw dir exists
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
    # get api_key
    api_key: str = sys.argv[1]
    # open file
    sequence_identifiers_file = open(os.path.join(DATA_DIR, FILE_NAME))
    sequence_identifiers = sequence_identifiers_file.readlines()
    # init identifiers
    identifiers: str = ''
    for index, identifier in enumerate(sequence_identifiers):
        # get accession id
        identifier: str = identifier.strip()
        # add it on list of identifier
        identifiers = f'{identifiers} {identifier}'
        if (index % 50 == 0 and index != 0) or index == len(sequence_identifiers) - 1:
            # init command
            command: str = f'ncbi-acc-download ' \
                           f'--api-key {api_key} ' \
                           f'--format fasta ' \
                           f'-o {os.path.join(RAW_DIR, f"file_{index}.fasta")} ' \
                           f'{identifiers}'
            print(command)
            subprocess.run(command, shell=True)
            time.sleep(2)
            identifiers = ''
