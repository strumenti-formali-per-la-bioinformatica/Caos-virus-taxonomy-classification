import os

if __name__ == '__main__':
    for kmer in [6]:
        for layers in [1]:
            for hidden in [128, 256]:
                for embed_dim in [64, 128, 256]:
                    for embed_dim_mlp in [64, 128, 256]:
                        command: str = f'python3 {os.path.join(os.getcwd(), "main.py ")}' \
                                       f'-read 250 ' \
                                       f'-overlap 200 ' \
                                       f'-k {kmer} ' \
                                       f'-batch 720 ' \
                                       f'-model diff_pool ' \
                                       f'-hidden {hidden} ' \
                                       f'-embedding {embed_dim} ' \
                                       f'-embedding_mlp {embed_dim_mlp} ' \
                                       f'-layers {layers}'

                        os.system(command)
