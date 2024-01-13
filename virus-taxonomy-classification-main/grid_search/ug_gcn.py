import os

if __name__ == '__main__':
    for kmer in [4]:
        for transformer_heads in [1]:
            for attention_layers in [1, 2, 3]:
                for layers in [1]:
                    for hidden in [64, 128, 256]:
                        for embed_dim in [64, 128, 256]:
                            command: str = f'python3 {os.path.join(os.getcwd(), "main.py ")}' \
                                           f'-read 250 ' \
                                           f'-overlap 200 ' \
                                           f'-k {kmer} ' \
                                           f'-batch 720 ' \
                                           f'-model ug_gcn ' \
                                           f'-hidden {hidden} ' \
                                           f'-embedding {embed_dim} ' \
                                           f'-layers {layers} ' \
                                           f'-att_layers {attention_layers} ' \
                                           f'-tf_heads {transformer_heads}'

                            os.system(command)
