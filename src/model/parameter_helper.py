def assemble_FK(n_layers, n_channels, n_polynomials):
    FK = (str(n_channels) + "," + str(n_polynomials) + ",") * n_layers
    return FK[:-1] # skip the last comma