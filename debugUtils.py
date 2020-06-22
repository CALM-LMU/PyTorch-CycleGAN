#!/usr/bin/python3
def out(tensor):
    return (tensor.cpu().detach().numpy().shape)