import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from foresight import ei


##################################
####            H             ####
##################################

def test_H_0():
    x = torch.zeros((4,))
    x[1] = 1
    assert ei.H(x).item() == 0

def test_H_1():
    x = torch.ones((4,)) / 4
    assert ei.H(x).item() == 2

def test_H_2():
    x = torch.ones((256,)) / 256
    assert ei.H(x).item() == 8

def test_H_3():
    x = torch.ones((4,4)) / 4
    assert all(ei.H(x, dim=0) == 2)

def test_H_4():
    x = torch.ones((4,4)) / 4
    assert all(ei.H(x, dim=1) == 2)

def test_H_5():
    x = torch.zeros((4,))
    assert ei.H(x).item() == 0


##################################
####         lin_norm         ####
##################################

def test_lin_norm_0():
    x = torch.ones((4,4))
    x_normed = ei.lin_norm(x) == 0.25
    for row in x_normed:
        assert all(row)

def test_lin_norm_1():
    """Check that negative entries become 0."""
    x = torch.ones((5, 5))
    x[:, 0] = -1
    x_normed = ei.lin_norm(x)
    assert all(x_normed[:, 0] == 0)
    for row in x_normed[:, 1:]:
        assert all(row == 0.25)

def test_lin_norm_2():
    """Check that rows of all 0s stay all 0s (no nan values via division by 0)."""
    x = torch.zeros((4,4))
    x_normed = ei.lin_norm(x)
    for row in x_normed:
        assert all(row == 0)


##################################
####   conv2d_create_matrix   ####
##################################

def test_conv2d_create_matrix_0():
    m = nn.Conv2d(1, 2, 2)
    m.weight = nn.Parameter(torch.ones((2, 1, 2, 2)))
    matrix = ei.conv2d_create_matrix(m, (1, 1, 3, 3), (1, 2, 2, 2))
    correct_matrix = torch.tensor([
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1]
        ]).to(torch.float32).t()
    assert all(torch.flatten(matrix == correct_matrix))

def test_conv2d_create_matrix_1():
    m = nn.Conv2d(1, 1, 2, stride=2)
    m.weight = nn.Parameter(torch.ones((1, 1, 2, 2)))
    matrix = ei.conv2d_create_matrix(m, (1, 1, 4, 4), (1, 1, 2, 2))
    correct_matrix = torch.tensor([
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
        ]).to(torch.float32).t()
    assert all(torch.flatten(matrix == correct_matrix))

def test_conv2d_create_matrix_2():
    m = nn.Conv2d(2, 1, 2)
    m.weight = nn.Parameter(torch.ones((1, 2, 2, 2)))
    matrix = ei.conv2d_create_matrix(m, (1, 2, 3, 3), (1, 1, 2, 2))
    correct_matrix = torch.tensor([
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1]
        ]).to(torch.float32).t()
    assert all(torch.flatten(matrix == correct_matrix))

def test_conv2d_create_matrix_3():
    m = nn.Conv2d(1, 1, 2, padding=1)
    m.weight = nn.Parameter(torch.ones((1, 1, 2, 2)))
    matrix = ei.conv2d_create_matrix(m, (1, 1, 3, 3), (1, 1, 4 ,4))
    correct_matrix = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]).to(torch.float32).t()
    assert all(torch.flatten(matrix == correct_matrix))

def test_conv2d_create_matrix_4():
    m = nn.Conv2d(1, 1, 2, padding=1, stride=2)
    m.weight = nn.Parameter(torch.ones((1, 1, 2, 2)))
    matrix = ei.conv2d_create_matrix(m, (1, 1, 3, 3), (1, 1, 2, 2))
    correct_matrix = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 1]
        ]).to(torch.float32).t()
    assert all(torch.flatten(matrix == correct_matrix))

def test_conv2d_create_matrix_5():
    m = nn.Conv2d(1, 1, (1, 2), padding=1, stride=(2, 2))
    m.weight = nn.Parameter(torch.ones((1, 1, 1, 2)))
    matrix = ei.conv2d_create_matrix(m, (1, 1, 3, 3), (1, 1, 3, 2))
    correct_matrix = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]).to(torch.float32).t()
    assert all(torch.flatten(matrix == correct_matrix))


##################################
#### avgpool2d_create_matrix  ####
##################################

def test_avgpool2d_create_matrix_0():
    m = nn.AvgPool2d(2)
    matrix = ei.avgpool2d_create_matrix(m, (1, 1, 4, 4), (1, 1, 2, 2))
    correct_matrix = torch.tensor([
            [0.25, 0.25, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25]
        ]).t()
    assert all(torch.flatten(matrix == correct_matrix))

def test_avgpool2d_create_matrix_1():
    m = nn.AvgPool2d(2, stride=1)
    matrix = ei.avgpool2d_create_matrix(m, (1, 1, 3, 3), (1, 1, 2, 2))
    correct_matrix = torch.tensor([
            [0.25, 0.25, 0, 0.25, 0.25, 0, 0, 0, 0],
            [0, 0.25, 0.25, 0, 0.25, 0.25, 0, 0, 0],
            [0, 0, 0, 0.25, 0.25, 0, 0.25, 0.25, 0],
            [0, 0, 0, 0, 0.25, 0.25, 0, 0.25, 0.25]
        ]).t()
    assert all(torch.flatten(matrix == correct_matrix))

def test_avgpool2d_create_matrix_2():
    m = nn.AvgPool2d((1, 2), stride=1)
    matrix = ei.avgpool2d_create_matrix(m, (1, 1, 3, 3), (1, 1, 3, 2))
    correct_matrix = torch.tensor([
            [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0.5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.5, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5]
        ]).t()
    assert all(torch.flatten(matrix == correct_matrix))


##################################
####       determinism        ####
##################################

def test_determinism_0():
    m = nn.Linear(4, 4, bias=False)
    m.weight = nn.Parameter(torch.ones((4, 4)))
    computed_det = ei.determinism(m, input=torch.randn(1, 4))
    true_det = 2
    assert type(computed_det) == float and computed_det == true_det


##################################
####        degeneracy        ####
##################################

def test_degeneracy_0():
    m = nn.Linear(4, 4, bias=False)
    m.weight = nn.Parameter(torch.ones((4, 4)))
    computed_deg = ei.degeneracy(m, input=torch.randn(1, 4))
    true_deg = 2
    assert type(computed_deg) == float and computed_deg == true_deg


##################################
####            ei            ####
##################################

def test_ei_0():
    m = nn.Linear(4, 4, bias=False)
    m.weight = nn.Parameter(torch.ones((4, 4)))
    computed_ei = ei.ei(m, input=torch.randn(1, 4))
    true_ei = 0
    assert type(computed_ei) == float and computed_ei == true_ei


