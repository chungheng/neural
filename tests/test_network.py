import pytest
from neural.network import (
    Symbol, Input, Container, Network    
)
from neural.model.neuron import HodgkinHuxley

def test_container():
    hhn = Container(HodgkinHuxley)