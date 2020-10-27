#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import yaml
import timeit
from latency import Latency




test_data  =  Latency.PrepareGroundTruth("gt.yml")


pred_data  =  Latency.PreparePredictionsData("pred.yml")


latency = Latency.LatencyCalculation(test_data, pred_data)


maks_latency =  Latency.getMax(latency)


average_latency = Latency.getAverage(latency)

