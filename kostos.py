import argparse
import csv
import itertools
import json
import math
import numpy as np
import subprocess
import sys

from copy import copy
from operator import itemgetter

def labeled_params(param_dict):
    ''' {p:[v]} -> [(p, v1), ..., (p, vn)] '''
    ret = []
    for p, vlist in param_dict.items():
        p_list = []
        for val in vlist:
            p_list.append((p, val))
        ret.append(p_list)
    return ret

'''
parse output of `bench` binary
'''
def parse_output(output):
    output_lines = output.decode("utf-8").split("\n")
    times = []
    for output_line in output_lines:
        output_line = output_line.strip()
        if output_line == "":
            continue
        output_line_tokens = output_line.split(": ")
        scheme = output_line_tokens[0]
        time = float(output_line_tokens[1].split()[0])
        times.append((scheme, time))
    return times

'''
parse output of C measurement
'''
def parse_measurement(output):
    output_lines = output.decode("utf-8").split("\n")

    for output_line in output_lines:
        output_line = output_line.strip()
        if output_line == "":
            continue
        output_line_tokens = output_line.split(": ")
        scheme = output_line_tokens[0]
        results = [x.split('=') for x in output_line_tokens[1].split(' ')]
        results = {x[0]:x[1] for x in results}
        return (results['time'], results['result'])

    return None

'''
call cost *model* to estimate cost of measuring parameter
'''
def estimate_measurement_cost(bench_name, fixed_flag_settings, param_flag):
    flag_settings = fixed_flag_settings + (' -t %s' % param_flag)
    output = subprocess.check_output("PYTHONPATH=. python %s/measure_cost.py %s"
                                     % (bench_name, flag_settings),
                                     shell=True)
    parsed_output = parse_output(output)
    return parsed_output[0]

'''
run algorithm to pick which parameter to measure
'''
def pick_param(bench_name, known_param_settings, ranges):
    fixed_flag_settings = (' '.join(['-%s %s' % (x[0], str(x[1])) for x in known_param_settings]))

    est_ranges = itertools.product(*labeled_params(ranges))
    if len(ranges) == 0:
        est_ranges = [[]]
    
    costs = {}
    min_costs = []
    for r in est_ranges:
        ## run estimation in cost model ##
        flag_settings = (' '.join(['-%s %s' % (x[0], str(x[1])) for x in r]))
        flag_settings = flag_settings + " " + fixed_flag_settings
        output = subprocess.check_output("PYTHONPATH=. python %s/model_cost.py %s"
                                         % (bench_name, flag_settings),
                                         shell=True)
        parsed_output = parse_output(output)
        r_costs = []
        for (scheme, cost) in parsed_output:
            try:
                costs[scheme].append(cost)
            except:
                costs[scheme] = [cost]
            r_costs.append((cost, dict(r), scheme))

        min_costs.append(min(r_costs))

    averages = {scheme:np.mean(costs) for scheme, costs in costs.items()}
    default_plan, default_cost = min(averages.items(), key=itemgetter(1))
    
    gains = {}
    
    for p, p_values in ranges.items():
        ## get expectation of optimal over the remaining parameters ##
        p_slices = {x:{} for x in p_values}
        for (cost, params, scheme) in min_costs:
            try:
                p_slices[params[p]][scheme].append(cost)
            except:
                p_slices[params[p]][scheme] = [cost]

        ## inner expectation ##
        means = {x:{s:np.mean(c) for s, c in v.items()} for x, v in p_slices.items()}
        
        argmin_plans = {x:min(v.items(), key=itemgetter(1)) for x, v in means.items()}

        ## outer expectation ##
        costs = [v[1] for k, v in argmin_plans.items()]
        mean_cost = np.mean(costs)
        loss = default_cost - mean_cost
        
        ## get measurement costs ##
        label, M_cost = estimate_measurement_cost(bench_name, fixed_flag_settings, p)

        gain = loss - M_cost
        gains[p] = gain
    
    if len(gains) == 0:
        return (default_plan, None)
        
    max_gain = max([(v, k) for k, v in gains.items()])
    if max_gain[0] <= 0:
        return (default_plan, None)
    
    return (default_plan, max_gain[1])

'''
measure params until no gains are left
'''
def measurement_loop(bench_name, known_param_settings, true_params, range_params):
    ## copy values to be mutated ##
    known_settings = list(known_param_settings.items())
    ranges = copy(range_params)

    fixed_flag_settings = (' '.join(['-%s %s' % (x[0], str(x[1])) for x in known_settings]))

    plan, measure_param = pick_param(bench_name, known_settings, ranges)
    measured_values = {}
    while measure_param != None:
        flag_settings = fixed_flag_settings + ' -%s %s -t %s' % (measure_param,
                                                                 true_params[measure_param],
                                                                 measure_param)

        ## run the real measurement ##
        output = subprocess.check_output("%s/measure %s 2>/dev/null"
                                         % (bench_name, flag_settings),
                                         shell=True)
        
        time, estimated_value = parse_measurement(output)
        print ("Estimated value of %s: %s" % (measure_param, estimated_value))
        measured_values[measure_param] = time
        fixed_flag_settings += ' -%s %s' % (measure_param, estimated_value)
        known_settings.append((measure_param, estimated_value))
        del ranges[measure_param]
        plan, measure_param = pick_param(bench_name, known_settings, ranges)

    return (plan, measured_values)
    
def compile_benchmark(bench_name):
    subprocess.call("make -C %s" % bench_name, shell=True)
 
def expand_arange(params_dict):
    '''
    if params specified by a range, unroll the range into values
    this function uses np.arange instead of np.linspace to specify a step size
    '''
    ret = {}
    for p, v in params_dict.items():
        if isinstance(v, dict):
            ret[p] = np.arange(v['start'], v['stop'], v['step'])

            if v['type'] == 'int':
                ret[p] = [int(x) for x in ret[p]]

        elif isinstance(v, list):
            ret[p] = v
        else:
            raise ValueError
    return ret

# Run adaptive optimizer on passed in benchmark, with provided params
# and range params.
def run_adaptive(bench_name, params, range_params):
    range_params = expand_arange(range_params)

    known_params = {}
    true_params = {}
            
    for flag, val in params:
        if flag in range_params:
            true_params[flag] = val
        else:
            known_params[flag] = val
                    
    plan, measurements = measurement_loop(bench_name, known_params, true_params, range_params)

    return plan, measurements

range_params = {
    's': {
        "start":0.0,
        "stop":1.0,
        "step":0.1,
        "type":"float"
    }
}

# Run passed in benchmark, with provided params.
def run_benchmark(bench_name, params):
    flag_settings = (' '.join(['-%s %s' % (x[0], str(x[1])) for x in params]))
    subprocess.call('make -C %s' % bench_name, shell=True)
    output = subprocess.check_output('./%s/bench %s' % (bench_name, flag_settings),
                                     shell=True)
    times = parse_output(output)
    return {key: value for (key, value) in times}
