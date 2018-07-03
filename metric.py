"""
Define metrics.
"""
import numpy as np
from collections import OrderedDict
dc = {
    0: 0.0,
    2: 1.0,
    5: 2.0,
    10: 3.0,
    30: 5.0,
    100: 8.0,
    300: 16.0,
    500: 24.0,
    700: 32.0,
    1024: 40.0,
    2048: 60.0,
    3072: 80.0,
    4096: 100.0
}
plan_cost = OrderedDict(sorted(dc.items(), key=lambda t: t[0]))


def get_plan_fix_cost(plan):
    return dc[plan]


def compute_cost_with_plan_and_usage(plan, usage):
    """
    Calculate data usage cost according to subscribed plan and actual data usage.
    :param plan: Monthly plan subscribed to.
    :param usage: actual usage, in MB.
    :return: billing cost.
    """
    if usage <= plan:
        return plan_cost[plan]
    else:
        extra_used = usage - plan
        if plan in [0, 2, 5]:
            extra_cost = extra_used * (0.001*1024)  # $0.001 per KB ~= $1 per MB
        else:
            extra_cost = extra_used * 0.29  # Assume no round-off
        return plan_cost[plan] + extra_cost


def get_best_plan_and_cost(usage, plan_used=None):
    """
    Given a usage, returns the best plan to subscribe and cost.
    If plan_used is given, also returns the cost using plan_used.
    :param usage:
    :param plan_used:
    :return:
    """
    plans_available = plan_cost.keys()
    cost_dict = dict()
    for p in plans_available:
        cost_dict[p] = compute_cost_with_plan_and_usage(plan=p, usage=usage)
    best_plan = min(cost_dict, key=cost_dict.get)
    best_cost = cost_dict[best_plan]
    if plan_used is None:
        return best_plan, best_cost
    else:
        return best_plan, best_cost, cost_dict[plan_used]


if __name__ == '__main__':
    for use in np.arange(0, 10, 0.1):
        p, c = get_best_plan_and_cost(usage=use)
        print("Use: {}, Best plan:{}, Best cost:{}".format(use, p, c))
