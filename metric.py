"""
Define metrics.
"""
import numpy as np
from collections import OrderedDict
dc = {
    2: 1.0,
    5: 2.0,
    10: 3.0,
    30: 5.0,
    100: 8.0,
    300: 16.0
}
plan_cost = OrderedDict(sorted(dc.items(), key=lambda t: t[0]))


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
        if plan in [2, 5]:
            extra_cost = extra_used * (0.001*1024)  # $0.001 per KB ~= $1 per MB
        else:
            extra_cost = extra_used * 0.29  # Assume no round-off
        return plan_cost[plan] + extra_cost


def get_best_plan_and_cost(usage, plan_used=None):
    """
    Given a usage, returns the best plan to subscribe and cost.
    If a plan is given, also returns the cost difference between best plan and plan_used.
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
        return best_plan, best_cost, cost_dict[plan_used] - best_cost


if __name__ == '__main__':
    for use in np.arange(0, 10, 0.1):
        p, c = get_best_plan_and_cost(usage=use)
        print("Use: {}, Best plan:{}, Best cost:{}".format(use, p, c))
