"""
Define metrics.
"""
import numpy as np
from collections import OrderedDict
from collections import Counter

# Fixed plan cost
fc = {
    #0: 0.0,
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

# Extra cost per MB
ec = {
    #0: 0.001*1024,
    2: 0.001*1024,
    5: 0.001*1024,
    10: 0.29,
    30: 0.29,
    100: 0.29,
    300: 0.29,
    500: 0.29,
    700: 0.29,
    1024: 0.29,
    2048: 0.29,
    3072: 0.29,
    4096: 0.29
}

plan_cost = OrderedDict(sorted(fc.items(), key=lambda t: t[0]))


def get_plan_fix_cost(plan):
    return fc[plan]


def compute_cost_with_plan_and_usage(plan, usage):
    """
    Given a plan and actual usage, returns cost of plan, cost with extra usage (when using more data than plan), and remaining quota (can be negative).
    :param plan: Monthly plan subscribed to.
    :param usage: actual usage, in MB.
    :return: a tuple of (plan_cost, extra_usage_cost, quota_left)
    """
    if usage <= plan:
        return plan_cost[plan], plan_cost[plan], plan-usage
    else:
        extra_used = usage - plan
        extra_cost = extra_used * ec[plan]
        return plan_cost[plan], plan_cost[plan] + extra_cost, plan-usage


def get_best_plan(usage):
    """
    Given usage, returns the best plan to subscribe.
    :param usage:
    :return:
    """
    plans_available = plan_cost.keys()
    cost_dict = dict()
    for p in plans_available:
        _, cost_dict[p], _ = compute_cost_with_plan_and_usage(plan=p, usage=usage)
    best_plan = min(cost_dict, key=cost_dict.get)
    return best_plan


def calculate_cost_with_data_pool(df, col_plan='plan_c', col_used='used', verbose=False):
    """

    :param df:
    :param col_plan:
    :param col_used:
    :param verbose:
    :return: total cost.
    """
    def get_plan_cost(plan, count):
        return fc[plan] * count

    def calculate_extra_cost(plan, used, pool_size):
        if used <= pool_size:
            return 0
        else:
            extra_used = used - pool_size
            return extra_used * ec[plan]

    if col_plan not in df.columns or col_used not in df.columns:
        raise("Plan or Usage column not found in data. col_plan={}, col_used={}, df columns={}".format(col_plan, col_used, df.columns))
    df_sel = df[[col_plan, col_used]]
    df_plan_usage = df_sel.groupby(col_plan)[col_used].agg(['count', 'sum']).reset_index().rename(columns={'sum': 'used'})  # total usage by plan
    df_plan_usage['pool_size'] = df_plan_usage['count'] * df_plan_usage[col_plan]
    df_plan_usage['remaining_quota'] = df_plan_usage['pool_size'] - df_plan_usage['used']
    df_plan_usage['pool_utilization'] = df_plan_usage['used']/df_plan_usage['pool_size']
    df_plan_usage['plan_cost'] = df_plan_usage.apply(lambda row: get_plan_cost(row[col_plan], row['count']), axis=1)
    df_plan_usage['extra_cost'] = df_plan_usage.apply(lambda row: calculate_extra_cost(row[col_plan], row[col_used], row['pool_size']), axis=1)
    df_plan_usage['total_cost'] = df_plan_usage['plan_cost'] + df_plan_usage['extra_cost']
    if verbose is True:
        print("Total devices: {}\nTotal cost: {}\nTotal usage:{}".format(sum(df_plan_usage['count']),
                                                                         sum(df_plan_usage['total_cost']),
                                                                         sum(df_plan_usage[col_used])))
        print(df_plan_usage)
    return sum(df_plan_usage['total_cost'])


def compare_plans(df, col_plan_x, col_plan_y, col_usage='used', index='iccid'):
    """
    Compare the total cost of 2 plans, df[col_plan_x] and df[col_plan_y], with usage given in column df[col_usage].
    :param df:
    :param col_plan_x:
    :param col_plan_y:
    :param col_usage:
    :param index: Unique ID of each device. Default: 'iccid'
    :return:
    """
    pass


def compute_best_global_plan_with_pool(df, col_id='iccid', col_usage='used', plans=fc):
    """
    Given usage of all devices individually, calculate the best global plan assignment strategy.
    Returns a copy of df with best plan
    :param df: DataFrame with id and usage of all devices.
    :param col_id:
    :param col_usage:
    :param plans: a Dict of plan: fixed cost {plan: cost}.
    :return:
    """
    def get_available_pool_size(dataframe, col_plan, col_used):
        """

        :param dataframe:
        :param col_plan:
        :param col_used:
        :return:
        """
        plan_cnt = dataframe.groupby(col_plan)[col_plan].count().to_dict()
        used_cnt = dataframe.groupby(col_plan)[col_used].sum().to_dict()
        pool_size = dict()
        # Initialize
        for k in fc.keys():
            pool_size[k] = 0
        # Update
        for k, v in plan_cnt.items():
            pool_size.update({k: k*v - used_cnt[k]})
        return pool_size

    # get best individual plan of each device, as heuristic and start position
    df['best_local_plan'] = df[col_usage].apply(get_best_plan)

    df_sorted = df.sort_values(by=[col_usage, 'best_local_plan'], ascending=True)  # sort devices by usage
    plans_sorted = sorted(plans.keys(), reverse=False)

    # assign best global plan to each device
    df['best_global_plan'] = df['best_local_plan']  # initialize
    avail_pool = get_available_pool_size(df_sorted, col_plan='best_local_plan', col_used='used')
    bins = iter(plans_sorted)
    cur_bin = next(bins)
    df_result = df_sorted.copy()
    for device in df_sorted.itertuples():
        if avail_pool[cur_bin] <= getattr(device, "used"):  # if current pool cannot accommodate this device...
            while getattr(device, "used") > cur_bin:  # ...try next bin, until there is a bin
                try:
                    cur_bin = next(bins)
                    avail_pool = get_available_pool_size(df_result, col_plan='best_global_plan', col_used='used')  # update pool
                except StopIteration:  # no more optimization
                    total_cost = calculate_cost_with_data_pool(df_result, col_plan='best_global_plan', col_used='used')
                    return df_result, total_cost
        if getattr(device, "best_local_plan") == cur_bin:  # this device is already in current bin. Move to next device
            continue
        # Change plan for this device
        df_result.set_value(getattr(device, col_id), 'best_global_plan', cur_bin)
        # update available data pool
        avail_pool = get_available_pool_size(df_result, col_plan='best_global_plan', col_used='used')
    total_cost = calculate_cost_with_data_pool(df_result, col_plan='best_global_plan', col_used='used')
    return df_result, total_cost


if __name__ == '__main__':
    for use in np.arange(0, 10, 0.1):
        p, c = get_best_plan(usage=use)
        print("Use: {}, Best plan:{}, Best cost:{}".format(use, p, c))
