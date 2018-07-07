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
    Given a plan and actual usage, returns cost of plan, cost with extra usage (when using more data than plan),
    and remaining quota (can be negative).
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
        #print("Plan cnt:{}".format(plan_cnt))
        #print("used_cnt:{}".format(used_cnt))
        pool_size = dict()
        # Initialize
        for k in fc.keys():
            if k in plan_cnt.keys():
                pool_size.update({k: k * plan_cnt[k] - used_cnt[k]})
            else:
                pool_size[k] = 0
        #print("Pool size: {}".format(pool_size))
        return pool_size

    # get best individual plan of each device, as heuristic and start position
    df['best_local_plan'] = df[col_usage].apply(get_best_plan)

    df_sorted = df.sort_values(by=[col_usage, 'best_local_plan'], ascending=True)  # sort devices by usage
    plans_sorted = sorted(plans.keys(), reverse=False)
    print("Available plans: {}".format(plans_sorted))

    # assign best global plan to each device
    avail_pool = get_available_pool_size(df_sorted, col_plan='best_local_plan', col_used='used')
    bins = iter(plans_sorted)
    cur_bin = next(bins)
    df_result = df_sorted.copy()
    df_result['best_global_plan'] = df_result['best_local_plan']  # initialize

    #print(df_sorted.head())
    #print(df_result.head())

    print("Starting assigning plans to devices...")
    cur_bin_size = 0.0
    for cnt, device in enumerate(df_sorted.itertuples()):
        if cnt % 1000 == 0:
            print("Processing device #{}".format(cnt))
        this_device_contrib_to_bin = cur_bin - getattr(device, col_usage)
        #print(cur_bin_size, cur_bin, getattr(device, col_usage), this_device_contrib_to_bin)
        while (cur_bin_size < 0) or (cur_bin_size < -1*this_device_contrib_to_bin):  # Current bin cannot accommodate
            print("Pool of size {} MB is full".format(cur_bin))
            try:
                cur_bin = next(bins)
                print('Trying pool of {} MB...'.format(cur_bin))
                avail_pool = get_available_pool_size(df_result, col_plan='best_global_plan',
                                                     col_used=col_usage)  # update pool
                cur_bin_size = avail_pool[cur_bin]
                this_device_contrib_to_bin = cur_bin - getattr(device, col_usage)
                print("Current pool size: {}".format(avail_pool))
            except StopIteration:  # no more optimization
                print("No more pools can be filled, compute total cost and return.")
                total_cost = calculate_cost_with_data_pool(df_result, col_plan='best_global_plan', col_used=col_usage)
                return df_result, total_cost

        # Assign bin
        if getattr(device, 'best_local_plan') == cur_bin:
            pass
        else:
            df_result.at[device.Index, 'best_global_plan'] = cur_bin
        cur_bin_size += this_device_contrib_to_bin  # update
    print("All devices processed.")
    avail_pool = get_available_pool_size(df_result, col_plan='best_global_plan', col_used=col_usage)  # update pool
    total_cost = calculate_cost_with_data_pool(df_result, col_plan='best_global_plan', col_used=col_usage)
    return df_result, total_cost, avail_pool


if __name__ == '__main__':
    for use in np.arange(0, 10, 0.1):
        p, c = get_best_plan(usage=use)
        print("Use: {}, Best plan:{}, Best cost:{}".format(use, p, c))
