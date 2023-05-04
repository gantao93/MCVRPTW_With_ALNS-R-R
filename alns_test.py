import copy
from dataclasses import dataclass
from typing import List,Any,Dict,Optional

import cvrplib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pandas as pd
from haversine import haversine
from alns import ALNS, State
from alns.accept import *
from alns.select import *
from alns.stop import *

from read_data import read_vrp_data


SEED = 1234
speed = 60/3600


#-----------------
# Initialize Data
#-----------------
vrp_data,vehicle_data = read_vrp_data(batch=6)
customers = list(vrp_data["收货地编码"])
num_customers = len(customers)
corr_xy = [(list(vrp_data["纬度"])[i],list(vrp_data["经度"])[i]) for i in range(num_customers)]
demands = list(vrp_data["订单总体积"])
start_time = list(vrp_data["开始时间"])
end_time = list(vrp_data["结束时间"])
serve_time = list(vrp_data["服务时间"])

vehicles = list(vehicle_data["车牌号"])
num_vehicle = len(list(vehicle_data["车牌号"]))
capacitys_type = list(vehicle_data["车厢体积"].unique())
capacitys = list(map(lambda x: x * 0.9, list(vehicle_data["车厢体积"])))
vehicle_lens_capacity_mapping = {
    "4.2":[12,13,14,14.4,15.23,15.45,15.53,15.55,15.96,16,16.01,16.2,16.22,16.35,16.55,16.88,16,18,20,22],
    "3":[],
    "2.8":[]
}

def get_distance():
    # 计算距离矩阵
    distance_matrix = np.zeros((num_customers, num_customers))
    for i in range(num_customers):
        for j in range(i + 1, num_customers):
            dis = haversine(corr_xy[i], corr_xy[j])
            distance_matrix[i][j] = distance_matrix[j][i] = dis
    return distance_matrix

distance = get_distance()

# ------------------------
# Solution state
# ------------------------
class CvrpState(State):
    """
    Solution state for CVRP. It has two data members, routes and unassigned.
    Routes is a list of list of integers, where each inner list corresponds to
    a single route denoting the sequence of customers to be visited. A route
    does not contain the start and end depot. Unassigned is a list of integers,
    each integer representing an unassigned customer.
    """

    def __init__(self, routes, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []

    def copy(self):
        return CvrpState(copy.deepcopy(self.routes), self.unassigned.copy())

    def objective(self):
        """
        Computes the total route costs.
        """
        #TODO 最小车辆数
        return sum(route_cost(route) for route in self.routes)

    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective()

    def find_route(self, customer):
        """
        Return the route that contains the passed-in customer.
        """
        for route in self.routes:
            if customer in route:
                return route

        raise ValueError(f"Solution does not contain customer {customer}.")


def route_cost(route):
    tour = [0] + route + [0]
    return sum(distance[tour[idx]][tour[idx + 1]] for idx in range(len(tour) - 1))


# ------------------------
# Repair operators
# ------------------------
def greedy_repair(state, rnd_state, vehicle_id):
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created.
    """
    rnd_state.shuffle(state.unassigned)

    while len(state.unassigned) != 0:
        customer = state.unassigned.pop()
        route, idx = best_insert(customer, state, vehicle_id)

        if route is not None:
            route.insert(idx, customer)
        else:
            state.routes.append([customer])

    return state


def best_insert(customer, state, vehicle_id):
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    """
    best_cost, best_route, best_idx = None, None, None

    for route in state.routes:
        for idx in range(len(route) + 1):

            if can_insert(customer, route, vehicle_id):
                cost = insert_cost(customer, route, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx


# TODO capacity 不同， 其他約束也應該在這裏考慮
def can_insert(customer, route, vehicle_id):
    """
    Checks if inserting customer does not exceed vehicle capacity.
    """
    # TODO 需求容量
    total = sum(demands[cust] for cust in route) + demands[customer]
    return total <= capacitys[vehicle_id]


def insert_cost(customer, route, idx):
    """
    Computes the insertion cost for inserting customer in route at idx.
    """
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]

    # Increase in cost by adding the customer
    cost = distance[pred][customer] + distance[customer][succ]
    cost -= distance[pred][succ]
    return cost


def repair(state, rnd_state):
    """
    Applies a set of repair operators to the solution state until all
    constraints are satisfied.
    """
    for vehicle_type in range(num_vehicle):
        state = greedy_repair(state, rnd_state, vehicle_type)
        # state = intra_relocate(state)#, rnd_state, vehicle_type)
        # state = inter_relocate(state)#, rnd_state, vehicle_type)
        # state = exchange(state)#, rnd_state, vehicle_type)

    return state


def intra_relocate(state: CvrpState) -> CvrpState:
    """
    Perform intra-route relocation operator. This operator removes a customer from one route and inserts it
    into another position in the same route.
    """
    # 随机选择一个路线
    route_idx = np.random.choice(len(state.routes))
    route = state.routes[route_idx]

    # 确保至少有两个顾客
    if len(route) < 3:
        return state

    # 随机选择一个顾客
    customer_idx = np.random.choice(range(1, len(route) - 1))
    customer = route[customer_idx]

    # 随机选择插入的位置
    insert_idx = np.random.choice(range(1, len(route)))

    # 如果插入位置和顾客所在位置相同，则不进行操作
    if insert_idx == customer_idx:
        return state

    # 计算插入后新的路线
    new_route = route[:customer_idx] + route[customer_idx + 1:]
    new_route = new_route[:insert_idx] + [customer] + new_route[insert_idx:]

    # 计算修改后的代价
    old_cost = sum(route_cost(_route) for _route in route)
    new_cost = sum(route_cost(_new_route) for _new_route in new_route)

    # 如果修改后的代价更优，则接受修改
    if new_cost < old_cost:
        state.routes[route_idx] = new_route

    return state


def inter_relocate(state: CvrpState) -> CvrpState:
    """
    Perform inter-route relocation operator. This operator removes a customer from one route and inserts it
    into another route at a different position.
    """
    # 确保有至少两条路线
    if len(state.routes) < 2:
        return state

    # 随机选择两条不同的路线
    route_idxs = np.random.choice(len(state.routes), size=2, replace=False)
    route1, route2 = state.routes[route_idxs[0]], state.routes[route_idxs[1]]
    # 确保两条路线都至少有一个顾客
    if len(route1) < 2 or len(route2) < 2:
        return state

    # 随机选择一个顾客

    customer_idx = np.random.choice(range(1, len(route1) - 1))
    customer = route1[customer_idx]

    # 随机选择插入的位置
    insert_idx = np.random.choice(range(1, len(route2)))

    # 计算插入后新的路线
    new_route1 = route1[:customer_idx] + route1[customer_idx + 1:]
    new_route2 = route2[:insert_idx] + [customer] + route2[insert_idx:]

    # 计算修改后的代价
    old_cost = sum(route_cost(_route1) for _route1 in route1) + sum(route_cost(_route2) for _route2 in route2)
    new_cost = sum(route_cost(_new_route1) for _new_route1 in new_route1)+sum(route_cost(_new_route2) for _new_route2 in new_route2)

    if new_cost < old_cost:
        state.routes[route_idxs[0]] = new_route1
        state.routes[route_idxs[1]] = new_route2

    return state

def exchange(state):
    """
    Exchange the customers between two positions in two different routes.
    """
    # Check if the exchange is valid
    # 随机选择两条不同的路线
    route_idxs = np.random.choice(len(state.routes), size=2, replace=False)
    route1, route2 = state.routes[route_idxs[0]], state.routes[route_idxs[1]]
    pos1 = np.random.choice(range(0, len(route1)))
    pos2 = np.random.choice(range(0, len(route2)))

    if route1 == route2 or pos1 == 0 or pos2 == 0:
        return None

    route1_demand = sum(demands[i] for i in route1)
    route2_demand = sum(demands[i] for i in route2)

    if route1_demand - demands[route1[pos1]] + demands[route2[pos2]] > capacitys[route_idxs[0]] or \
            route2_demand - demands[route2[pos2]] + demands[route1[pos1]] > capacitys[route_idxs[1]]:
        return None

    # Perform the exchange
    state.routes[route_idxs[0]][pos1], state.routes[route_idxs[1]][pos2] = state.routes[route_idxs[1]][pos2], state.routes[route_idxs[0]][pos1]
    return state


# ------------------------
# Initial solution
# ------------------------
# demands_all = demands
#
# # # 简单初始化解
# def initial_solution():
#     # 根据每个客户和depot之间的距离排序客户列表
#     # sorted_customers = sorted(problem.customers, key=lambda c: problem.distance(c.location, problem.customers[0].location))
#     dis_order = [haversine(corr_xy[0], corr_xy[i]) for i in range(1,num_customers)]
#     cus_loc_mapping = dict(zip(list(range(1,num_customers)), dis_order))
#
#     def sort_dict_by_value(d):
#         sorted_dict = {}
#         sorted_keys = sorted(d, key=d.get)
#         for key in sorted_keys:
#             sorted_dict[key] = d[key]
#         return sorted_dict
#
#     customers_order = list(sort_dict_by_value(cus_loc_mapping).keys())
#
#     sorted_customers = [0]+customers_order
#
#     # 按需求分类
#     class_demands = [[] for _ in range(num_vehicle)]
#     for cus_id in sorted_customers[1:]:  # 跳过depot
#         for i in range(num_vehicle):
#             if demands_all[cus_id] <= capacitys[i]:
#                 print(demands_all[cus_id],capacitys[i])
#                 class_demands[i].append(cus_id)
#                 break
#         else:
#             class_demands[0].append(cus_id)  # 如果没有合适的车辆，则将其分配给第一个车辆
#
#     print("class_demands",class_demands)
#     # 为每个车辆分配路径
#     solution = [[] for _ in range(num_vehicle)]
#     for vehicle_idx, demands in enumerate(class_demands):
#         route = []
#         capacity_left = capacitys[vehicle_idx]
#         time_left = 0
#         depot_loc = corr_xy[0]#0  # 初始位置为depot
#         # for _demand in demands:
#         for inx in range(1,len(demands)):
#             demand_loc = corr_xy[inx]
#             distance_to_demand = haversine(depot_loc, demand_loc)
#             distance_from_demand = haversine(depot_loc, demand_loc)  # 回到depot的距离
#             time_to_demand = time_left + distance_to_demand
#             start_tw, end_tw = start_time[inx],end_time[inx]
#             if time_to_demand + serve_time[inx] <= end_tw and capacity_left >= demands[inx]:
#                 # 可以满足要求，添加到路径中
#                 route.append(inx)
#                 capacity_left -= demands[inx]
#                 time_left += distance_to_demand + serve_time[inx]
#                 location = demand_loc
#             else:
#                 # 无法满足要求，结束路径
#                 solution[vehicle_idx].append(route)# + [0])  # 添加回到depot的路径
#                 route = []
#                 capacity_left = capacitys[vehicle_idx]
#                 time_left = 0
#                 location = 0
#         if route:
#             # 添加最后一个路径
#             solution[vehicle_idx].append(route) #+ [0])
#
#     # 删除空路径
#     solution = [s for s in solution if s]
#     print(solution)
#     return solution

def neighbors(customer):
    """
    Return the nearest neighbors of the customer, excluding the depot.
    """
    locations = np.argsort(distance[customer])
    return locations[locations != 0]


def nearest_neighbor():
    """
    Build a solution by iteratively constructing routes, where the nearest
    customer is added until the route has met the vehicle capacity and TW limit.
    """

    routes = []
    unvisited = list(range(1, num_customers))
    time_remaining = [0] * num_customers

    while unvisited:
        route = [0]  # Start at the depot
        route_demands = 0
        vehicle_id = 0
        time_elapsed = start_time[0]  # depot开始时间


        while unvisited:
            # Add the nearest unvisited customer to the route till max capacity
            current = route[-1]
            nearest = [nb for nb in neighbors(current) if nb in unvisited][0]

            # ------------------
            # 容 量 约 束
            # ------------------
            # Check if adding the nearest customer violates the capacity constraint of the current vehicle
            if route_demands + demands[nearest] > capacitys[vehicle_id]:
                # If it does, try the next vehicle
                vehicle_id += 1
                if vehicle_id >= num_vehicle:
                    break
                continue

            # # ------------------
            # # 时 间 窗 约 束
            # # ------------------
            if capacitys[vehicle_id] in vehicle_lens_capacity_mapping.get("4.2"):
                time_elapsed += 1.5 * 60 * 60
            else:
                time_elapsed += 1.0 * 60 * 60

            if time_elapsed + distance[current][nearest] / speed + serve_time[nearest] > end_time[nearest]:
                break
            if time_elapsed + distance[current][nearest] / speed + serve_time[nearest] < start_time[nearest]:
                time_elapsed = start_time[nearest]

            route.append(nearest)
            unvisited.remove(nearest)
            route_demands += demands[nearest]
            time_elapsed += distance[current][nearest] / speed + serve_time[nearest]
            time_remaining[nearest] = end_time[nearest] - time_elapsed


        customers = route[1:]  # Remove the depot
        print("customers",customers)
        routes.append(customers)
    print("routes:>>>>>>>>>>",routes) #初始解

    return CvrpState(routes)

# ------------------------
# Slack-induced substring removal
# ------------------------
MAX_STRING_REMOVALS = 3
MAX_STRING_SIZE = 10

def string_removal(state, rnd_state):
    """
    Remove partial routes around a randomly chosen customer.
    """
    destroyed = state.copy()

    avg_route_size = int(np.mean([len(route) for route in state.routes]))
    max_string_size = max(MAX_STRING_SIZE, avg_route_size)
    max_string_removals = min(len(state.routes), MAX_STRING_REMOVALS)

    destroyed_routes = []
    center = rnd_state.randint(1, num_customers) # TODO

    for customer in neighbors(center):
        if len(destroyed_routes) >= max_string_removals:
            break

        if customer in destroyed.unassigned:
            continue

        route = destroyed.find_route(customer)
        if route in destroyed_routes:
            continue

        customers = remove_string(route, customer, max_string_size, rnd_state)
        destroyed.unassigned.extend(customers)
        destroyed_routes.append(route)

    return destroyed


def remove_string(route, cust, max_string_size, rnd_state):
    """
    Remove a string that constains the passed-in customer.
    """
    # Find consecutive indices to remove that contain the customer
    size = rnd_state.randint(1, min(len(route), max_string_size) + 1)
    start = route.index(cust) - rnd_state.randint(size)
    idcs = [idx % len(route) for idx in range(start, start + size)]

    # Remove indices in descending order
    removed_customers = []
    for idx in sorted(idcs, reverse=True):
        removed_customers.append(route.pop(idx))

    return removed_customers


# ------------------------
# Heuristic solution
# ------------------------
alns = ALNS(rnd.RandomState(SEED))
alns.add_destroy_operator(string_removal)
#alns.add_repair_operator(greedy_repair)
alns.add_repair_operator(repair)

init = nearest_neighbor()
select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
accept = RecordToRecordTravel.autofit(init.objective(), 0.02, 0, 6000)
stop = MaxRuntime(60*15)

result = alns.iterate(init, select, accept, stop)

solution = result.best_state
objective = solution.objective()

print(f"Customer numer is {num_customers}")
print(f"Best heuristic objective is {objective}.")
print(f"solution: {solution.routes}")

routes_solutions = [i for i in solution.routes if len(i) != 0]
print(f"solution: {routes_solutions}")
# 计算每条route的容量
routes_loads = [sum(demands[cus] for cus in route) for route in routes_solutions]
print("routes_loads",routes_loads)
# 为每条route的loads匹配最佳车型，准则:对于车而言在容量内装得越多越好
def find_match_vehicle_type(route_load,capacitys_type):
    best_vehicle_type = 10000
    for ty in capacitys_type:
        if ty >= route_load:
            best_vehicle_type = min(best_vehicle_type,ty-route_load)
    return best_vehicle_type+route_load

routes_vehicle_type = [find_match_vehicle_type(route_load,capacitys_type) for route_load in routes_loads]
print(f"routes_type: {routes_vehicle_type}")

# 随机选择车id,车型数量多的话用更大车型匹配
from collections import Counter
print(f"所有车型统计数量:{Counter(routes_vehicle_type)}")

print(f"验证所有节点是否都包含{len(sum(solution.routes,[]))} ,{num_customers}")
print(f"最小车辆数: {len(routes_solutions)}")

