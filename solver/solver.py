from pulp import LpMaximize, LpProblem, LpStatus, LpVariable
import numpy as np


class XVar:
    def __init__(self, field_name: str = None, val_prod: float = None):
        self.field_name = field_name
        self.val_prod = val_prod
        self.weight = float(1)

    def set_weight(self, r_model):
        for var in r_model.variables():
            if var.name == self.field_name:
                self.weight = var.value()


class TestConstraint:
    def __init__(self, name: str = None, x_values: list = None, y_value: float = None, operator: str = None):
        self.name = name
        self.x_values = x_values
        self.operator = operator
        self.y_value = y_value

    def get_dot_product_constraint_lp(self, l_var):
        return np.dot(l_var, self.x_values) <= self.y_value, self.name


def obj_function(l_var, l_field_val):
    return np.dot(l_var, l_field_val)


def solve(model_name, sense, x_vars, l_constraint, obj_function_param):
    model_lp = LpProblem(name=model_name, sense=sense)

    l_var = []
    l_val = []
    for idx, each in enumerate(x_vars):
        obj: XVar = each
        l_var.append(LpVariable(name=obj.field_name))
        l_val.append(obj.val_prod)

    for each_constraint in l_constraint:
        model_lp += each_constraint.get_dot_product_constraint_lp(l_var)

    # Objective Function
    model_lp += obj_function_param(l_var, l_val)
    model_lp.solve()

    for each_field in x_vars:
        each_field.set_swing_factor(model_lp)

    return x_vars


if __name__ == '__main__':
    result_node = solve(
        model_name='Total Profit',
        sense=LpMaximize,
        x_vars=[
            XVar(field_name=str(1), val_prod=450),
            XVar(field_name=str(2), val_prod=1150),
            XVar(field_name=str(3), val_prod=800),
            XVar(field_name=str(5), val_prod=400),
        ],
        l_constraint=[
            TestConstraint(name='glue', x_values=[50, 50, 100, 50], y_value=5800),
            TestConstraint(name='pressing', x_values=[5, 15, 10, 5], y_value=730),
            TestConstraint(name='pine chips', x_values=[500, 400, 300, 200], y_value=29200),
            TestConstraint(name='oak chips', x_values=[500, 750, 250, 500], y_value=60500),
        ],
        obj_function_param=obj_function
    )

    # print(f"status: {model.status}, {LpStatus[model.status]}")
    # print(f"objective: {model.objective.value()}")
    # for var in r_model.variables():
    #     print(f"{var.name}: {var.value()}")
    # for name, constraint in model.constraints.items():
    #     print(f"{name}: {constraint.value()}")

    for var in result_node:
        print(f"{var.field_name}: {var.swing_factor}")
