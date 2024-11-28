from .rand_selector import RandSelector,FarthestPointSelector
from .H_reg import HRegSelector,VIMCSelector,KSMetricSelector

methods_dict = {"rand": RandSelector, "H_reg": HRegSelector,
                "VI_MC": VIMCSelector, "FP": FarthestPointSelector, "KS": KSMetricSelector}