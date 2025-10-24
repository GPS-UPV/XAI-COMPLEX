##############################################
#                  LIBRERIAS                 #
##############################################
import array
import json
import os
import re
import time
from datetime import timedelta
from pprint import pprint

import numpy as np
from deap import base, creator
from IGJSP.generador import Generator
from minizinc import Instance, Model, Result, Solver, Status
from numpyencoder import NumpyEncoder

def compute_time_limit(nb_jobs: int, nb_machines: int, nb_speeds: int, time_min_ms: int = 400, time_max_ms: int = 30000) -> float:
    nb_jobs = int(nb_jobs)
    nb_machines = int(nb_machines)
    nb_speeds = max(1, int(nb_speeds))
    total = nb_jobs * nb_machines * nb_speeds
    if total <= 1:
        return int(max(1.0, time_min_ms / 1000.0))
    i = max(nb_jobs - 200, 0)
    j = max(nb_machines - 200, 0)
    k = max(nb_speeds - 1, 0)
    idx = i * (nb_machines * nb_speeds) + j * nb_speeds + k
    val = time_min_ms + (time_max_ms - time_min_ms) * (idx / (total - 1))
    return int(max(1.0, int(val) / 1000.0))

        if dim_str.isdigit():
            return int(dim_str)
        elif dim_str in definitions:
            return len(definitions[dim_str]) if isinstance(definitions[dim_str], range) else definitions[dim_str]
        else:
            try:
                return int(dim_str)
            except ValueError:
                raise ValueError(f"Dimensión '{dim_str}' no encontrada en las definiciones.")


class SOLVER:
    def __init__(self, problem, solver="cp-sat"):
        self.solver = solver
        self.problem = problem
        self.numJobs = self.problem.numJobs
        self.numMchs = self.problem.numMchs
        self.rddd = self.problem.rddd
        self.speed = self.problem.speed

        self.time = np.array(self.problem.ProcessingTime).flatten()
        self.energy = np.array(self.problem.EnergyConsumption).flatten()
        if problem.Orden.size == 0:
            self.precedence = np.array([list(range(self.numMchs))] * self.numJobs).flatten()
        else:
            self.precedence = problem.Orden.flatten()

        self.instance = []
        self.solution = []
        self.solution_resch = []
        self.model = []
    

    def solve(self, timeout=60, verbose=True, model_path="", path=""):
        
        self.model = Model(f"Minizinc/Models/RD/JSP{self.rddd}.mzn")
        self.model.add_file(path,parse_data=True)
        # self.model.add_string(f"JOBS = 1..{self.numJobs};\n")
        # self.model.add_string(f"MACHINES = 1..{self.numMchs};\n")
        # self.model.add_string(f"SPEED = {self.speed};\n")
        # self.model.add_string(f"time = array3d(JOBS,MACHINES,1..SPEED,[{','.join(map(str, self.time))}]);\n")
        # self.model.add_string(f"energy = array3d(JOBS,MACHINES,1..SPEED,[{','.join(map(str, self.energy))}]);\n")
        # self.model.add_string(f"precedence = array2d(JOBS,MACHINES,[{','.join(map(str, self.precedence))}]);\n" )
        # if verbose:
        #     print(f"JOBS = 1..{self.numJobs};\n")
        #     print(f"MACHINES = 1..{self.numMchs};\n")
        #     print(f"SPEED = {self.speed};\n")
        #     print(f"time = array3d(JOBS,MACHINES,1..SPEED,[{','.join(map(str, self.time))}]);\n")
        #     print(f"energy = array3d(JOBS,MACHINES,1..SPEED,[{','.join(map(str, self.energy))}]);\n")
        #     print(f"precedence = array2d(JOBS,MACHINES,[{','.join(map(str, self.precedence))}]);\n")

        if model_path != "":
            with open(model_path, "w", encoding="utf-8") as file:
                file.writelines(self.model.__dict__["_code_fragments"])
                
        self.instance = Instance(solver=Solver.lookup(self.solver), model=self.model)
        try:
            result = self.instance.solve(
                timeout =timedelta(milliseconds=timeout),
                free_search=True,
                processes=18,
            )
            if result:
                self.solution = result.__dict__
                if "status" in self.solution.keys():
                    self.solution["status"] = str(result.status)
                if "solution" in self.solution.keys():
                    self.solution["solution"] = result.solution.__dict__
                if "time" in self.solution["statistics"].keys():
                    self.solution["statistics"]["time"] = (self.solution["statistics"]["time"].total_seconds() *1000)
                if "optTime" in self.solution["statistics"].keys():
                    self.solution["statistics"]["optTime"] = (self.solution["statistics"]["optTime"].total_seconds() * 1000)
                if "flatTime" in self.solution["statistics"].keys():
                    self.solution["statistics"]["flatTime"] = (self.solution["statistics"]["flatTime"].total_seconds() * 1000)
                if "initTime" in self.solution["statistics"].keys():
                    self.solution["statistics"]["initTime"] = (self.solution["statistics"]["initTime"].total_seconds() * 1000)
                if "solveTime" in self.solution["statistics"].keys():
                    self.solution["statistics"]["solveTime"] = (self.solution["statistics"]["solveTime"].total_seconds() * 1000)
            else:
                self.solution = {"solution": None}
        except Exception as e:
            self.solution = {"solution": None}
            print(f"Error: {e}")
        if path != "":
            with open(f"{path}-{self.solver}.json", "w", encoding="utf-8") as f:
                json.dump(self.solution, f, indent=4,cls=NumpyEncoder)
        return self.solution
    