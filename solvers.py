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
from numpyencoder import NumpyEncoder

from minizinc import Instance, Model, Result, Solver, Status



def compute_time_limit(
    nb_jobs: int,
    nb_machines: int,
    nb_speeds: int,
    max_job: int = 200,
    max_machine: int = 200,
    time_min_ms: int = 400,
    time_max_ms: int = 300000
) -> int:

    nb_jobs = max(1, nb_jobs)
    nb_machines = max(1, nb_machines)
    nb_speeds = max(1, nb_speeds)
    max_job = max(1, max_job)
    max_machine = max(1, max_machine)
    
    norm_jobs = min(nb_jobs / max_job, 1.0)
    norm_mchs = min(nb_machines / max_machine, 1.0)
    
    size_factor = (norm_jobs + norm_mchs) / 2.0
    
    val = time_min_ms + (time_max_ms - time_min_ms) * size_factor

    val *= nb_speeds ** 0.5

    return int(val)



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
    

    def solve(self, timeout=60000, verbose=True, model_path="", path="", processes=18):
        self.model = Model(f"Minizinc\Models\RD\JSP{self.rddd}.mzn") #.\Minizinc\Models\RD\JSP0.mzn

        if path:
            # modo fichero .dzn
            self.model.add_file(path, parse_data=True)
        else:
            # modo en memoria
            self.model.add_string(f"JOBS = 1..{self.numJobs};\n")
            self.model.add_string(f"MACHINES = 1..{self.numMchs};\n")
            self.model.add_string(f"SPEED = {self.speed};\n")
            self.model.add_string(
                f"time = array3d(JOBS,MACHINES,1..SPEED,[{','.join(map(str, self.time))}]);\n"
            )
            self.model.add_string(
                f"energy = array3d(JOBS,MACHINES,1..SPEED,[{','.join(map(str, self.energy))}]);\n"
            )
            self.model.add_string(
                f"precedence = array2d(JOBS,MACHINES,[{','.join(map(str, self.precedence))}]);\n"
            )

        if verbose:
            print(f"Model: Minizinc/JSP{self.rddd}.mzn")
            print(f"Jobs={self.numJobs}, Machines={self.numMchs}, Speed={self.speed}")

        if model_path:
            with open(model_path, "w", encoding="utf-8") as file:
                file.writelines(self.model.__dict__["_code_fragments"])

        self.instance = Instance(solver=Solver.lookup(self.solver), model=self.model)

        try:
            result = self.instance.solve(
                timeout=timedelta(milliseconds=timeout),
                free_search=True,
                processes=processes,
            )

            if result:
                self.solution = result.__dict__

                if "status" in self.solution:
                    self.solution["status"] = str(result.status)

                if "solution" in self.solution:
                    self.solution["solution"] = result.solution.__dict__

                if "time" in self.solution["statistics"]:
                    self.solution["statistics"]["time"] = (
                        self.solution["statistics"]["time"].total_seconds() * 1000
                    )
                if "optTime" in self.solution["statistics"]:
                    self.solution["statistics"]["optTime"] = (
                        self.solution["statistics"]["optTime"].total_seconds() * 1000
                    )
                if "flatTime" in self.solution["statistics"]:
                    self.solution["statistics"]["flatTime"] = (
                        self.solution["statistics"]["flatTime"].total_seconds() * 1000
                    )
                if "initTime" in self.solution["statistics"]:
                    self.solution["statistics"]["initTime"] = (
                        self.solution["statistics"]["initTime"].total_seconds() * 1000
                    )
                if "solveTime" in self.solution["statistics"]:
                    self.solution["statistics"]["solveTime"] = (
                        self.solution["statistics"]["solveTime"].total_seconds() * 1000
                    )
            else:
                self.solution = {"solution": None}

        except Exception as e:
            self.solution = {"solution": None, "error": str(e)}
            print(f"Error: {e}")

        return self.solution