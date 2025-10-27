import json
import os
import re
from datetime import timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

from minizinc import Instance, Model, Solver
from solvers import compute_time_limit

# ----------------------------------------------------------
# Configuración general
# ----------------------------------------------------------
DZN_DIR = Path("./instances")
MODEL_DIR = Path("./Minizinc/Models/RD/")
OUT_DIR = Path("./solutions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOLVER_NAME = "cp-sat"
VERBOSE = True

NAME_RE = re.compile(r"^(?P<jobs>\d+)_(?P<mchs>\d+)-(?P<typ>\d+)-(?P<spd>\d+)\.dzn$", re.IGNORECASE)

def parse_dzn_name(filename: str):
    m = NAME_RE.match(filename)
    if not m:
        raise ValueError(f"Nombre de instancia no válido: {filename}")
    return (
        int(m.group("jobs")),
        int(m.group("mchs")),
        int(m.group("typ")),
        int(m.group("spd")),
    )

def main():
    dzn_files = sorted(DZN_DIR.glob("*.dzn"))
    pbar = tqdm(dzn_files)
    if not dzn_files:
        print(f"[WARN] No se encontraron archivos .dzn en {DZN_DIR}")
        return

    pbar.set_description(f"[INFO] Se encontraron {len(dzn_files)} instancias en {DZN_DIR}")
    max_job,max_machine = np.array([list(map(int,x.stem.split("-")[0].split("_"))) for x in dzn_files]).max(axis=0)
    
    ok, fail = 0, 0
    for dzn_path in pbar:
        try:
            out_file = OUT_DIR / f"{dzn_path.stem}-{SOLVER_NAME}.json"
            if os.path.exists(out_file):
                pbar.set_description(f"⚠️  Saltando {dzn_path.name}, ya existe solución en {out_file.name}")
                continue
            jobs, mchs, typ, spd = parse_dzn_name(dzn_path.name)
            timeout_ms = compute_time_limit(jobs, mchs, spd,max_job,max_machine)

            model_path = MODEL_DIR / f"JSP{typ}.mzn"
            if not model_path.exists():
                model_path = MODEL_DIR / "JSP.mzn"
                if not model_path.exists():
                    raise FileNotFoundError(f"No se encontró modelo MiniZinc para tipo {typ}")

            pbar.set_description(f"🧩 Ejecutando {dzn_path.name} Jobs={jobs}, Mchs={mchs}, Type={typ}, Speed={spd}, Timeout={timeout_ms} ms")

            model = Model(model_path)
            model.add_file(dzn_path)
            instance = Instance(solver=Solver.lookup(SOLVER_NAME), model=model)
            result = instance.solve(
                timeout = timedelta(milliseconds=timeout_ms),
                free_search = True,
                processes = 10,
            )

            if result:
                solution = result.__dict__
                
                solution["statistics"]["maxTime"] = timeout_ms
                if "status" in solution.keys():
                    solution["status"] = str(result.status)
                if "solution" in solution.keys():
                    solution["solution"] = result.solution.__dict__
                    values = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", solution["solution"]["_output_item"])][:2]
                    solution["statistics"]["makespan"] = values[0]
                    solution["statistics"]["energy"] = values[1]
                if "time" in solution["statistics"].keys():
                    solution["statistics"]["time"] = (solution["statistics"]["time"].total_seconds() *1000)
                if "optTime" in solution["statistics"].keys():
                    solution["statistics"]["optTime"] = (solution["statistics"]["optTime"].total_seconds() * 1000)
                if "flatTime" in solution["statistics"].keys():
                    solution["statistics"]["flatTime"] = (solution["statistics"]["flatTime"].total_seconds() * 1000)
                if "initTime" in solution["statistics"].keys():
                    solution["statistics"]["initTime"] = (solution["statistics"]["initTime"].total_seconds() * 1000)
                if "solveTime" in solution["statistics"].keys():
                    solution["statistics"]["solveTime"] = (solution["statistics"]["solveTime"].total_seconds() * 1000)
            else:
                solution = {"solution": None}

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(solution, f, indent=2, ensure_ascii=False)

            status = solution.get("status", "UNKNOWN")
            pbar.set_description(f"  ✅ Finalizado - status={status}")
            ok += 1

        except Exception as e:
            print(f"❌ Error en {dzn_path.name}: {e}")
            fail += 1

    print(f"\n[RESUMEN] Ejecuciones correctas: {ok} | Fallos: {fail}")
    print(f"Resultados guardados en {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
