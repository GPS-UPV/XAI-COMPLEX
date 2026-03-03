import os
from IGJSP.generador import JSP

def main():
    
    files = [os.path.join("./TaillardInstances", f) for f in os.listdir("./TaillardInstances")]
    
    for f in files:
        a = JSP.loadTaillardFile(f)
        a.generate_maxmin_objective_values()
        a.EnergyConsumption = a.ProcessingTime
        json_file = f.replace(".txt", ".json").split("\\")[-1]
        a.saveJsonFile(os.path.join("./TaillardInstancesJSON", json_file))
    
if __name__ == "__main__":
    main()