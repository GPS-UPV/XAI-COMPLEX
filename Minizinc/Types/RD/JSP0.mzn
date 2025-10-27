% Conjuntos
set of int: JOBS;
set of int: MACHINES;
int: SPEED;

% Datos
array[JOBS,MACHINES,1..SPEED] of int: time;
array[JOBS,MACHINES,1..SPEED] of int: energy;
array[JOBS,MACHINES] of int: precedence;

%%% Normalizaciones
% Max makespan
int: maxMakespan = sum([max([time[j,m,s] | s in 1..SPEED]) | j in JOBS, m in MACHINES]);
% Min makespan
int: minMakespan = max([sum([min([time[j,m,s] | s in 1..SPEED]) | m in MACHINES]) | j in JOBS]);
% Max energy
int: maxEnergy = sum([max([energy[j,m,s] | s in 1..SPEED]) | j in JOBS, m in MACHINES]);
% Min energy
int: minEnergy = sum([min([energy[j,m,s] | s in 1..SPEED]) | j in JOBS, m in MACHINES]);

% Variables
array[JOBS,MACHINES] of var 0..sum(time): start_time;
array[JOBS,MACHINES] of var 1..SPEED: SpeedScaling;

%Restriccion de precedencia
constraint forall(j in JOBS, p in MACHINES, s in MACHINES where p < s)(
  if precedence[j,p] > precedence[j,s] then
      start_time[j,p] >= start_time[j,s] + time[j,s,SpeedScaling[j,s]]
  else
      start_time[j,s] >= start_time[j,p] + time[j,p,SpeedScaling[j,p]]
  endif
);

% Realizacion de trabajos a la vez
constraint forall(j1 in JOBS, j2 in JOBS, m in MACHINES where j1 < j2)(
  start_time[j1,m] + time[j1,m,SpeedScaling[j1,m]] <= start_time[j2,m] \/ start_time[j2,m] + time[j2,m,SpeedScaling[j2,m]] <= start_time[j1,m]
);

% Variable para el makespan (tiempo total de finalización)
var float: makespan                     = max([start_time[j,m] + time[j,m,SpeedScaling[j,m]] | m in MACHINES, j in JOBS]);
var float: consumedEnergy               = sum(m in MACHINES, j in JOBS) (energy[j,m,SpeedScaling[j,m]]);
var float: normalized_makespan          = (makespan - minMakespan) / (maxMakespan - minMakespan);
var 0.0..1.0: normalized_consumedEnergy = if maxEnergy > minEnergy then ((consumedEnergy - minEnergy)/(maxEnergy - minEnergy)) else 0.0 endif;

var float: ob = normalized_makespan + normalized_consumedEnergy;
% Función objetivo
solve minimize ob;

% Salida
output ["Objective function: \(normalized_makespan) + \(normalized_consumedEnergy) = \(ob) \n"];
output ["Objective function : M-\(makespan) + E-\(consumedEnergy) = \(ob) \n"];
output ["Start Times: \(start_time)\n"];
output ["Speed Scaling; \(SpeedScaling)\n"];