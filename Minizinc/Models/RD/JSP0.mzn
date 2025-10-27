% Conjuntos
set of int: JOBS;
set of int: MACHINES;
int: SPEED;

% Datos
array[JOBS,MACHINES,1..SPEED] of int: time;
array[JOBS,MACHINES,1..SPEED] of int: energy;
array[JOBS,MACHINES] of int: precedence;

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
var int: makespan                     = max([start_time[j,m] + time[j,m,SpeedScaling[j,m]] | m in MACHINES, j in JOBS]);
var int: consumedEnergy               = sum(m in MACHINES, j in JOBS) (energy[j,m,SpeedScaling[j,m]]);

var int: ob = makespan + consumedEnergy;
% Función objetivo
solve minimize ob;

% Salida
output ["Objective function: \(makespan) + \(consumedEnergy) = \(ob) \n"];
output ["Start Times: \(start_time)\n"];
output ["Speed Scaling; \(SpeedScaling)\n"];