JSP machine-precedence analysis results
======================================

Input: .

Parsed instances: 162
Parse errors: 2
Families: abz, ft, la, orb, swv, ta, yn

What was analysed?
------------------
The route structure of each job:
    job j: machine_1 -> machine_2 -> ... -> machine_m

Main files
----------
- precedence_summary_by_file.csv
    One row per instance. Includes route diversity, position entropy, pairwise bias,
    adjacent graph density, transitive graph density, etc.

- precedence_summary_by_family.csv
    Aggregated statistics per benchmark family: abz, ft, la, orb, swv, ta, yn...

- precedence_summary_by_family_size.csv
    Aggregated statistics by family and size. This is often the safest comparison
    because some families mix several numbers of machines.

- machine_position_by_family_size.csv
    Frequency of machine m appearing at route position k.

- adjacent_machine_precedence_by_family_size.csv
    Frequency of immediate transitions m_a -> m_b.

- pairwise_machine_precedence_by_family_size.csv
    For each pair of machines {a,b}, estimates P(a before b).
    Values close to 0.5 indicate little directional bias.
    Values close to 0 or 1 indicate a strong/near-deterministic precedence relation.

Key metrics
-----------
- route_diversity_ratio:
    distinct machine routes / number of jobs.
    1.0 means every job has a different route.

- mean_position_entropy_norm:
    Entropy of machine positions, normalised to [0,1].
    Near 1.0 means machines are spread almost uniformly across positions.

- mean_abs_pairwise_bias:
    Mean |P(a before b) - 0.5| across machine pairs.
    Near 0 means no systematic machine precedence direction.
    Higher values mean stronger structural bias.

- adjacent_edge_density:
    Fraction of possible directed immediate machine transitions observed.

- transitive_edge_density:
    Fraction of possible directed transitive machine precedences observed.

- bidirectional_pair_ratio:
    Fraction of unordered machine pairs for which both directions appear in different jobs.

Recommended interpretation
--------------------------
Use family_size outputs for rigorous comparisons. Family-only outputs are useful,
but may mix different numbers of machines, especially in la, ta, swv, ft, etc.
