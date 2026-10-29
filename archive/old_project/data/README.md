# Dataset Record — xAI Viral Attribution

## Phase 1: Pipeline validation (complete)

- VMR release:        ICTV MSL41 v1 (2026-03-20)
- Retrieval:          NCBI efetch via VMR "Virus GENBANK accession" column
- Sequences:          20 (VMR exemplars only)
- CD-HIT-EST:         4.8.1, -c 0.90 -n 8
- Clusters:           17 (15 singletons, 2 multi-member)
- Split:              species-level, seed 42
- Leak check:         PASS (cd-hit-2d, 0 matches)

## Phase 2: Balanced stratified benchmark (complete)

- Sampling seed:      42
- CAP per species:    87 (binding constraint: Lassa, 87 available)
- Stratification:     collection year (all species);
                      serotype (Dengue); lineage (SARS-CoV-2)
- Species included:   9 (Hendra excluded — 20 complete genomes only,
                      Paramyxoviridae represented by Nipah)
- Families:           5 (Coronaviridae, Filoviridae, Arenaviridae,
                      Paramyxoviridae, Flaviviridae)
- Sequences:          783 (87 x 9, exact, all species)
- CD-HIT-EST:         4.8.1, -c 0.90 -n 8
- Clusters:           76
- Split:              species-level
    train (5 species): Lassa, Marburg, Nipah, SARS-CoV-2, Zika  = 435
    val   (2 species): MERS-CoV, Yellow fever                    = 174
    test  (2 species): Dengue, Ebola                             = 174
- Leak check:         PASS (cd-hit-est-2d, 0 matches at 90%)

## Known limitations

- Held-out generalisation is within-family: Ebola (Filoviridae)
  held out with Marburg in train; Dengue (Flaviviridae) held out
  with Zika and Yellow fever in train. Paramyxoviridae and
  Arenaviridae are in-distribution only.
- Per-family plausibility is reportable for Filoviridae and
  Flaviviridae; the other three families get in-distribution scores.
- Lassa contributed its entire complete-genome population (87);
  retrying failed fetches is not possible if any accession is
  retired from NCBI.
